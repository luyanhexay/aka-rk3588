// Source: aka-sg2002/tennis.cpp - ported to rk3588 / RKNN YOLOv8
// Description: chase-only state machine
//   - UVC camera (MJPEG) -> libjpeg-turbo decode -> RKNN YOLOv8 inference
//   - Motor driver abstraction: UART (ESP32-C3) or PWM
//   - Smooth continuous differential steering (no stop-and-turn)
//   - Ctrl-C safe exit

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <algorithm>
#include <vector>

#include <turbojpeg.h>

#include "logger.hpp"
#include "motor/motor.hpp"
#include "capture/uvc_capture.hpp"
#include "detect/detect.hpp"
#include "test_cmds.hpp"

// ── Build-time tunables ───────────────────────────────────────────────────────
#define ENABLE_SAVE_IMAGE 0
#define SKIP_FRAMES       0  // Process every Nth frame (0=process all)

// ── Camera / model parameters ─────────────────────────────────────────────────
static const int FRAME_WIDTH  = 640;
static const int FRAME_HEIGHT = 480;
static const int MODEL_W      = 640;
static const int MODEL_H      = 640;

// ── Control parameters ────────────────────────────────────────────────────────
// Smooth differential steering:
//   left_speed  = base_speed + bias
//   right_speed = base_speed - bias
// bias = K_TURN * (offset / half_width), clamped to [-MAX_TURN_BIAS, MAX_TURN_BIAS]
// offset>0 (ball on right) → bias>0 → left faster → turn right
static const int   CHASE_SPEED_FAR  = 40;   // was 60
static const int   CHASE_SPEED_NEAR = 14;   // was 22

static const float AREA_FAR         = 0.02f;
static const float AREA_NEAR        = 0.35f;
static const float AREA_BRAKE       = 0.20f;  // 进入制动区（提前减速）

static const int   BRAKE_SPEED      = 14;     // 制动区速度
static const float AREA_STOP        = 0.28f;  // 触发停止
static const float AREA_REVERSE     = 0.50f;  // 球太近 → 后退
static const int   REVERSE_SPEED    = 20;     // 后退速度

static const float AREA_STOP_EXIT   = 0.20f;  // 重新追球阈值
static const int   STOP_CONFIRM_CNT = 4;      // 1帧立刻停
static const int   BRAKE_PULSE_US   = 350000; // 持续制动 400ms

static const float K_TURN              = 25.0f;  // 转向增益
static const int   MAX_TURN_BIAS_FAR   = 20;     // 远距离: < CHASE_SPEED_FAR(35) → 纯差速
static const int   MAX_TURN_BIAS_NEAR  = 50;     // 近距离: > BRAKE_SPEED(13) → 轴转
static const int   CENTER_DEAD_ZONE    = 40;     // px: 追球死区
static const int   STOP_CENTER_ZONE    = 20;     // px: 停止确认死区（更严格）
static const int   ALIGN_PIVOT_SPD     = 35;     // 对准时轴转最大速度
static const int   ALIGN_PIVOT_MIN     = 14;     // 对准时轴转最小速度（克服静摩擦）

static const int   SEARCH_FRAMES    = 15;     // 球消失后继续转向查找的帧数 (~0.75s @20fps)
static const int   SEARCH_PIVOT_SPD = 22;     // 查找时轴转速度

// ── Globals for signal handler ────────────────────────────────────────────────
static Motor*              g_motor      = nullptr;
static UvcCapture*         g_capture    = nullptr;
static rknn_app_context_t* g_rknn_ctx   = nullptr;
static int                 g_saved_stderr = -1;
static int                 g_devnull    = -1;

static void cleanup_and_exit() {
    if (g_motor)    g_motor->standby();
    if (g_capture)  g_capture->close();
    if (g_rknn_ctx) detect_deinit(g_rknn_ctx);
    if (g_saved_stderr >= 0 && g_devnull >= 0)
        dup2(g_saved_stderr, STDERR_FILENO);
}

static void signal_handler(int /*sig*/) {
    cleanup_and_exit();
    exit(0);
}

// ── Timing helper ─────────────────────────────────────────────────────────────
static long elapsed_us(const struct timeval& start) {
    struct timeval now; gettimeofday(&now, nullptr);
    return (now.tv_sec - start.tv_sec) * 1000000L + (now.tv_usec - start.tv_usec);
}

// ── JPEG decode -> RGB letterbox ─────────────────────────────────────────────
static int decode_mjpeg(const uint8_t* jpeg_data, size_t jpeg_len,
                        uint8_t* rgb_out, int out_w, int out_h,
                        int* pad_x = nullptr, int* pad_y = nullptr,
                        float* scale_out = nullptr,
                        long* t_header = nullptr, long* t_decomp = nullptr,
                        long* t_copy = nullptr)
{
    struct timeval t0;
    tjhandle tj = tjInitDecompress();
    if (!tj) return -1;

    gettimeofday(&t0, nullptr);
    int w, h, subsamp, colorspace;
    if (tjDecompressHeader3(tj, jpeg_data, jpeg_len, &w, &h, &subsamp, &colorspace) < 0)
        { tjDestroy(tj); return -1; }
    if (t_header) *t_header = elapsed_us(t0);

    float scale = std::min((float)out_w / w, (float)out_h / h);
    int new_w = (int)(w * scale + 0.5f);
    int new_h = (int)(h * scale + 0.5f);
    int off_x = (out_w - new_w) / 2;
    int off_y = (out_h - new_h) / 2;
    if (pad_x)     *pad_x     = off_x;
    if (pad_y)     *pad_y     = off_y;
    if (scale_out) *scale_out = scale;

    memset(rgb_out, 114, out_w * out_h * 3);
    uint8_t* tmp = (uint8_t*)malloc(new_w * new_h * 3);
    if (!tmp) { tjDestroy(tj); return -1; }

    gettimeofday(&t0, nullptr);
    int ret = tjDecompress2(tj, jpeg_data, jpeg_len,
                            tmp, new_w, 0, new_h, TJPF_RGB, TJFLAG_FASTDCT);
    tjDestroy(tj);
    if (t_decomp) *t_decomp = elapsed_us(t0);

    if (ret < 0) { free(tmp); return -1; }

    gettimeofday(&t0, nullptr);
    for (int y = 0; y < new_h; y++)
        memcpy(rgb_out + ((y + off_y) * out_w + off_x) * 3,
               tmp     + y * new_w * 3, new_w * 3);
    if (t_copy) *t_copy = elapsed_us(t0);

    free(tmp);
    return 0;
}

// ── Dynamic base speed ────────────────────────────────────────────────────────
static int base_speed(float area_ratio) {
    if (area_ratio >= AREA_BRAKE) return BRAKE_SPEED;  // 制动区: 锁最低速
    float t = (area_ratio - AREA_FAR) / (AREA_BRAKE - AREA_FAR);
    t = std::max(0.0f, std::min(1.0f, t));
    return (int)(CHASE_SPEED_FAR + t * (BRAKE_SPEED - CHASE_SPEED_FAR));
}

// ── Usage ─────────────────────────────────────────────────────────────────────
static void usage(const char* prog) {
    LOGI("Usage:");
    LOGI("  %s <model.rknn> [uart_dev] [uvc_device_index]", prog);
    LOGI("  Example: %s tennis.rknn /dev/ttyS3 0", prog);
    LOGI("  %s test-uvc   [uvc_index]               -- capture one frame -> capture.jpg", prog);
    LOGI("  %s test-yolo  <model.rknn> [uvc_index]  -- detect one frame  -> result.jpg", prog);
    LOGI("  %s test-motor [uart_dev] [speed=N]       -- motor test", prog);
    LOGI("  %s test-arm   [uart_dev] <cmd|a0 a1 a2>  -- arm servo test (default /dev/ttyUSB1)", prog);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 2) { usage(argv[0]); return 1; }

    if (strcmp(argv[1], "test-uvc") == 0) {
        int idx = (argc >= 3) ? atoi(argv[2]) : 0;
        return cmd_test_uvc(idx);
    }
    if (strcmp(argv[1], "test-yolo") == 0) {
        if (argc < 3) { LOGE("test-yolo requires <model.rknn>"); usage(argv[0]); return 1; }
        int idx = (argc >= 4) ? atoi(argv[3]) : 0;
        return cmd_test_yolo(argv[2], idx);
    }
    if (strcmp(argv[1], "test-motor") == 0) {
        const char* dev = (argc >= 3) ? argv[2] : "/dev/ttyUSB0";
        return cmd_test_motor(dev, argc, argv);
    }
    if (strcmp(argv[1], "test-arm") == 0) {
        const char* dev = (argc >= 3) ? argv[2] : "/dev/ttyUSB1";
        return cmd_test_arm(dev, argc, argv);
    }

    const char* model_path = argv[1];
    const char* uart_dev   = (argc >= 3) ? argv[2] : "/dev/ttyS3";
    int         uvc_index  = (argc >= 4) ? atoi(argv[3]) : 0;

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    Motor motor(MotorDriverType::UART, uart_dev);
    g_motor = &motor;
    LOGI("Motor initialized (UART %s)", uart_dev);

    UvcCapture capture;
    g_capture = &capture;
    if (capture.open(uvc_index, FRAME_WIDTH, FRAME_HEIGHT, 30) != 0) {
        LOGE("Failed to open UVC device %d", uvc_index); return 1;
    }
    LOGI("Camera opened (%dx%d)", FRAME_WIDTH, FRAME_HEIGHT);

    rknn_app_context_t rknn_ctx;
    g_rknn_ctx = &rknn_ctx;
    if (detect_init(model_path, &rknn_ctx) != 0) {
        LOGE("Failed to load model: %s", model_path); return 1;
    }
    int model_w = rknn_ctx.model_width;
    int model_h = rknn_ctx.model_height;
    LOGI("Model input %dx%d", model_w, model_h);

    // Suppress rknn runtime stderr spam
    g_devnull      = open("/dev/null", O_WRONLY);
    g_saved_stderr = dup(STDERR_FILENO);
    dup2(g_devnull, STDERR_FILENO);

    const size_t MJPEG_BUF = 1024 * 1024;
    uint8_t* mjpeg_buf = (uint8_t*)malloc(MJPEG_BUF);
    uint8_t* rgb_buf   = (uint8_t*)malloc(model_w * model_h * 3);
    if (!mjpeg_buf || !rgb_buf) { LOGE("OOM"); return 1; }

    dup2(g_saved_stderr, STDERR_FILENO);
    LOGI("Warming up camera (skip 20 frames)...");
    dup2(g_devnull, STDERR_FILENO);
    for (int i = 0; i < 20; i++) capture.getFrame(mjpeg_buf, MJPEG_BUF, 500);

    int  frame_idx    = 0;
    int  proc_cnt     = 0;
    long t_decode_acc = 0, t_infer_acc = 0, t_ctrl_acc = 0;

    // ── Last-seen tracking for search-after-loss ──────────────────────────────
    int  last_offset      = 0;
    int  last_seen_frame  = -999;

    // ── Stop state ────────────────────────────────────────────────────────────
    bool stopped          = false;  // locked stop state
    int  stop_confirm_cnt = 0;      // consecutive frames meeting stop condition

    // ── Chase loop ────────────────────────────────────────────────────────────
    while (true) {
        struct timeval t_start, t_stage;
        gettimeofday(&t_start, nullptr);
        frame_idx++;

        int jpeg_len = capture.getFrame(mjpeg_buf, MJPEG_BUF, 200);
        if (jpeg_len <= 0) {
            dup2(g_saved_stderr, STDERR_FILENO);
            LOGW("[Frame %d] No frame (timeout)", frame_idx);
            dup2(g_devnull, STDERR_FILENO);
            usleep(10000);
            continue;
        }

#if SKIP_FRAMES > 0
        if ((frame_idx % (SKIP_FRAMES + 1)) != 0) continue;
#endif
        proc_cnt++;

        long th=0, td=0, tc=0;
        int  lb_x=0, lb_y=0;
        float lb_sc=1.0f;
        if (decode_mjpeg(mjpeg_buf, jpeg_len, rgb_buf, model_w, model_h,
                         &lb_x, &lb_y, &lb_sc, &th, &td, &tc) != 0) continue;
        t_decode_acc += th + td + tc;

        long ti=0, tr=0, to=0, tp=0;
        std::vector<detection> dets;
        detect_run(&rknn_ctx, rgb_buf, model_w, model_h,
                   FRAME_WIDTH, FRAME_HEIGHT, lb_x, lb_y, lb_sc,
                   0.5f, 0.45f, dets, &ti, &tr, &to, &tp);
        t_infer_acc += ti + tr + to + tp;

        // ── Smooth differential steering ──────────────────────────────────────
        gettimeofday(&t_stage, nullptr);
        const int half_w = FRAME_WIDTH / 2;

        if (!dets.empty()) {
            int best = 0;
            for (int i = 1; i < (int)dets.size(); i++)
                if (dets[i].bbox.w * dets[i].bbox.h >
                    dets[best].bbox.w * dets[best].bbox.h)
                    best = i;

            const box& b     = dets[best].bbox;
            float area_ratio = (b.w * b.h) / (float)(FRAME_WIDTH * FRAME_HEIGHT);
            int   ball_cx    = (int)b.x;
            int   offset     = ball_cx - half_w;   // <0 = ball on left

            // Update last-seen tracking
            last_offset     = offset;
            last_seen_frame = frame_idx;

            int spd  = base_speed(area_ratio);

            // ── 区域标签（用于日志）───────────────────────────────────────────
            const char* zone = (area_ratio >= AREA_REVERSE) ? "REVERSE" :
                               (area_ratio >= AREA_STOP)    ? "STOP"    :
                               (area_ratio >= AREA_BRAKE)   ? "BRAKE"   :
                               (area_ratio >= AREA_NEAR)    ? "NEAR"    :
                               (area_ratio >= AREA_FAR)     ? "FAR"     : "LOST";

            // ── Stop state: exit only when ball moves far away ────────────────
            if (stopped) {
                if (area_ratio >= AREA_REVERSE) {
                    // 即使已停止，球太近也要后退
                    stopped = false;
                    stop_confirm_cnt = 0;
                    dup2(g_saved_stderr, STDERR_FILENO);
                    printf("[STATE] STOPPED->REVERSE  area=%.3f\n", area_ratio);
                    dup2(g_devnull, STDERR_FILENO);
                    // fall through to REVERSE logic below
                } else if (area_ratio < AREA_STOP_EXIT) {
                    stopped = false;
                    stop_confirm_cnt = 0;
                    dup2(g_saved_stderr, STDERR_FILENO);
                    printf("[STATE] RESUME  area=%.3f zone=%s\n", area_ratio, zone);
                    dup2(g_devnull, STDERR_FILENO);
                } else {
                    motor.standby();
                    t_ctrl_acc += elapsed_us(t_stage);
                    continue;
                }
            }

            // ── 后退: 球占画面过大 ────────────────────────────────────────────
            if (area_ratio >= AREA_REVERSE) {
                int rev_left  = (offset > CENTER_DEAD_ZONE)  ? -REVERSE_SPEED + 5 :
                                (offset < -CENTER_DEAD_ZONE) ? -REVERSE_SPEED - 5 :
                                -REVERSE_SPEED;
                int rev_right = (offset > CENTER_DEAD_ZONE)  ? -REVERSE_SPEED - 5 :
                                (offset < -CENTER_DEAD_ZONE) ? -REVERSE_SPEED + 5 :
                                -REVERSE_SPEED;
                motor.drive(rev_left, rev_right);
                dup2(g_saved_stderr, STDERR_FILENO);
                printf("[STATE] REVERSE  area=%.3f off=%d  L=%d R=%d\n",
                       area_ratio, offset, rev_left, rev_right);
                dup2(g_devnull, STDERR_FILENO);
                t_ctrl_acc += elapsed_us(t_stage);
                continue;
            }

            // ── Stop condition: close enough AND centered, confirm N frames ───
            if (area_ratio >= AREA_STOP && abs(offset) <= STOP_CENTER_ZONE) {
                stop_confirm_cnt++;
                if (stop_confirm_cnt >= STOP_CONFIRM_CNT) {
                    dup2(g_saved_stderr, STDERR_FILENO);
                    printf("[STATE] BRAKING  area=%.3f off=%d  %dms...\n",
                           area_ratio, offset, BRAKE_PULSE_US/1000);
                    dup2(g_devnull, STDERR_FILENO);
                    struct timeval tb; gettimeofday(&tb, nullptr);
                    while (elapsed_us(tb) < BRAKE_PULSE_US) {
                        motor.brake();
                        usleep(20000);
                    }
                    motor.standby();
                    stopped = true;
                    dup2(g_saved_stderr, STDERR_FILENO);
                    printf("[STATE] STOPPED  area=%.3f\n", area_ratio);
                    dup2(g_devnull, STDERR_FILENO);
                } else {
                    motor.brake();
                }
                t_ctrl_acc += elapsed_us(t_stage);
                continue;
            } else if (area_ratio >= AREA_STOP && abs(offset) > STOP_CENTER_ZONE) {
                // Close but not centered → proportional pivot to align (avoid oscillation)
                stop_confirm_cnt = 0;
                float t = std::min(1.0f, (float)abs(offset) / (float)half_w);
                int pivot_spd = (int)(ALIGN_PIVOT_MIN + t * (ALIGN_PIVOT_SPD - ALIGN_PIVOT_MIN));
                int pivot = (offset > 0) ? pivot_spd : -pivot_spd;
                motor.drive(pivot, -pivot);
                dup2(g_saved_stderr, STDERR_FILENO);
                printf("[STATE] ALIGN  area=%.3f off=%3d  pivot=%d\n",
                       area_ratio, offset, pivot);
                dup2(g_devnull, STDERR_FILENO);
                t_ctrl_acc += elapsed_us(t_stage);
                continue;
            } else {
                stop_confirm_cnt = 0;
            }

            // ── Chase ─────────────────────────────────────────────────────────
            int bias = (abs(offset) <= CENTER_DEAD_ZONE)
                       ? 0
                       : (int)(K_TURN * offset / (float)half_w);
            int max_bias = (area_ratio >= AREA_BRAKE) ? MAX_TURN_BIAS_NEAR : MAX_TURN_BIAS_FAR;
            bias = std::max(-max_bias, std::min(max_bias, bias));

            // Ball on right (bias>0) → left wheel faster → turn right
            int left_spd  = std::max(-100, std::min(100, spd + bias));
            int right_spd = std::max(-100, std::min(100, spd - bias));

            motor.drive(left_spd, right_spd);

            dup2(g_saved_stderr, STDERR_FILENO);
            long frame_us = elapsed_us(t_start);
            printf("[STATE] CHASE zone=%-6s area=%.3f off=%3d bias=%3d  L=%3d R=%3d  fps=%.1f\n",
                   zone, area_ratio, offset, bias, left_spd, right_spd, 1e6f / frame_us);
            dup2(g_devnull, STDERR_FILENO);

        } else {
            int frames_lost = frame_idx - last_seen_frame;
            if (last_seen_frame >= 0 && frames_lost <= SEARCH_FRAMES) {
                int pivot = (last_offset >= 0) ? SEARCH_PIVOT_SPD : -SEARCH_PIVOT_SPD;
                motor.drive(pivot, -pivot);
                dup2(g_saved_stderr, STDERR_FILENO);
                printf("[STATE] SEARCH lost=%d/%d  pivot=%s\n",
                       frames_lost, SEARCH_FRAMES, last_offset >= 0 ? "R" : "L");
                dup2(g_devnull, STDERR_FILENO);
            } else {
                motor.standby();
                dup2(g_saved_stderr, STDERR_FILENO);
                printf("[STATE] LOST   waiting...\n");
                dup2(g_devnull, STDERR_FILENO);
            }
        }

        t_ctrl_acc += elapsed_us(t_stage);
    }

    free(mjpeg_buf);
    free(rgb_buf);
    cleanup_and_exit();
    return 0;
}
