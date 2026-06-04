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
#include "arm/arm.hpp"

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
static const int   CHASE_SPEED_FAR  = 40;   // 语义速度(1~100)，Motor 层映射到实际PWM
static const int   CHASE_SPEED_NEAR = 3;   // 接近时速度

static const float AREA_FAR         = 0.02f;
static const float AREA_NEAR        = 0.35f;
static const float AREA_BRAKE       = 0.20f;  // 进入制动区（提前减速）

static const int   BRAKE_SPEED      = 3;     // 制动区速度（语义）
static const float AREA_STOP        = 0.28f;  // 触发停止
static const float AREA_REVERSE     = 0.50f;  // 球太近 → 后退
static const int   REVERSE_SPEED    = 15;     // 后退速度（语义）

static const float AREA_STOP_EXIT   = 0.20f;  // 重新追球阈值
static const int   STOP_CONFIRM_CNT = 4;      // 确认帧数
static const int   BRAKE_PULSE_US   = 350000; // 持续制动 350ms
static const int   STOP_CENTER_OFFSET = 90;   // px: 停止目标偏右（夹爪在右侧）

static const float K_TURN              = 25.0f;
static const int   MAX_TURN_BIAS_FAR   = 5;     // 远距离差速偏置上限（语义）
static const int   MAX_TURN_BIAS_NEAR  = 10;     // 近距离轴转上限（语义）
static const int   CENTER_DEAD_ZONE    = 15;     // px: 追球死区
static const int   STOP_CENTER_ZONE    = 5;     // px: 停止确认死区
static const int   ALIGN_PIVOT_SPD     = 15;     // 对准轴转最大速度（语义）
static const int   ALIGN_PIVOT_MIN     = 3;     // 对准轴转最小速度（语义，Motor层保证可动）

static const int   SEARCH_FRAMES    = 25;
static const int   SEARCH_PIVOT_SPD = 10;     // 查找轴转速度（语义）

static const int   ALIGN_STALL_FRAMES  = 20;    // 连续N帧偏移无变化 → 判定卡死
static const int   ALIGN_STALL_MOVE_PX = 10;    // 变化小于此值视为未移动
static const int   ALIGN_KICK_SPD      = 35;    // 卡死踢出速度（语义）
static const int   ALIGN_KICK_US       = 180000;// 踢出持续时间 180ms

// ── Globals for signal handler ────────────────────────────────────────────────
static Motor*              g_motor      = nullptr;
static UvcCapture*         g_capture    = nullptr;
static rknn_app_context_t* g_rknn_ctx   = nullptr;
static Arm*                g_arm        = nullptr;
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
    LOGI("  %s <model.rknn> [uart_dev] [uvc_device_index] [arm_dev]", prog);
    LOGI("  Example: %s tennis.rknn /dev/ttyS3 0 /dev/ttyUSB1", prog);
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
    const char* arm_dev    = (argc >= 5) ? argv[4] : "/dev/ttyUSB1";

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    Motor motor(MotorDriverType::UART, uart_dev);
    g_motor = &motor;
    // 最小可动速度，低于此值电机转不动。换车时通过 test-motor 测试后在此修改
    motor.set_min_speed(15);
    LOGI("Motor initialized (UART %s)  min_speed=%d", uart_dev, motor.get_min_speed());

    Arm arm(arm_dev);
    g_arm = &arm;
    arm.grab_pos();  // 归位到初始姿势
    LOGI("Arm initialized (%s)", arm_dev);

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
    int  align_cnt        = 0;      // frames spent in ALIGN (timeout → force grab)

    // ── ALIGN stall detection ─────────────────────────────────────────────────
    // 记录最近 ALIGN_STALL_FRAMES 帧的 offset，判断是否卡死
    static const int STALL_BUF = 30;
    int  align_off_buf[STALL_BUF] = {};
    int  align_off_head = 0;
    bool align_kicking  = false;    // 正在执行踢出脉冲

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

            // ── Stop condition: close enough AND at target offset, confirm N frames ───
            int stop_off = offset - STOP_CENTER_OFFSET;  // 目标：球在中心偏右
            if (area_ratio >= AREA_STOP && abs(stop_off) <= STOP_CENTER_ZONE) {
                stop_confirm_cnt++;
                align_cnt = 0;
                align_off_head = 0;
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
                    printf("[STATE] STOPPED  area=%.3f  -> GRAB\n", area_ratio);
                    dup2(g_devnull, STDERR_FILENO);
                    // ── 执行抓球 ──────────────────────────────────────────────
                    if (g_arm) g_arm->grab();
                } else {
                    motor.brake();
                }
                t_ctrl_acc += elapsed_us(t_stage);
                continue;
            } else if (area_ratio >= AREA_STOP && abs(stop_off) > STOP_CENTER_ZONE) {
                // Close but not at target offset → proportional pivot to align
                stop_confirm_cnt = 0;
                align_cnt++;

                // ── 卡死检测：记录 offset 历史，若近N帧无移动则踢出 ──────────
                align_off_buf[align_off_head % STALL_BUF] = offset;
                align_off_head++;
                bool stalled = false;
                if (align_cnt >= ALIGN_STALL_FRAMES) {
                    int mn = align_off_buf[0], mx = align_off_buf[0];
                    int check = std::min(align_cnt, STALL_BUF);
                    for (int i = 1; i < check; i++) {
                        int v = align_off_buf[i];
                        if (v < mn) mn = v;
                        if (v > mx) mx = v;
                    }
                    stalled = (mx - mn) < ALIGN_STALL_MOVE_PX;
                }

                if (stalled) {
                    int kick = (stop_off > 0) ? ALIGN_KICK_SPD : -ALIGN_KICK_SPD;
                    dup2(g_saved_stderr, STDERR_FILENO);
                    printf("[STATE] ALIGN_KICK  area=%.3f off=%3d stop_off=%3d  kick=%d  %dms\n",
                           area_ratio, offset, stop_off, kick, ALIGN_KICK_US/1000);
                    dup2(g_devnull, STDERR_FILENO);
                    struct timeval tk; gettimeofday(&tk, nullptr);
                    while (elapsed_us(tk) < ALIGN_KICK_US) {
                        motor.drive(kick, -kick);
                        usleep(20000);
                    }
                    motor.brake();
                    // 重置历史，避免连续踢
                    align_off_head = 0;
                    align_cnt = 0;
                    t_ctrl_acc += elapsed_us(t_stage);
                    continue;
                }

                float t = std::min(1.0f, (float)abs(stop_off) / (float)half_w);
                int pivot_spd = (int)(ALIGN_PIVOT_MIN + t * (ALIGN_PIVOT_SPD - ALIGN_PIVOT_MIN));
                int pivot = (stop_off > 0) ? pivot_spd : -pivot_spd;
                motor.drive(pivot, -pivot);
                dup2(g_saved_stderr, STDERR_FILENO);
                printf("[STATE] ALIGN  area=%.3f off=%3d stop_off=%3d  pivot=%d  [%d]\n",
                       area_ratio, offset, stop_off, pivot, align_cnt);
                dup2(g_devnull, STDERR_FILENO);
                t_ctrl_acc += elapsed_us(t_stage);
                continue;
            } else {
                stop_confirm_cnt = 0;
                align_cnt = 0;
                align_off_head = 0;
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
