// test_cmds.cpp — debug sub-commands (test-uvc / test-yolo / test-motor / test-arm)
#include "test_cmds.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <vector>

#include <turbojpeg.h>

#include "logger.hpp"
#include "capture/uvc_capture.hpp"
#include "detect/detect.hpp"
#include "motor/motor.hpp"
#include "arm/arm.hpp"

static const int FRAME_WIDTH  = 640;
static const int FRAME_HEIGHT = 480;

// ── helpers (local copies) ────────────────────────────────────────────────────
static long _elapsed_us(const struct timeval& s) {
    struct timeval now; gettimeofday(&now, nullptr);
    return (now.tv_sec - s.tv_sec) * 1000000L + (now.tv_usec - s.tv_usec);
}

static int _decode_mjpeg(const uint8_t* jpeg, size_t len,
                         uint8_t* out, int ow, int oh,
                         int* pad_x, int* pad_y, float* scale_out)
{
    tjhandle tj = tjInitDecompress();
    if (!tj) return -1;
    int w, h, subsamp, cs;
    if (tjDecompressHeader3(tj, jpeg, len, &w, &h, &subsamp, &cs) < 0)
        { tjDestroy(tj); return -1; }

    float sc = std::min((float)ow / w, (float)oh / h);
    int nw = (int)(w * sc + 0.5f), nh = (int)(h * sc + 0.5f);
    int ox = (ow - nw) / 2, oy = (oh - nh) / 2;
    if (pad_x)    *pad_x    = ox;
    if (pad_y)    *pad_y    = oy;
    if (scale_out)*scale_out = sc;

    memset(out, 114, ow * oh * 3);
    uint8_t* tmp = (uint8_t*)malloc(nw * nh * 3);
    if (!tmp) { tjDestroy(tj); return -1; }
    int ret = tjDecompress2(tj, jpeg, len, tmp, nw, 0, nh, TJPF_RGB, TJFLAG_FASTDCT);
    tjDestroy(tj);
    if (ret < 0) { free(tmp); return -1; }
    for (int y = 0; y < nh; y++)
        memcpy(out + ((y + oy) * ow + ox) * 3, tmp + y * nw * 3, nw * 3);
    free(tmp);
    return 0;
}

static int _save_jpeg(const char* path, const uint8_t* data, size_t len) {
    FILE* f = fopen(path, "wb");
    if (!f) { LOGE("Cannot open %s", path); return -1; }
    fwrite(data, 1, len, f);
    fclose(f);
    LOGI("Saved %zu bytes → %s", len, path);
    return 0;
}

static void _draw_box(uint8_t* rgb, int iw, int ih,
                      int x, int y, int w, int h,
                      uint8_t r, uint8_t g, uint8_t b)
{
    auto cl = [](int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); };
    int x0=cl(x,0,iw-1), y0=cl(y,0,ih-1), x1=cl(x+w-1,0,iw-1), y1=cl(y+h-1,0,ih-1);
    for (int px=x0;px<=x1;px++) {
        uint8_t* t=rgb+(y0*iw+px)*3; uint8_t* bt=rgb+(y1*iw+px)*3;
        t[0]=bt[0]=r; t[1]=bt[1]=g; t[2]=bt[2]=b;
    }
    for (int py=y0;py<=y1;py++) {
        uint8_t* l=rgb+(py*iw+x0)*3; uint8_t* ri=rgb+(py*iw+x1)*3;
        l[0]=ri[0]=r; l[1]=ri[1]=g; l[2]=ri[2]=b;
    }
}

static int _save_rgb_as_jpeg(const char* path, const uint8_t* rgb, int w, int h) {
    tjhandle tj = tjInitCompress();
    if (!tj) return -1;
    unsigned char* buf = nullptr; unsigned long bl = 0;
    int ret = tjCompress2(tj, rgb, w, 0, h, TJPF_RGB,
                          &buf, &bl, TJSAMP_420, 90, TJFLAG_FASTDCT);
    tjDestroy(tj);
    if (ret < 0) return -1;
    FILE* f = fopen(path, "wb"); if (!f) { tjFree(buf); return -1; }
    fwrite(buf, 1, bl, f); fclose(f); tjFree(buf);
    LOGI("Saved %lu bytes → %s", bl, path);
    return 0;
}

// ── test-uvc ──────────────────────────────────────────────────────────────────
int cmd_test_uvc(int uvc_index)
{
    UvcCapture capture;
    if (capture.open(uvc_index, FRAME_WIDTH, FRAME_HEIGHT, 30) != 0) {
        LOGE("Failed to open UVC device %d", uvc_index); return 1;
    }
    LOGI("Camera opened (%dx%d), warming up 20 frames...", FRAME_WIDTH, FRAME_HEIGHT);
    const size_t BUF = 1024 * 1024;
    uint8_t* buf = (uint8_t*)malloc(BUF);
    for (int i = 0; i < 20; i++) capture.getFrame(buf, BUF, 500);
    int len = capture.getFrame(buf, BUF, 2000);
    capture.close();
    if (len <= 0) { LOGE("No frame received"); free(buf); return 1; }
    LOGI("Got MJPEG frame: %d bytes", len);
    _save_jpeg("capture.jpg", buf, len);
    free(buf);
    return 0;
}

// ── test-yolo ─────────────────────────────────────────────────────────────────
int cmd_test_yolo(const char* model_path, int uvc_index)
{
    UvcCapture capture;
    if (capture.open(uvc_index, FRAME_WIDTH, FRAME_HEIGHT, 30) != 0) {
        LOGE("Failed to open UVC device %d", uvc_index); return 1;
    }
    rknn_app_context_t ctx;
    if (detect_init(model_path, &ctx) != 0) {
        LOGE("Failed to load model: %s", model_path); capture.close(); return 1;
    }
    int mw = ctx.model_width, mh = ctx.model_height;
    LOGI("Model input %dx%d", mw, mh);

    const size_t MBUF = 1024 * 1024;
    uint8_t* mjpeg_buf = (uint8_t*)malloc(MBUF);
    LOGI("Warming up camera (20 frames)...");
    for (int i = 0; i < 20; i++) capture.getFrame(mjpeg_buf, MBUF, 500);
    int jpeg_len = capture.getFrame(mjpeg_buf, MBUF, 2000);
    capture.close();
    if (jpeg_len <= 0) {
        LOGE("No frame"); free(mjpeg_buf); detect_deinit(&ctx); return 1;
    }
    _save_jpeg("capture.jpg", mjpeg_buf, jpeg_len);

    uint8_t* rgb = (uint8_t*)malloc(mw * mh * 3);
    int px=0, py=0; float sc=1.0f;
    if (_decode_mjpeg(mjpeg_buf, jpeg_len, rgb, mw, mh, &px, &py, &sc) != 0) {
        LOGE("JPEG decode failed"); free(mjpeg_buf); free(rgb); detect_deinit(&ctx); return 1;
    }
    free(mjpeg_buf);
    LOGI("Letterbox: scale=%.4f pad=(%d,%d)", sc, px, py);

    // Display buffer at native resolution (reverse letterbox)
    uint8_t* disp = (uint8_t*)malloc(FRAME_WIDTH * FRAME_HEIGHT * 3);
    for (int y = 0; y < FRAME_HEIGHT; y++) {
        int sy = std::max(0, std::min((int)(y * sc) + py, mh - 1));
        for (int x = 0; x < FRAME_WIDTH; x++) {
            int sx = std::max(0, std::min((int)(x * sc) + px, mw - 1));
            const uint8_t* s = rgb + (sy * mw + sx) * 3;
            uint8_t*       d = disp + (y * FRAME_WIDTH + x) * 3;
            d[0]=s[0]; d[1]=s[1]; d[2]=s[2];
        }
    }

    std::vector<detection> dets;
    {
        int sv = dup(STDOUT_FILENO), dn = open("/dev/null", O_WRONLY);
        dup2(dn, STDOUT_FILENO); close(dn);
        detect_run(&ctx, rgb, mw, mh, FRAME_WIDTH, FRAME_HEIGHT, px, py, sc,
                   0.5f, 0.45f, dets, nullptr, nullptr, nullptr, nullptr);
        fflush(stdout); dup2(sv, STDOUT_FILENO); close(sv);
    }
    detect_deinit(&ctx); free(rgb);

    LOGI("Detections: %d", (int)dets.size());
    for (int i = 0; i < (int)dets.size(); i++) {
        const detection& d = dets[i];
        LOGI("  [%d] cls=%d score=%.3f bbox=(%.0f,%.0f,%.0f,%.0f)",
             i, d.cls, d.score, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h);
        _draw_box(disp, FRAME_WIDTH, FRAME_HEIGHT,
                  (int)(d.bbox.x - d.bbox.w*0.5f),
                  (int)(d.bbox.y - d.bbox.h*0.5f),
                  (int)d.bbox.w, (int)d.bbox.h, 0, 255, 0);
    }
    _save_rgb_as_jpeg("result.jpg", disp, FRAME_WIDTH, FRAME_HEIGHT);
    free(disp);
    return 0;
}

// ── test-motor ────────────────────────────────────────────────────────────────
int cmd_test_motor(const char* uart_dev, int argc, char** argv)
{
    int speed = 0;
    for (int i = 3; i < argc; i++)
        if (strncmp(argv[i], "speed=", 6) == 0) speed = atoi(argv[i]+6);

    printf("=== test-motor: %s ===\n", uart_dev);
    Motor motor(MotorDriverType::UART, uart_dev);

    if (speed > 0) {
        printf("Forward %d%% for 5s...\n", speed);
        motor.forward(speed);
        sleep(5);
        motor.standby();
    } else {
        printf("\n[1] Forward 50%% for 2s\n");  motor.forward(50);  sleep(2);
        printf("\n[2] Stop\n");                  motor.standby();    sleep(1);
        printf("\n[3] Backward 50%% for 2s\n"); motor.backward(50); sleep(2);
        printf("\n[4] Stop\n");                  motor.standby();    sleep(1);
        printf("\n[5] Turn left 50%% for 2s\n"); motor.left(50);    sleep(2);
        printf("\n[6] Stop\n");                  motor.standby();    sleep(1);
        printf("\n[7] Turn right 50%% for 2s\n");motor.right(50);   sleep(2);
        printf("\n[8] Stop\n");                  motor.standby();
    }
    printf("\n=== test-motor DONE ===\n");
    return 0;
}

// ── cmd_test_arm ──────────────────────────────────────────────────────────────
// Usage:
//   tennis test-arm [dev]                     -- show help
//   tennis test-arm [dev] grab                -- grab sequence
//   tennis test-arm [dev] release             -- release gripper
//   tennis test-arm [dev] show                -- lift and show
//   tennis test-arm [dev] pos                 -- move to home/ready
//   tennis test-arm [dev] demo                -- pos->grab->show->release
//   tennis test-arm [dev] <a0> <a1> <a2>      -- set 3 servo angles (0~270)
//   tennis test-arm [dev] set <id> <angle>    -- set single servo angle
int cmd_test_arm(const char* uart_dev, int argc, char** argv)
{
    // argv layout: argv[0]=tennis argv[1]="test-arm" argv[2]=dev_or_cmd ...
    // uart_dev is already resolved by main; remaining args start at argv[3]
    printf("=== test-arm  dev=%s ===\n", uart_dev);

    Arm arm(uart_dev, 115200);

    // Collect sub-args: everything after argv[2] (the dev)
    int sub_argc = argc - 3;   // args after <dev>
    char** sub = argv + 3;

    if (sub_argc == 0) {
        printf(
            "Usage:\n"
            "  tennis test-arm [dev] grab\n"
            "  tennis test-arm [dev] release\n"
            "  tennis test-arm [dev] show\n"
            "  tennis test-arm [dev] pos\n"
            "  tennis test-arm [dev] demo\n"
            "  tennis test-arm [dev] <a0> <a1> <a2>   (set 3 servo angles 0~270)\n"
            "  tennis test-arm [dev] set <id> <angle>  (set single servo)\n"
        );
        return 0;
    }

    const char* cmd = sub[0];

    if (strcmp(cmd, "grab") == 0) {
        printf("Executing grab sequence...\n"); fflush(stdout);
        arm.grab();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "release") == 0) {
        printf("Releasing gripper...\n"); fflush(stdout);
        arm.release();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "show") == 0) {
        printf("Showing ball...\n"); fflush(stdout);
        arm.show();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "pos") == 0) {
        printf("Moving to home/ready position...\n"); fflush(stdout);
        arm.grab_pos();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "demo") == 0) {
        printf("Demo: pos -> grab -> show -> release\n"); fflush(stdout);
        arm.grab_pos();  usleep(1500000);
        arm.grab();      usleep(2000000);
        arm.show();      usleep(2000000);
        arm.release();
        printf("Demo done.\n");
        return 0;
    }
    if (strcmp(cmd, "set") == 0 && sub_argc == 3) {
        int   id    = atoi(sub[1]);
        float angle = atof(sub[2]);
        printf("servo %d -> %.1f deg\n", id, angle); fflush(stdout);
        arm.set_angle(id, angle);
        usleep(1500000);  // 等待舵机到位
        return 0;
    }
    if (sub_argc == 3) {
        float a0 = atof(sub[0]);
        float a1 = atof(sub[1]);
        float a2 = atof(sub[2]);
        printf("servo 0=%.0f  1=%.0f  2=%.0f\n", a0, a1, a2); fflush(stdout);
        arm.set_angle(0, a0);
        arm.set_angle(1, a1);
        arm.set_angle(2, a2);
        usleep(1500000);  // 等待舵机到位
        return 0;
    }

    printf("Unknown arm command: %s\n", cmd);
    return 1;
}

