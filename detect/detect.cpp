#include "detect.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sys/time.h>

#include "rknn_api.h"
#include "image_utils.h"
#include "common.h"

// ── Timing helper ───────────────────────────────────────────────────────────────
static long elapsed_us(const struct timeval& start)
{
    struct timeval now;
    gettimeofday(&now, nullptr);
    return (now.tv_sec - start.tv_sec) * 1000000L + (now.tv_usec - start.tv_usec);
}

// ── NMS ───────────────────────────────────────────────────────────────────────
static float iou_cx(float ax, float ay, float aw, float ah,
                    float bx, float by, float bw, float bh)
{
    float ax1 = ax - aw * 0.5f, ay1 = ay - ah * 0.5f;
    float ax2 = ax + aw * 0.5f, ay2 = ay + ah * 0.5f;
    float bx1 = bx - bw * 0.5f, by1 = by - bh * 0.5f;
    float bx2 = bx + bw * 0.5f, by2 = by + bh * 0.5f;
    float iw = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
    float ih = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    float inter = iw * ih;
    float uni = aw * ah + bw * bh - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

// ── Init / deinit ─────────────────────────────────────────────────────────────
int detect_init(const char* model_path, rknn_app_context_t* ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    if (init_yolov8_model(model_path, ctx) != 0) {
        fprintf(stderr, "[detect] init_yolov8_model failed\n");
        return -1;
    }
    fprintf(stderr, "[detect] model loaded  w=%d h=%d ch=%d  n_out=%d\n",
            ctx->model_width, ctx->model_height, ctx->model_channel,
            ctx->io_num.n_output);
    for (int i = 0; i < (int)ctx->io_num.n_output; i++) {
        rknn_tensor_attr& a = ctx->output_attrs[i];
        fprintf(stderr, "[detect]   out[%d] dims=[%d,%d,%d,%d] type=%s\n",
                i, a.dims[0], a.dims[1], a.dims[2], a.dims[3],
                get_type_string(a.type));
    }
    return 0;
}

void detect_deinit(rknn_app_context_t* ctx)
{
    release_yolov8_model(ctx);
}

// ── Inference ─────────────────────────────────────────────────────────────────
// rgb_data : RGB888 letterboxed to model_w×model_h by decode_mjpeg().
// pad_x/y  : pixel offset added during letterbox.
// lbox_scale: scale factor (original → model).
// Outputs bbox in original camera frame (orig_w × orig_h), cx/cy center format.
// Timing outputs (optional, pass nullptr to skip): t_input, t_run, t_output, t_post
int detect_run(rknn_app_context_t* ctx,
               const uint8_t* rgb_data, int model_w, int model_h,
               int orig_w, int orig_h,
               int pad_x, int pad_y, float lbox_scale,
               float conf_thresh, float iou_thresh,
               std::vector<detection>& dets,
               long* t_input, long* t_run, long* t_output, long* t_post)
{
    struct timeval t_start, t_stage;
    gettimeofday(&t_start, nullptr);

    dets.clear();

    // 1. Set input
    gettimeofday(&t_stage, nullptr);
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index  = 0;
    inputs[0].type   = RKNN_TENSOR_UINT8;
    inputs[0].fmt    = RKNN_TENSOR_NHWC;
    inputs[0].size   = model_w * model_h * ctx->model_channel;
    inputs[0].buf    = const_cast<unsigned char*>(rgb_data);

    int ret = rknn_inputs_set(ctx->rknn_ctx, 1, inputs);
    if (ret < 0) { fprintf(stderr, "[detect] rknn_inputs_set %d\n", ret); return -1; }
    long t_input_us = elapsed_us(t_stage);

    // 2. Run
    gettimeofday(&t_stage, nullptr);
    ret = rknn_run(ctx->rknn_ctx, nullptr);
    if (ret < 0) { fprintf(stderr, "[detect] rknn_run %d\n", ret); return -1; }
    long t_run_us = elapsed_us(t_stage);

    // 3. Get output as FP32
    gettimeofday(&t_stage, nullptr);
    int n_out = ctx->io_num.n_output;
    rknn_output outputs[n_out];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < n_out; i++) { outputs[i].index = i; outputs[i].want_float = 1; }
    ret = rknn_outputs_get(ctx->rknn_ctx, n_out, outputs, nullptr);
    if (ret < 0) { fprintf(stderr, "[detect] rknn_outputs_get %d\n", ret); return -1; }
    long t_output_us = elapsed_us(t_stage);

    // 4. Parse output
    // 支持两种格式：
    // - 单输出: [1, 5, N] 或 [1, N, 5] - x,y,w,h,conf 合并
    // - 双输出: boxes=[1,4,N], scores=[1,1,N] - YOLOv8 标准格式
    gettimeofday(&t_stage, nullptr);

    std::vector<detection> cands;

    if (n_out == 2) {
        // 双输出格式: YOLOv8 标准格式
        // outputs[0]: boxes [1, 4, N]
        // outputs[1]: scores [1, 1, N]
        const float* boxes = (const float*)outputs[0].buf;
        const float* scores = (const float*)outputs[1].buf;

        const rknn_tensor_attr& box_attr = ctx->output_attrs[0];
        int N = box_attr.dims[2];  // 8400

        for (int j = 0; j < N; j++) {
            float conf = scores[j];
            if (conf < conf_thresh) continue;

            // boxes: [x, y, w, h] × N
            float cx = boxes[0 * N + j];
            float cy = boxes[1 * N + j];
            float w  = boxes[2 * N + j];
            float h  = boxes[3 * N + j];

            // Undo letterbox → original camera frame
            float rx = (cx - pad_x) / lbox_scale;
            float ry = (cy - pad_y) / lbox_scale;
            float rw = w / lbox_scale;
            float rh = h / lbox_scale;
            rx = std::max(0.0f, std::min(rx, (float)orig_w));
            ry = std::max(0.0f, std::min(ry, (float)orig_h));

            detection d;
            d.bbox.x = rx; d.bbox.y = ry;
            d.bbox.w = rw; d.bbox.h = rh;
            d.score = conf; d.cls = 0; d.batch_idx = 0;
            cands.push_back(d);
        }
    } else {
        // 单输出格式: [1, 5, N] 或 [1, N, 5]
        const rknn_tensor_attr& oa = ctx->output_attrs[0];
        int d1 = oa.dims[1], d2 = oa.dims[2];
        bool layout_5N = (d1 < d2);
        int N = layout_5N ? d2 : d1;
        int F = layout_5N ? d1 : d2;
        const float* buf = (const float*)outputs[0].buf;

        for (int j = 0; j < N; j++) {
            float cx, cy, w, h, conf;
            if (layout_5N) {
                cx = buf[0*N+j]; cy = buf[1*N+j];
                w  = buf[2*N+j]; h  = buf[3*N+j]; conf = buf[4*N+j];
            } else {
                cx = buf[j*F+0]; cy = buf[j*F+1];
                w  = buf[j*F+2]; h  = buf[j*F+3]; conf = buf[j*F+4];
            }
            if (conf < conf_thresh) continue;

            // Undo letterbox → original camera frame
            float rx = (cx - pad_x) / lbox_scale;
            float ry = (cy - pad_y) / lbox_scale;
            float rw = w / lbox_scale;
            float rh = h / lbox_scale;
            rx = std::max(0.0f, std::min(rx, (float)orig_w));
            ry = std::max(0.0f, std::min(ry, (float)orig_h));

            detection d;
            d.bbox.x = rx; d.bbox.y = ry;
            d.bbox.w = rw; d.bbox.h = rh;
            d.score = conf; d.cls = 0; d.batch_idx = 0;
            cands.push_back(d);
        }
    }

    // 6. NMS
    std::sort(cands.begin(), cands.end(),
              [](const detection& a, const detection& b){ return a.score > b.score; });
    std::vector<bool> sup(cands.size(), false);
    for (int i = 0; i < (int)cands.size(); i++) {
        if (sup[i]) continue;
        dets.push_back(cands[i]);
        for (int j = i+1; j < (int)cands.size(); j++) {
            if (!sup[j] &&
                iou_cx(cands[i].bbox.x, cands[i].bbox.y, cands[i].bbox.w, cands[i].bbox.h,
                       cands[j].bbox.x, cands[j].bbox.y, cands[j].bbox.w, cands[j].bbox.h)
                > iou_thresh)
                sup[j] = true;
        }
    }
    long t_post_us = elapsed_us(t_stage);

    // 7. Release
    gettimeofday(&t_stage, nullptr);
    rknn_outputs_release(ctx->rknn_ctx, n_out, outputs);
    long t_release = elapsed_us(t_stage);

    long t_total = elapsed_us(t_start);

    // Export timing outputs if requested
    if (t_input)  *t_input  = t_input_us;
    if (t_run)    *t_run    = t_run_us;
    if (t_output) *t_output = t_output_us;
    if (t_post)   *t_post   = t_post_us;

    // Print timing breakdown - always print for debugging
    // printf("[detect] total=%.1fms  in=%.1f run=%.1f out=%.1f post=%.1f rel=%.1f  cands=%d nms=%d\n",
    //         t_total / 1000.0f,
    //         t_input_us / 1000.0f,
    //         t_run_us / 1000.0f,
    //         t_output_us / 1000.0f,
    //         t_post_us / 1000.0f,
    //         t_release / 1000.0f,
    //         (int)cands.size(), (int)dets.size());
    fflush(stdout);

    return (int)dets.size();
}
