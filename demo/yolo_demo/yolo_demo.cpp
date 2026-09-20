/**
 * YOLOv5 昇腾NPU推理 Demo
 * Model output shape: 1*25200*85
 * Input size: 640x640 RGB
 * Compile define: USE_OPENCV
 * Run: ./yolo_demo model.om test.jpg [mode]
 * mode:0-auto 1-NPU 2-ONNX
 */
#include "npu_cann_adapter.h"
#include "../yolo_common.h"   /* 预处理/后处理公共实现(letterbox/decode/NMS/save) */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;
#define INPUT_W     640
#define INPUT_H     640

/* 阈值:可在 main 里通过 argv 覆盖
 * 默认值针对标准模型(conf=0.25 / nms=0.45);
 * yolov5su.onnx 是正常模型,用默认值即可
 */
static float g_conf_thresh = 0.25f;   /* 最终置信度阈值 */
static float g_obj_pass_th = 0.10f;   /* obj_sig 预筛,仅经典布局使用 */
static float g_nms_thresh  = 0.45f;   /* NMS IoU */

#define OUTPUT_DIR  "./output"

static void create_output_dir()
{
    if (!fs::exists(OUTPUT_DIR)) fs::create_directories(OUTPUT_DIR);
}
/* read_binary_file / letterbox / load_image_tensor / save_result_img
 * 均由 ../yolo_common.h 提供 */

int main(int argc, char* argv[])
{
    create_output_dir();
    const char* model_path = "./yolov5s.om";
    const char* img_path   = "./test.jpg";
    uint32_t mode = NPU_DRIVER_FORMAT_AUTO;
    int loop = 100;
    const int dev_id = 0;
    LetterBoxParam lb_param{1.0f, 0, 0};
    int ret = 0;
    if(argc >= 2) model_path = argv[1];
    if(argc >= 3) img_path   = argv[2];
    if(argc >= 4) mode = static_cast<uint32_t>(atoi(argv[3]));
    if(argc >= 5) loop = atoi(argv[4]);
    if(argc >= 6) g_conf_thresh = static_cast<float>(atof(argv[5]));
    if(argc >= 7) g_nms_thresh  = static_cast<float>(atof(argv[6]));
    if(mode > 2) mode = 0;
    if(loop <= 0) loop = 100;
    if(g_conf_thresh <= 0.0f) g_conf_thresh = 0.25f;
    if(g_nms_thresh  <= 0.0f) g_nms_thresh  = 0.45f;
    if(g_obj_pass_th <= 0.0f) g_obj_pass_th = 0.10f;
    YoloThresholds th{g_conf_thresh, g_obj_pass_th, g_nms_thresh};
    YoloOutputLayout lay;   /* warmup 后由 yolo_detect_layout 填充 */
    printf("=============================================\n");
    printf("          YOLOv5 NPU Inference Demo\n");
    printf("Model: %s\nInput: %s\nMode: %d(0:auto 1:NPU 2:ONNX CPU)\n", model_path, img_path, mode);
    printf("Thresholds: conf=%.2f  nms=%.2f  obj-pass=%.2f\n", g_conf_thresh, g_nms_thresh, g_obj_pass_th);
    printf("Output dir: %s\n", OUTPUT_DIR);
    printf("=============================================\n");
    {
        std::vector<unsigned char> input_tensor;
#ifdef USE_OPENCV
        printf("\n[Step1] Preprocess image...\n");
        if (load_image_tensor(img_path, input_tensor, lb_param) != 0)
        {
            printf("Try load binary file...\n");
            if (read_binary_file(img_path, input_tensor) != 0)
            {
                size_t sz = 3 * INPUT_W * INPUT_H * sizeof(float);
                input_tensor.resize(sz, 0);
                printf("Use zero tensor for test\n");
            }
        }
#else
        if (read_binary_file(input_tensor) != 0)
        {
            size_t sz = 3 * INPUT_W * sizeof(float);
            input_tensor.resize(sz, 0);
        }
#endif
        printf("Input tensor size: %zu bytes\n", input_tensor.size());
        printf("\n[Step2] Create NPU handle\n");
        NPUHandle handle = NPU_Create();
        if (!handle)
        {
            fprintf(stderr, "NPU_Create fail: %s\n", NPU_GetLastError(nullptr));
            goto clean_all;
        }
        printf("\n[Step3] Load model\n");
        NPUModelLoadMsg load_msg;
        memset(&load_msg, 0, sizeof(load_msg));
        load_msg.npu_count = 1;
        load_msg.npu_ids[0] = dev_id;
        strncpy(load_msg.model_path, model_path, sizeof(load_msg.model_path) - 1);
        load_msg.model_format = mode;
        auto t_load_s = std::chrono::high_resolution_clock::now();
        int load_ret = NPU_Model_Load(handle, &load_msg, sizeof(load_msg));
        auto t_load_e = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load_e - t_load_s).count();
        if (load_ret != 0)
        {
            fprintf(stderr, "Load model err: %s\n", NPU_GetLastError(handle));
            NPU_Destroy(handle);
            goto clean_all;
        }
        printf("Load success, cost %.2f ms\n", load_ms);
        const size_t out_buf_sz = 25200 * 85 * sizeof(float);
        std::vector<unsigned char> out_buf(out_buf_sz);
        unsigned int actual_sz = static_cast<unsigned int>(out_buf_sz);
        std::vector<DetectBox> dets;
        float* out_ptr = reinterpret_cast<float*>(out_buf.data());

        printf("\n[Step4] Warm up (sync infer x 5)\n");
        for (int i = 0; i < 5; i++)
        {
            ret = NPU_Model_Infer(handle, input_tensor.data(), static_cast<unsigned int>(input_tensor.size()), out_buf.data(), &actual_sz);
            if (ret != 0)
            {
                fprintf(stderr, "Warmup err: %s\n", NPU_GetLastError(handle));
                NPU_Model_Unload(handle);
                NPU_Destroy(handle);
                goto clean_all;
            }
        }

        /* 探测后端类型: Event 只在 NPU (.om) 后端可用 */
        NPUEvent probe = nullptr;
        bool is_npu_backend = (NPU_EventCreate(handle, &probe) == 0);
        if (probe) NPU_EventDestroy(handle, probe);
        if (!is_npu_backend) {
            /* ONNX Runtime CPU 太慢, benchmark 只跑 1 次 */
            int before = loop;
            loop = 1;
            if (before != 1) {
                printf("[WARN] ONNX Runtime CPU 后端太慢, benchmark loop 从 %d 降为 1\n", before);
            }
        }
        printf("Warmup done, start benchmark loop = %d\n", loop);

        /* 检测输出布局: 优先模型声明 shape,兜底用 warmup 返回的实际字节数 */
        int lay_dims[8] = {0};
        int lay_nd = 0;
        NPUModelStatus st;
        if (NPU_GetModelStatus(handle, &st) == 0) {
            lay_nd = st.output_shape.num_dims;
            for (int i = 0; i < lay_nd && i < 8; i++) lay_dims[i] = st.output_shape.dims[i];
        }
        lay = yolo_detect_layout(lay_dims, lay_nd, actual_sz);
        yolo_print_layout(lay);

        /* 创建 Event 用于精确设备计时（ONNX 后端不支持 Event，降级只用 Host Timer） */
        NPUEvent ev_start = nullptr, ev_end = nullptr;
        bool has_event = true;
        if (NPU_EventCreate(handle, &ev_start) != 0 || NPU_EventCreate(handle, &ev_end) != 0)
        {
            fprintf(stderr, "[WARN] Event not available: %s (ONNX Runtime fallback to host timer)\n",
                    NPU_GetLastError(handle));
            has_event = false;
        }

        std::vector<float> dev_ms_list, host_ms_list;
        dev_ms_list.reserve(loop);
        host_ms_list.reserve(loop);
        bool async_available = true;  /* 某些芯片(如 Ascend310)不支持 aclmdlExecuteAsync,需持续降级 */

        for (int i = 0; i < loop; i++)
        {
            /* ---- 1. Device Timer (仅 NPU 后端) ---- */
            if (has_event) {
                NPU_EventRecord(handle, ev_start, nullptr);
                if (async_available) {
                    ret = NPU_Model_InferAsync(handle, nullptr,
                        input_tensor.data(), static_cast<unsigned int>(input_tensor.size()),
                        out_buf.data(), &actual_sz);
                    if (ret != 0) {
                        async_available = false;
                        ret = NPU_Model_Infer(handle,
                            input_tensor.data(), static_cast<unsigned int>(input_tensor.size()),
                            out_buf.data(), &actual_sz);
                    }
                } else {
                    ret = NPU_Model_Infer(handle,
                        input_tensor.data(), static_cast<unsigned int>(input_tensor.size()),
                        out_buf.data(), &actual_sz);
                }
                NPU_EventRecord(handle, ev_end, nullptr);
                NPU_EventSynchronize(handle, ev_end);
                if (ret != 0) {
                    fprintf(stderr, "Device-timer loop %d err: %s\n", i, NPU_GetLastError(handle));
                    break;
                }
                float dev_ms = 0.0f;
                NPU_EventElapsedTime(handle, &dev_ms, ev_start, ev_end);
                dev_ms_list.push_back(dev_ms);
            }

            /* ---- 2. Host Timer: 同步 Infer 用户视角 wall time ---- */
            auto hs = std::chrono::high_resolution_clock::now();
            ret = NPU_Model_Infer(handle,
                input_tensor.data(), static_cast<unsigned int>(input_tensor.size()),
                out_buf.data(), &actual_sz);
            auto he = std::chrono::high_resolution_clock::now();
            if (ret != 0)
            {
                fprintf(stderr, "Sync Infer loop %d err: %s\n", i, NPU_GetLastError(handle));
                break;
            }
            double host_ms = static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(he - hs).count()) / 1000.0;
            host_ms_list.push_back(static_cast<float>(host_ms));
        }

        if (has_event) {
            NPU_EventDestroy(handle, ev_start);
            NPU_EventDestroy(handle, ev_end);
        }

        /* --- 打印完整统计 --- */
        auto summarize = [](const std::vector<float>& xs, const char* tag) {
            if (xs.empty()) return;
            std::vector<float> v(xs);
            std::sort(v.begin(), v.end());
            float sum = 0.0f;
            for (float x : v) sum += x;
            const int n = static_cast<int>(v.size());
            auto pct = [&](float p) -> float {
                int idx = std::min(n - 1, std::max(0, static_cast<int>(p * n)));
                return v[idx];
            };
            float avg = sum / n;
            printf("[%-16s] n=%3d  avg=%6.2fms  min=%6.2fms  max=%6.2fms"
                   "  p50=%6.2f  p90=%6.2f  p99=%6.2f  FPS=%7.2f\n",
                tag, n, avg, v.front(), v.back(),
                pct(0.50f), pct(0.90f), pct(0.99f), 1000.0f / avg);
        };
        printf("\n[Step5] Benchmark Results\n");
        printf("------------------------------------------------------------\n");
        summarize(dev_ms_list,  "Device (Event) ");
        summarize(host_ms_list, "Host   (Sync)  ");
        if (!dev_ms_list.empty() && !host_ms_list.empty())
        {
            float dev_avg = 0, host_avg = 0;
            for (auto x : dev_ms_list) dev_avg += x;
            for (auto x : host_ms_list) host_avg += x;
            dev_avg /= dev_ms_list.size();
            host_avg /= host_ms_list.size();
            printf("Host 开销 = %.2f ms  (Host - Device)\n", host_avg - dev_avg);
            printf("Device 效率 = %.1f%%  (Device / Host)\n", 100.0f * dev_avg / host_avg);
        }
        printf("------------------------------------------------------------\n");
        printf("\n[Step6] Post process\n");
        dets.clear();
        yolo_decode(out_ptr, lay, th, dets);
        printf("Raw det: %zu\n", dets.size());
        // NMS 极大值抑制
        yolo_nms(dets, th.nms);
        printf("After NMS: %zu\n", dets.size());
        /* 打印 top-5 便于人工核对 */
        for (size_t i = 0; i < dets.size() && i < 5; i++) {
            const auto& d = dets[i];
            printf("  det[%zu] cx=%.1f cy=%.1f w=%.1f h=%.1f conf=%.3f cls=%d(%s)\n",
                   i, d.x, d.y, d.w, d.h, d.conf, d.class_id,
                   (d.class_id >= 0 && d.class_id < YOLO_NUM_CLS) ? kClassNames[d.class_id] : "unk");
        }
#ifdef USE_OPENCV
        if (!dets.empty())
        {
            save_result_img(img_path, dets, lb_param, OUTPUT_DIR);
        }
#endif
        NPU_Model_Unload(handle);
        NPU_Destroy(handle);
    }
clean_all:
    printf("\n[Step7] Inference finished\n");
    return 0;
}