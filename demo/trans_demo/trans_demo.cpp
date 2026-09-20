/**
 * TinyViT (Vision Transformer) 昇腾NPU推理 Demo
 * 调用自封装库 libnpu_driver (ACL/ONNX Runtime 双后端)
 *
 * 模型: TinyViT 6层 dim=192 heads=3 patch=16, 输入 1x3x224x224, 输出 1000 类
 * 权重为随机初始化(见 gen_vit_onnx.py), 例程目的是验证 Transformer
 * (MatMul/Softmax/LayerNorm/GELU) 在 310B NPU 上的算子兼容性与性能
 *
 * Run: ./trans_demo model/tiny_vit.om infer/dog.jpg [loop] [labels]
 */
#include "npu_cann_adapter.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <unistd.h>   /* usleep */

#define INPUT_SIZE 224
#define NUM_CLASSES 1000
#define OUTPUT_DIR "./output"

/* 归一化参数 (与 timm vit_small_patch16_224.augreg_in1k 训练一致) */
static const float kMean[3] = {0.5f, 0.5f, 0.5f};
static const float kStd[3]  = {0.5f, 0.5f, 0.5f};

/* ===================== 工具函数 ===================== */

static std::vector<std::string> load_labels(const char* path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (!f.is_open()) return labels;
    std::string line;
    while (std::getline(f, line)) {
        size_t comma = line.find(',');
        if (comma != std::string::npos) line = line.substr(0, comma);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        size_t start = line.find_first_not_of(' ');
        if (start == std::string::npos) continue;
        labels.push_back(line.substr(start));
    }
    if (!labels.empty())
        printf("[INFO] Loaded %zu class labels from %s\n", labels.size(), path);
    return labels;
}

static const char* get_label(const std::vector<std::string>& labels, int id) {
    if (id >= 0 && id < (int)labels.size()) return labels[id].c_str();
    return "unknown";
}

/* softmax (数值稳定) */
static void softmax(const float* logits, float* probs, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

struct TopK { int index; float prob; };
static void top_k(const float* probs, int n, int k, std::vector<TopK>& out) {
    out.clear();
    for (int i = 0; i < n; i++) {
        TopK t{i, probs[i]};
        out.push_back(t);
    }
    std::partial_sort(out.begin(), out.begin() + k, out.end(),
                      [](const TopK& a, const TopK& b) { return a.prob > b.prob; });
    out.resize(k);
}

/* benchmark 统计打印 */
static void print_benchmark(const std::vector<float>& xs, const char* tag) {
    if (xs.empty()) return;
    std::vector<float> v(xs);
    std::sort(v.begin(), v.end());
    float sum = 0.0f;
    for (float x : v) sum += x;
    const int n = (int)v.size();
    auto pct = [&](float p) -> float {
        int idx = std::min(n - 1, std::max(0, (int)(p * n)));
        return v[idx];
    };
    float avg = sum / n;
    printf("[%-16s] n=%3d  avg=%6.2fms  min=%6.2fms  max=%6.2fms"
           "  p50=%6.2f  p90=%6.2f  p99=%6.2f  FPS=%7.2f\n",
           tag, n, avg, v.front(), v.back(),
           pct(0.50f), pct(0.90f), pct(0.99f), 1000.0f / avg);
}

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>

/* 预处理: resize 224 -> RGB -> /255 -> (x-mean)/std -> NCHW float32 */
static int preprocess_image(const char* img_path, std::vector<unsigned char>& tensor) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        fprintf(stderr, "[ERROR] Read image fail: %s\n", img_path);
        return -1;
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

    tensor.resize(3 * INPUT_SIZE * INPUT_SIZE * sizeof(float));
    float* dst = reinterpret_cast<float*>(tensor.data());
    for (int y = 0; y < INPUT_SIZE; y++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            cv::Vec3f pix = rgb.at<cv::Vec3f>(y, x);
            float ch[3] = {pix[0], pix[1], pix[2]};
            for (int c = 0; c < 3; c++)
                dst[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                    (ch[c] - kMean[c]) / kStd[c];
        }
    }
    printf("[INFO] Img %dx%d -> resize %dx%d NCHW RGB normalized\n",
           img.cols, img.rows, INPUT_SIZE, INPUT_SIZE);
    return 0;
}
#endif /* USE_OPENCV */

/* ===================== Main ===================== */
int main(int argc, char* argv[]) {
    const char* model_path = "./model/tiny_vit.om";
    const char* img_path = "./infer/dog.jpg";
    const char* labels_path = "./classname.txt";
    int loop = 100;
    bool enable_prof = false;            /* 是否采集每层算子耗时 */
    const char* prof_dir = nullptr;      /* profiling 输出目录, 默认关闭 */
    bool enable_mon = false;             /* 是否开启实时内存/利用率监控曲线 */
    int  mon_interval_ms = 10;           /* 采样间隔 */

    /* argv: model img loop [labels|prof_dir|monitor[:ms]] [prof_dir|monitor[:ms]] [monitor[:ms]] */
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) img_path = argv[2];
    if (argc >= 4) loop = atoi(argv[3]);
    for (int i = 4; i < argc && i <= 6; i++) {
        if (strstr(argv[i], "monitor")) {
            enable_mon = true;
            const char* colon = strchr(argv[i], ':');
            if (colon) mon_interval_ms = atoi(colon + 1);
        } else if (strstr(argv[i], ".txt")) {
            labels_path = argv[i];
        } else {
            prof_dir = argv[i];
            enable_prof = true;
        }
    }
    if (loop <= 0) loop = 100;
    /* 若未指定 prof_dir 但启用了 profiling, 用默认目录 */
    if (enable_prof && !prof_dir) prof_dir = "./prof_out";

    printf("=============================================\n");
    printf("     TinyViT (Transformer) NPU Demo\n");
    printf("Model: %s\nInput: %s\nLoop:  %d\n", model_path, img_path, loop);
    printf("LayerProfiling: %s%s%s\n", enable_prof ? "ON -> " : "OFF",
           enable_prof ? prof_dir : "",
           enable_prof ? " (csv 解析后打印 Top-N 算子耗时)" : "");
    printf("Monitor:        %s\n", enable_mon ?
           "ON (实时内存/利用率曲线)" : "OFF (argv 加 monitor[:ms] 开启)");
    printf("=============================================\n");

    /* 必须在 NPU_Create 之前设置 profiling 输出目录(aclprofInit 需在 aclInit 前) */
    if (enable_prof) {
        if (NPU_LayerProfiling_SetOutput(prof_dir) != 0) {
            fprintf(stderr, "[WARN] SetOutput 失败: %s, profiling 关闭\n",
                    NPU_GetLastError(nullptr));
            enable_prof = false;
        }
    }
    setvbuf(stdout, nullptr, _IOLBF, 0);   /* 行缓冲, 便于观察卡点 */
    setvbuf(stderr, nullptr, _IOLBF, 0);

    /* 加载标签: 命令行指定 -> 本目录/上级 classname.txt -> resnet_demo 的 */
    std::vector<std::string> labels;
    const char* candidates[] = {
        labels_path, "classname.txt", "../classname.txt",
        "../resnet_demo/classname.txt"
    };
    for (const char* cand : candidates) {
        labels = load_labels(cand);
        if (!labels.empty()) break;
    }

#ifdef USE_OPENCV
    /* ---------- Step1: 预处理 ---------- */
    printf("\n[Step1] Preprocess image\n");
    std::vector<unsigned char> input_tensor;
    if (preprocess_image(img_path, input_tensor) != 0) return -1;
#else
    fprintf(stderr, "[ERROR] 需要 OpenCV, 请用 -DUSE_OPENCV 编译\n");
    return -1;
#endif

    /* ---------- Step2: 创建 NPU 句柄 ---------- */
    printf("\n[Step2] Create NPU handle\n"); fflush(stdout);
    NPUHandle h = NPU_Create();
    if (!h) {
        fprintf(stderr, "[ERROR] NPU_Create failed: %s\n", NPU_GetLastError(h));
        return -1;
    }
    printf("[NPU] Handle created\n"); fflush(stdout);

    /* ---------- Step3: 加载模型 ---------- */
    printf("\n[Step3] Load model\n"); fflush(stdout);
    NPUModelLoadMsg load_msg;
    memset(&load_msg, 0, sizeof(load_msg));
    load_msg.npu_count = 1;
    load_msg.npu_ids[0] = 0;
    load_msg.model_format = NPU_DRIVER_FORMAT_AUTO;  /* .om 优先, onnx 自动转 */
    strncpy(load_msg.model_path, model_path, sizeof(load_msg.model_path) - 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = NPU_Model_Load(h, &load_msg, sizeof(load_msg));
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ret != 0) {
        fprintf(stderr, "[ERROR] NPU_Model_Load failed: %s\n", NPU_GetLastError(h));
        NPU_Destroy(h);
        NPU_Finalize();
        return -1;
    }
    printf("[NPU] Model loaded in %.0f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count()); fflush(stdout);

    /* 查询模型状态 */
    NPUModelStatus status;
    memset(&status, 0, sizeof(status));
    NPU_GetModelStatus(h, &status);
    printf("[NPU] Input:  %u bytes, dims=%d\n", status.input_size, status.input_shape.num_dims);
    printf("[NPU] Output: %u bytes, dims=%d\n", status.output_size, status.output_shape.num_dims); fflush(stdout);

    /* ---------- Step3.5: 资源监控初始化 ---------- */
    NPUMemInfo mem_loaded{};                     /* 模型加载后的设备内存 */
    NPU_GetDevMemory(h, &mem_loaded);
    NPU_Monitor_Reset(h);                        /* 峰值追踪从当前开始 */
    printf("\n[Step3.5] Monitor: dev-mem after model load: used=%.1fMB / total=%.1fMB (%.1f%%)\n",
           mem_loaded.used_bytes / 1048576.0,
           mem_loaded.total_bytes / 1048576.0,
           mem_loaded.used_ratio);

    /* 实时监控曲线: 后台线程周期采样 */
    if (enable_mon) {
        if (NPU_Monitor_Start(h, (uint32_t)mon_interval_ms) != 0)
            fprintf(stderr, "[WARN] Monitor 启动失败: %s\n", NPU_GetLastError(h));
    }
    fflush(stdout);

    /* ---------- Step4: Warmup ---------- */
    printf("\n[Step4] Warmup (x5)\n"); fflush(stdout);
    std::vector<unsigned char> out_buf(status.output_size);
    for (int i = 0; i < 5; i++) {
        unsigned int out_len = (unsigned int)out_buf.size();
        ret = NPU_Model_Infer(h, input_tensor.data(), (unsigned int)input_tensor.size(),
                              out_buf.data(), &out_len);
        if (ret != 0) {
            fprintf(stderr, "[ERROR] Warmup infer %d failed: %s\n", i, NPU_GetLastError(h));
            break;
        }
    }
    printf("Warmup done\n"); fflush(stdout);

    /* ---------- Step5: Benchmark (Event 设备计时 + Host 计时) ---------- */
    printf("\n[Step5] Benchmark loop=%d\n", loop); fflush(stdout);
    NPUEvent ev_start = nullptr, ev_end = nullptr;
    bool has_event = (NPU_EventCreate(h, &ev_start) == 0 &&
                      NPU_EventCreate(h, &ev_end) == 0);
    if (!has_event)
        printf("[WARN] Event 不可用, 仅用 Host 计时\n");
    fflush(stdout);

    std::vector<float> dev_ms_list, host_ms_list;
    dev_ms_list.reserve(loop);
    host_ms_list.reserve(loop);
    bool async_available = true;

    /* 启动算子耗时采集(若启用), 覆盖整段 benchmark */
    if (enable_prof) {
        printf("[Step5.1] LayerProfiling_Start ...\n"); fflush(stdout);
        if (NPU_LayerProfiling_Start(h) != 0) {
            fprintf(stderr, "[WARN] LayerProfiling_Start 失败: %s\n", NPU_GetLastError(h));
            enable_prof = false;
        } else {
            printf("[Step5.1] Start done\n"); fflush(stdout);
        }
    }

    for (int i = 0; i < loop; i++) {
        unsigned int out_len = (unsigned int)out_buf.size();
        float dev_ms = 0.0f;

        if (has_event) NPU_EventRecord(h, ev_start, nullptr);
        auto hs = std::chrono::high_resolution_clock::now();

        if (has_event && async_available) {
            ret = NPU_Model_InferAsync(h, nullptr,
                input_tensor.data(), (unsigned int)input_tensor.size(),
                out_buf.data(), &out_len);
            if (ret != 0) {
                async_available = false;   /* 部分芯片不支持 ExecuteAsync, 降级 */
                ret = NPU_Model_Infer(h, input_tensor.data(),
                    (unsigned int)input_tensor.size(), out_buf.data(), &out_len);
            }
        } else {
            ret = NPU_Model_Infer(h, input_tensor.data(), (unsigned int)input_tensor.size(),
                                  out_buf.data(), &out_len);
        }

        auto he = std::chrono::high_resolution_clock::now();
        if (has_event) {
            NPU_EventRecord(h, ev_end, nullptr);
            NPU_EventSynchronize(h, ev_end);
            NPU_EventElapsedTime(h, &dev_ms, ev_start, ev_end);
        }
        if (ret != 0) {
            fprintf(stderr, "[ERROR] Infer %d failed: %s\n", i, NPU_GetLastError(h));
            break;
        }
        if (has_event) dev_ms_list.push_back(dev_ms);
        host_ms_list.push_back((float)std::chrono::duration<double, std::milli>(he - hs).count());
    }

    printf("------------------------------------------------------------\n");
    print_benchmark(dev_ms_list, "Device (Event) ");
    print_benchmark(host_ms_list, "Host   (chrono) ");
    printf("------------------------------------------------------------\n");

    /* ---------- Step5.5: 运行时资源监控结果 ---------- */
    /* 实时监控曲线: 停止采样, 打印 ASCII 曲线并保存 PNG 到 ../monitor/ */
    if (enable_mon && NPU_Monitor_IsRunning(h)) {
        NPU_Monitor_Stop(h);
        NPU_Monitor_PrintCurve(h, 64);
        char png_path[256] = {0};
        if (NPU_Monitor_SavePng(h, "./monitor", png_path, sizeof(png_path)) == 0) {
            printf("[Monitor] 曲线图已保存: %s\n", png_path);
        } else {
            fprintf(stderr, "[WARN] PNG 保存失败: %s\n", NPU_GetLastError(h));
        }
        /* 附带利用率曲线数据摘要 */
        NPUSamplePoint hist[NPU_MONITOR_MAX_SAMPLES];
        uint32_t hn = 0;
        if (NPU_Monitor_GetHistory(h, hist, NPU_MONITOR_MAX_SAMPLES, &hn) == 0 && hn > 0) {
            int u_max = -1; size_t m_max = 0;
            for (uint32_t i = 0; i < hn; i++) {
                u_max = std::max(u_max, std::max(hist[i].cube, hist[i].aicpu));
                m_max = std::max(m_max, hist[i].used_bytes);
            }
            printf("[Monitor] samples=%u  dev-mem max=%.1fMB  AI-Core max=%d%%\n",
                   hn, m_max / 1048576.0, u_max);
        }
    }

    NPUMemPeak peak{};
    if (NPU_Monitor_GetPeak(h, &peak) == 0) {
        printf("[Monitor] dev-mem peak: %.1f MB (cur %.1f MB) @ sample #%llu\n",
               peak.peak_used_bytes / 1048576.0,
               peak.cur_used_bytes / 1048576.0,
               (unsigned long long)peak.sample_count);
        fflush(stdout);
    }
    NPUUtilization util{};
    if (NPU_GetUtilization(h, &util) == 0) {
        printf("[Monitor] utilization: cube=%d%% vector=%d%% aicpu=%d%% mem=%d%%\n",
               util.cube, util.vector, util.aicpu, util.memory);
    } else {
        printf("[Monitor] utilization: 不支持 (%s)\n", NPU_GetLastError(h));
    }
    printf("------------------------------------------------------------\n");

    if (ev_start) NPU_EventDestroy(h, ev_start);
    if (ev_end) NPU_EventDestroy(h, ev_end);

    /* 停止采集, 触发 CANN 写入 csv */
    if (enable_prof) {
        if (NPU_LayerProfiling_Stop(h) == 0) {
            /* CANN 异步刷盘, 给 200ms 缓冲 */
            usleep(200 * 1000);
            printf("\n========== Per-Layer Profile (Top-20 by total_us) ==========\n");
            const int kMax = 64;
            NPULayerInfo layers[kMax];
            int actual = 0;
            if (NPU_LayerProfiling_Parse(prof_dir, layers, kMax, &actual) == 0) {
                printf("  %-4s  %-22s  %-50s  %10s  %10s  %10s\n",
                       "#", "OpType", "OpName", "Total(us)", "Avg(us)", "Count");
                printf("  %-4s  %-22s  %-50s  %10s  %10s  %10s\n",
                       "----", "----------------------",
                       "--------------------------------------------------",
                       "----------", "----------", "----------");
                int show = actual < 20 ? actual : 20;
                uint64_t grand_total = 0;
                for (int i = 0; i < show; ++i) {
                    printf("  %-4d  %-22s  %-50s  %10llu  %10llu  %10u\n",
                           i + 1,
                           layers[i].op_type[0] ? layers[i].op_type : "-",
                           layers[i].op_name,
                           (unsigned long long)layers[i].total_us,
                           (unsigned long long)layers[i].duration_us,
                           layers[i].task_count);
                    grand_total += layers[i].total_us;
                }
                printf("  ----------------------------------------------------------------------\n");
                printf("  Top-%d 累计耗时: %llu us  (%.2f ms)  |  共 %d 个算子被记录\n",
                       show, (unsigned long long)grand_total,
                       grand_total / 1000.0, actual);
                printf("  完整 csv 见: %s/PROF_xxx/op_summary_*.csv\n", prof_dir);
            } else {
                fprintf(stderr, "[WARN] 解析 csv 失败: %s\n", NPU_GetLastError(nullptr));
            }
            printf("============================================================\n");

            /* 生成自包含 HTML 可视化报告(自动放到最新 PROF_xxx 子目录下) */
            if (NPU_LayerProfiling_ExportHtml(prof_dir, nullptr) == 0) {
                printf("\n[Profiling] HTML 报告已生成, 见上面 [NPU] ExportHtml auto path 日志\n");
            } else {
                fprintf(stderr, "[WARN] 生成 HTML 失败: %s\n", NPU_GetLastError(nullptr));
            }
        } else {
            fprintf(stderr, "[WARN] LayerProfiling_Stop 失败: %s\n", NPU_GetLastError(h));
        }
    }

    /* ---------- Step6: 后处理 + 分类结果 ---------- */
    printf("\n[Step6] Post process\n");
    float* logits = reinterpret_cast<float*>(out_buf.data());
    int num_cls = (int)(out_buf.size() / sizeof(float));
    if (num_cls > NUM_CLASSES) num_cls = NUM_CLASSES;
    printf("[INFO] Output: %d classes\n", num_cls);

    std::vector<float> probs(num_cls);
    softmax(logits, probs.data(), num_cls);

    std::vector<TopK> top5;
    top_k(probs.data(), num_cls, 5, top5);

    printf("\n========== Top-5 Classification ==========\n");
    for (int i = 0; i < (int)top5.size(); i++) {
        printf("  #%d: class_id=%-4d  prob=%6.2f%%  %s\n",
               i + 1, top5[i].index, top5[i].prob * 100.0f,
               get_label(labels, top5[i].index));
    }
    printf("==========================================\n");
    //printf("[NOTE] 随机权重模型, 类别无语义; 本例程验证 Transformer 算子在 NPU 上的推理链路与性能\n");

    /* ---------- Step7: 清理 ---------- */
    printf("\n[Step7] Cleanup\n");
    NPU_Model_Unload(h);
    NPU_Destroy(h);
    NPU_Finalize();
    printf("Done\n");
    return 0;
}
