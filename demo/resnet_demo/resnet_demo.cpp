/**
 * ResNet-18 昇腾 NPU 分类推理 Demo
 * 调用 libnpu_driver.a (NPU_* API) 加载 .om 模型进行推理
 *
 * 模型: ResNet-18
 * 输入: 1x3x224x224 (NCHW, float32, RGB)
 * 输出: 1x1000      (ImageNet logits)
 *
 * 预处理: resize(256) -> center crop(224) -> BGR2RGB -> normalize -> NCHW
 * 后处理: softmax -> top-5
 *
 * 用法: ./resnet_demo model.om image.jpg [loop]
 *   loop: benchmark 循环次数, 默认 100
 */
#include "npu_cann_adapter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <filesystem>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

/* ResNet-18 标准预处理参数 */
#define RESNET_INPUT_W   224
#define RESNET_INPUT_H   224
#define RESNET_RESIZE    256      /* 先 resize 到 256, 再 center crop 224 */
#define RESNET_NUM_CLS   1000

/* ImageNet normalize */
static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3]  = {0.229f, 0.224f, 0.225f};

/* ImageNet top-20 常见类名(完整 1000 类可用 imagenet_1000.txt 加载) */
static const char* kImagenetTop[] = {
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "robin", "bulbul", "jay", "magpie", "mockingbird"
    /* ... 共 1000 类, 这里只列前 20, 其余用 class_id 显示 */
};

#define OUTPUT_DIR "./output"

/* ===================== 工具函数 ===================== */

static const char* get_class_name(int class_id) {
    if (class_id >= 0 && class_id < (int)(sizeof(kImagenetTop)/sizeof(kImagenetTop[0])))
        return kImagenetTop[class_id];
    return "unknown";
}

/* 从文件加载 ImageNet 类标签(每行一个类; 逗号后同义词截掉, 只留主名) */
static std::vector<std::string> load_imagenet_labels(const char* path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (!f.is_open()) return labels;
    std::string line;
    while (std::getline(f, line)) {
        /* "tench, Tinca tinca" -> "tench": 只保留逗号前主名 */
        size_t comma = line.find(',');
        if (comma != std::string::npos) line = line.substr(0, comma);
        /* 去掉行尾 \r 与首尾空格 */
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
    return get_class_name(id);
}

/* softmax */
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

/* top-k */
struct TopK {
    int index;
    float prob;
};

static void top_k(const float* probs, int n, int k, std::vector<TopK>& out) {
    std::vector<TopK> all(n);
    for (int i = 0; i < n; i++) { all[i] = {i, probs[i]}; }
    std::partial_sort(all.begin(), all.begin() + k, all.end(),
        [](const TopK& a, const TopK& b) { return a.prob > b.prob; });
    for (int i = 0; i < k; i++) out.push_back(all[i]);
}

/* ===================== 预处理 ===================== */
#ifdef USE_OPENCV
static int preprocess_image(const char* img_path, std::vector<unsigned char>& tensor) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        fprintf(stderr, "[ERROR] Read image fail: %s\n", img_path);
        return -1;
    }
    printf("[INFO] Original image: %dx%d\n", img.cols, img.rows);

    /* Step1: resize 到 256 (保持长宽比, 短边 256) */
    int ow = img.cols, oh = img.rows;
    float scale = (float)RESNET_RESIZE / std::min(ow, oh);
    int rw = (int)(ow * scale);
    int rh = (int)(oh * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(rw, rh), cv::INTER_LINEAR);
    printf("[INFO] Resized: %dx%d -> %dx%d\n", ow, oh, rw, rh);

    /* Step2: center crop 224x224 */
    int x = (rw - RESNET_INPUT_W) / 2;
    int y = (rh - RESNET_INPUT_H) / 2;
    cv::Rect crop_roi(x, y, RESNET_INPUT_W, RESNET_INPUT_H);
    cv::Mat cropped = resized(crop_roi);
    printf("[INFO] Center crop: (%d,%d) %dx%d\n", x, y, RESNET_INPUT_W, RESNET_INPUT_H);

    /* Step3: BGR -> RGB */
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    /* Step4: normalize (x/255 - mean) / std */
    rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);
    cv::Scalar mean(kMean[0], kMean[1], kMean[2]);
    cv::Scalar stddev(kStd[0], kStd[1], kStd[2]);
    cv::Mat normalized;
    cv::subtract(rgb, cv::Scalar(mean), normalized);
    cv::divide(normalized, cv::Scalar(stddev), normalized);

    /* Step5: HWC -> CHW (NCHW, batch=1) */
    tensor.resize(3 * RESNET_INPUT_W * RESNET_INPUT_H * sizeof(float));
    float* dst = reinterpret_cast<float*>(tensor.data());
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < RESNET_INPUT_H; h++) {
            for (int w = 0; w < RESNET_INPUT_W; w++) {
                dst[c * RESNET_INPUT_H * RESNET_INPUT_W + h * RESNET_INPUT_W + w]
                    = normalized.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    printf("[INFO] Tensor: 1x3x224x224 float32, size=%zu bytes\n", tensor.size());
    return 0;
}
#endif /* USE_OPENCV */

/* ===================== Benchmark 辅助 ===================== */
static void print_benchmark(const std::vector<float>& xs, const char* tag) {
    if (xs.empty()) return;
    std::vector<float> v(xs);
    std::sort(v.begin(), v.end());
    float sum = std::accumulate(v.begin(), v.end(), 0.0f);
    int n = (int)v.size();
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

/* ===================== Main ===================== */
int main(int argc, char* argv[]) {
    const char* model_path = "./resnet18.om";
    const char* img_path = "./test.jpg";
    const char* labels_path = "./classname.txt";
    int loop = 100;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) img_path = argv[2];
    if (argc >= 4) loop = atoi(argv[3]);
    if (argc >= 5) labels_path = argv[4];
    if (loop <= 0) loop = 100;

    printf("=============================================\n");
    printf("       ResNet-18 NPU 分类推理 Demo\n");
    printf("Model: %s\nInput: %s\nLoop:  %d\n", model_path, img_path, loop);
    printf("=============================================\n");

    /* 加载标签: 依次尝试 命令行指定 → classname.txt → imagenet_1000.txt
     * (同时探测当前目录与上级目录, 兼容从 build/ 运行) */
    std::vector<std::string> labels;
    const char* candidates[] = {
        labels_path, "classname.txt", "../classname.txt",
        "imagenet_1000.txt", "../imagenet_1000.txt"
    };
    for (const char* cand : candidates) {
        labels = load_imagenet_labels(cand);
        if (!labels.empty()) { labels_path = cand; break; }
    }
    if (labels.empty())
        printf("[WARN] No label file found, class names fallback to builtin table\n");

#ifndef USE_OPENCV
    fprintf(stderr, "[ERROR] 需 OpenCV, 请用 -DUSE_OPENCV 编译\n");
    return -1;
#else
    /* ---------- Step1: 预处理 ---------- */
    printf("\n[Step1] Preprocess image...\n");
    std::vector<unsigned char> input_tensor;
    if (preprocess_image(img_path, input_tensor) != 0) return -1;

    /* ---------- Step2: 创建 NPU 句柄 ---------- */
    printf("\n[Step2] Create NPU handle\n");
    NPUHandle h = NPU_Create();
    if (!h) {
        fprintf(stderr, "[ERROR] NPU_Create failed: %s\n", NPU_GetLastError(h));
        return -1;
    }
    printf("[NPU] Handle created\n");

    /* ---------- Step3: 加载模型 ---------- */
    printf("\n[Step3] Load model\n");
    NPUModelLoadMsg load_msg;
    memset(&load_msg, 0, sizeof(load_msg));
    load_msg.npu_count = 1;
    load_msg.npu_ids[0] = 0;
    load_msg.model_format = NPU_DRIVER_FORMAT_NPU;  /* 强制 NPU (.om) */
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
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[NPU] Model loaded in %.0f ms\n", load_ms);

    /* 查询模型状态 */
    NPUModelStatus status;
    memset(&status, 0, sizeof(status));
    NPU_GetModelStatus(h, &status);
    printf("[NPU] Input:  %u bytes, dims=%d(%dx%dx%dx%d)\n",
           status.input_size, status.input_shape.num_dims,
           status.input_shape.dims[0], status.input_shape.dims[1],
           status.input_shape.dims[2], status.input_shape.dims[3]);
    printf("[NPU] Output: %u bytes, dims=%d(%dx%dx%d)\n",
           status.output_size, status.output_shape.num_dims,
           status.output_shape.dims[0], status.output_shape.dims[1],
           status.output_shape.dims[2]);

    /* ---------- Step4: Warmup ---------- */
    printf("\n[Step4] Warmup (x5)\n");
    std::vector<unsigned char> dummy_out(status.output_size);
    unsigned int dummy_len = status.output_size;
    for (int i = 0; i < 5; i++) {
        unsigned int out_len = (unsigned int)dummy_out.size();
        ret = NPU_Model_Infer(h, input_tensor.data(), (unsigned int)input_tensor.size(),
                              dummy_out.data(), &out_len);
        if (ret != 0) {
            fprintf(stderr, "[ERROR] Warmup infer %d failed: %s\n", i, NPU_GetLastError(h));
            break;
        }
        dummy_len = out_len;
    }
    printf("Warmup done\n");

    /* ---------- Step5: Benchmark ---------- */
    printf("\n[Step5] Benchmark loop=%d\n", loop);

    /* 创建 Event 用于设备端计时 */
    NPUEvent ev_start = nullptr, ev_end = nullptr;
    NPU_EventCreate(h, &ev_start);
    NPU_EventCreate(h, &ev_end);

    std::vector<float> dev_ms_list, host_ms_list;
    dev_ms_list.reserve(loop);
    host_ms_list.reserve(loop);

    for (int i = 0; i < loop; i++) {
        unsigned int out_len = (unsigned int)dummy_out.size();

        NPU_EventRecord(h, ev_start, nullptr);
        auto hs = std::chrono::high_resolution_clock::now();

        ret = NPU_Model_Infer(h, input_tensor.data(), (unsigned int)input_tensor.size(),
                              dummy_out.data(), &out_len);
        if (ret != 0) {
            fprintf(stderr, "[ERROR] Infer %d failed: %s\n", i, NPU_GetLastError(h));
            break;
        }

        auto he = std::chrono::high_resolution_clock::now();
        NPU_EventRecord(h, ev_end, nullptr);
        NPU_EventSynchronize(h, ev_end);

        float dev_ms = 0.0f;
        NPU_EventElapsedTime(h, &dev_ms, ev_start, ev_end);
        dev_ms_list.push_back(dev_ms);
        host_ms_list.push_back((float)std::chrono::duration<double, std::milli>(he - hs).count());
    }

    printf("------------------------------------------------------------\n");
    print_benchmark(dev_ms_list, "Device (Event) ");
    print_benchmark(host_ms_list, "Host   (chrono) ");
    printf("------------------------------------------------------------\n");

    if (ev_start) NPU_EventDestroy(h, ev_start);
    if (ev_end) NPU_EventDestroy(h, ev_end);

    /* ---------- Step6: 后处理 + 分类结果 ---------- */
    printf("\n[Step6] Post process\n");
    float* logits = reinterpret_cast<float*>(dummy_out.data());
    int num_cls = (int)(dummy_len / sizeof(float));
    if (num_cls > RESNET_NUM_CLS) num_cls = RESNET_NUM_CLS;
    printf("[INFO] Output: %d classes\n", num_cls);

    /* softmax */
    std::vector<float> probs(num_cls);
    softmax(logits, probs.data(), num_cls);

    /* top-5 */
    std::vector<TopK> top5;
    top_k(probs.data(), num_cls, 5, top5);

    printf("\n========== Top-5 Classification ==========\n");
    for (int i = 0; i < (int)top5.size(); i++) {
        printf("  #%d: class_id=%-4d  prob=%6.2f%%  %s\n",
               i + 1, top5[i].index, top5[i].prob * 100.0f,
               get_label(labels, top5[i].index));
    }
    printf("==========================================\n");

    /* ---------- Step7: 清理 ---------- */
    printf("\n[Step7] Cleanup\n");
    NPU_Model_Unload(h);
    NPU_Destroy(h);
    NPU_Finalize();
    printf("[Done] Inference finished\n");

    return 0;
#endif
}
