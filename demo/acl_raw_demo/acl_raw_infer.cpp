/**
 * 纯官方 ACL 推理 Demo (不依赖 libnpu_driver)
 * 直接调用 acl/acl_mdl.h, 用同一 .om 模型 + 同一张图推理,
 * 用于和 yolo_demo(走 libnpu_driver)对比结果一致性。
 *
 * 用法: ./acl_raw_infer model.om img.jpg [loop] [conf] [nms]
 * 输出: ./output/<img>.jpg (带检测框) + benchmark 统计
 */
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>

#include "yolo_common.h"

#define ACL_RAW_OUTPUT_DIR "./output"

static const char* acl_err_str(aclError e) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "aclErr=%d", (int)e);
    return buf;
}

/* 打印模型 IO 信息(用 aclmdlDesc 查询) */
static void print_model_io(aclmdlDesc* desc) {
    size_t n_in = aclmdlGetNumInputs(desc);
    size_t n_out = aclmdlGetNumOutputs(desc);
    printf("[Model] inputs=%zu  outputs=%zu\n", n_in, n_out);
    for (size_t i = 0; i < n_in; i++) {
        printf("  in[%zu] size=%zu bytes\n", i, aclmdlGetInputSizeByIndex(desc, i));
    }
    for (size_t i = 0; i < n_out; i++) {
        printf("  out[%zu] size=%zu bytes\n", i, aclmdlGetOutputSizeByIndex(desc, i));
    }
}

int main(int argc, char* argv[]) {
    const char* model_path = "./yolov5s.om";
    const char* img_path = "./test.jpg";
    int loop = 100;
    YoloThresholds th;
    th.conf = 0.25f; th.obj_pass = 0.51f; th.nms = 0.45f;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) img_path = argv[2];
    if (argc >= 4) loop = atoi(argv[3]);
    if (argc >= 5) th.conf = (float)atof(argv[4]);
    if (argc >= 6) th.nms = (float)atof(argv[5]);
    if (argc >= 7) th.obj_pass = (float)atof(argv[6]);
    if (loop <= 0) loop = 100;
    if (th.conf <= 0) th.conf = 0.25f;
    if (th.nms <= 0) th.nms = 0.45f;
    if (th.obj_pass <= 0) th.obj_pass = 0.51f;

    printf("=============================================\n");
    printf("     YOLOv5 纯 ACL 推理 (无 libnpu_driver)\n");
    printf("Model: %s\nInput: %s\nLoop: %d\n", model_path, img_path, loop);
    printf("Thresholds: conf=%.2f  nms=%.2f  obj-pass=%.2f\n", th.conf, th.nms, th.obj_pass);
    printf("Output dir: %s\n", ACL_RAW_OUTPUT_DIR);
    printf("=============================================\n");

    /* ---------- 1. 初始化 ACL ---------- */
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        fprintf(stderr, "aclInit fail: %s\n", acl_err_str(ret));
        return -1;
    }
    int32_t dev_id = 0;
    ret = aclrtSetDevice(dev_id);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "aclrtSetDevice fail: %s\n", acl_err_str(ret)); aclFinalize(); return -1; }

    aclrtContext ctx = nullptr;
    ret = aclrtCreateContext(&ctx, dev_id);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "aclrtCreateContext fail: %s\n", acl_err_str(ret)); aclrtResetDevice(dev_id); aclFinalize(); return -1; }
    aclrtSetCurrentContext(ctx);

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "aclrtCreateStream fail: %s\n", acl_err_str(ret)); aclrtDestroyContext(ctx); aclrtResetDevice(dev_id); aclFinalize(); return -1; }

    /* ---------- 2. 加载模型 ---------- */
    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t model_id = 0;
    ret = aclmdlLoadFromFile(model_path, &model_id);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclmdlLoadFromFile fail: %s (path=%s)\n", acl_err_str(ret), model_path);
        aclrtDestroyStream(stream); aclrtDestroyContext(ctx); aclrtResetDevice(dev_id); aclFinalize();
        return -1;
    }
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[Step1] Model loaded in %.0f ms, model_id=%u\n", load_ms, model_id);

    aclmdlDesc* desc = aclmdlCreateDesc();
    ret = aclmdlGetDesc(desc, model_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclmdlGetDesc fail: %s\n", acl_err_str(ret));
        aclmdlUnload(model_id); aclrtDestroyStream(stream); aclrtDestroyContext(ctx);
        aclrtResetDevice(dev_id); aclFinalize();
        return -1;
    }
    print_model_io(desc);

    /* ---------- 3. 预处理图片 ---------- */
#ifdef USE_OPENCV
    std::vector<unsigned char> input_tensor;
    LetterBoxParam lb_param{1.0f, 0, 0};
    if (load_image_tensor(img_path, input_tensor, lb_param) != 0) {
        aclmdlUnload(model_id); aclmdlDestroyDesc(desc); aclrtDestroyStream(stream);
        aclrtDestroyContext(ctx); aclrtResetDevice(dev_id); aclFinalize();
        return -1;
    }
#else
    fprintf(stderr, "[ERROR] 需 OpenCV,请用 -DUSE_OPENCV 编译\n");
    aclmdlUnload(model_id); aclmdlDestroyDesc(desc); aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx); aclrtResetDevice(dev_id); aclFinalize();
    return -1;
#endif

    /* ---------- 4. 分配 device 内存 + dataset ---------- */
    size_t in_size = aclmdlGetInputSizeByIndex(desc, 0);
    size_t out_size = aclmdlGetOutputSizeByIndex(desc, 0);
    if (in_size < input_tensor.size()) {
        fprintf(stderr, "[WARN] model input size %zu < tensor size %zu, 截断\n", in_size, input_tensor.size());
    }
    size_t copy_in = std::min(in_size, input_tensor.size());

    void* in_dev = nullptr;
    void* out_dev = nullptr;
    ret = aclrtMalloc(&in_dev, in_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc(in) fail: %s\n", acl_err_str(ret));
        aclmdlUnload(model_id); aclmdlDestroyDesc(desc); aclrtDestroyStream(stream);
        aclrtDestroyContext(ctx); aclrtResetDevice(dev_id); aclFinalize();
        return -1;
    }
    ret = aclrtMalloc(&out_dev, out_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc(out) fail: %s\n", acl_err_str(ret));
        aclrtFree(in_dev); aclmdlUnload(model_id); aclmdlDestroyDesc(desc);
        aclrtDestroyStream(stream); aclrtDestroyContext(ctx); aclrtResetDevice(dev_id); aclFinalize();
        return -1;
    }

    aclmdlDataset* in_dataset = aclmdlCreateDataset();
    aclmdlDataset* out_dataset = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(in_dataset, aclCreateDataBuffer(in_dev, in_size));
    aclmdlAddDatasetBuffer(out_dataset, aclCreateDataBuffer(out_dev, out_size));

    /* ---------- 5. Warmup ---------- */
    printf("\n[Step4] Warmup (x5)\n");
    for (int i = 0; i < 5; i++) {
        ret = aclrtMemcpy(in_dev, in_size, input_tensor.data(), copy_in, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { fprintf(stderr, "H2D warmup fail: %s\n", acl_err_str(ret)); break; }
        ret = aclmdlExecute(model_id, in_dataset, out_dataset);
        if (ret != ACL_SUCCESS) { fprintf(stderr, "Execute warmup fail: %s\n", acl_err_str(ret)); break; }
    }
    printf("Warmup done\n");

    /* ---------- 6. Benchmark loop ---------- */
    std::vector<float> dev_ms_list, host_ms_list;
    dev_ms_list.reserve(loop);
    host_ms_list.reserve(loop);

    aclrtEvent ev_start = nullptr, ev_end = nullptr;
    aclrtCreateEvent(&ev_start);
    aclrtCreateEvent(&ev_end);

    printf("\n[Step5] Benchmark loop=%d\n", loop);
    for (int i = 0; i < loop; i++) {
        aclrtRecordEvent(ev_start, stream);
        auto hs = std::chrono::high_resolution_clock::now();
        ret = aclrtMemcpy(in_dev, in_size, input_tensor.data(), copy_in, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { fprintf(stderr, "H2D loop %d fail: %s\n", i, acl_err_str(ret)); break; }
        ret = aclmdlExecute(model_id, in_dataset, out_dataset);
        if (ret != ACL_SUCCESS) { fprintf(stderr, "Execute loop %d fail: %s\n", i, acl_err_str(ret)); break; }
        aclrtRecordEvent(ev_end, stream);
        aclrtSynchronizeEvent(ev_end);
        auto he = std::chrono::high_resolution_clock::now();

        float dev_ms = 0.0f;
        aclrtEventElapsedTime(&dev_ms, ev_start, ev_end);
        dev_ms_list.push_back(dev_ms);
        double host_ms = std::chrono::duration<double, std::milli>(he - hs).count();
        host_ms_list.push_back((float)host_ms);
    }

    if (ev_start) aclrtDestroyEvent(ev_start);
    if (ev_end) aclrtDestroyEvent(ev_end);

    auto summarize = [](const std::vector<float>& xs, const char* tag) {
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
    };
    printf("------------------------------------------------------------\n");
    summarize(dev_ms_list, "Device (Event) ");
    summarize(host_ms_list, "Host   (chrono) ");
    printf("------------------------------------------------------------\n");

    /* ---------- 7. D2H + decode + NMS + 保存 ---------- */
    std::vector<unsigned char> out_buf(out_size);
    ret = aclrtMemcpy(out_buf.data(), out_size, out_dev, out_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "D2H fail: %s\n", acl_err_str(ret));
    } else {
        float* out_ptr = reinterpret_cast<float*>(out_buf.data());
        std::vector<DetectBox> dets;
        decode_yolov5(out_ptr, dets, th);
        printf("Raw det: %zu\n", dets.size());
        yolo_nms(dets, th.nms);
        printf("After NMS: %zu\n", dets.size());
#ifdef USE_OPENCV
        save_result_img(img_path, dets, lb_param, ACL_RAW_OUTPUT_DIR);
#endif
    }

    /* ---------- 8. 清理 ---------- */
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(in_dataset); i++) {
        aclDestroyDataBuffer(aclmdlGetDatasetBuffer(in_dataset, i));
    }
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(out_dataset); i++) {
        aclDestroyDataBuffer(aclmdlGetDatasetBuffer(out_dataset, i));
    }
    aclmdlDestroyDataset(in_dataset);
    aclmdlDestroyDataset(out_dataset);
    aclrtFree(out_dev);
    aclrtFree(in_dev);
    aclmdlUnload(model_id);
    aclmdlDestroyDesc(desc);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(dev_id);
    aclFinalize();
    printf("[Done] aclFinalize ok\n");
    return 0;
}
