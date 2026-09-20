/**
 * YOLOv5 预处理 / 后处理公共函数
 * 供 yolo_demo(走 libnpu_driver)和 acl_raw_demo(走官方 ACL)共用,
 * 保证两边比较的是同一套预处理 / 后处理逻辑,只差推理路径。
 */
#ifndef YOLO_COMMON_H
#define YOLO_COMMON_H

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <filesystem>

#define YOLO_INPUT_W 640
#define YOLO_INPUT_H 640
#define YOLO_NUM_CLS 80
#define YOLO_PAD_COLOR 114.f

struct DetectBox {
    float x, y, w, h;
    float conf;
    int class_id;
};

struct LetterBoxParam {
    float scale;
    int dw;
    int dh;
};

/* 阈值:可被各 demo 覆盖 */
struct YoloThresholds {
    float conf = 0.25f;
    float obj_pass = 0.10f;
    float nms = 0.45f;
};

/* YOLOv5 anchors + grid */
static const float kAnchors[3][6] = {
    {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
    {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
    {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
};
static const int kGridSizes[3] = {80, 40, 20};

static const char* kClassNames[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

static const float kColorList[80][3] = {
    {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125}, {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933}, {0.635, 0.078, 0.184}, {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600}, {1.000, 0.000, 0.000}, {1.000, 0.500, 0.000}, {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 1.000}, {0.667, 0.000, 1.000}, {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000}, {0.333, 1.000, 0.000}, {0.667, 0.333, 0.000}, {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000}, {1.000, 0.333, 0.000}, {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500}, {0.000, 0.667, 0.500}, {0.000, 1.000, 0.500}, {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500}, {0.333, 0.667, 0.500}, {0.333, 1.000, 0.500}, {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500}, {0.667, 0.667, 0.500}, {0.667, 1.000, 0.500}, {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500}, {1.000, 0.667, 0.500}, {1.000, 1.000, 0.500}, {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000}, {0.333, 0.000, 1.000}, {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000}, {0.333, 1.000, 1.000}, {0.667, 0.000, 1.000}, {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000}, {0.667, 1.000, 1.000}, {1.000, 0.000, 1.000}, {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000}, {0.333, 0.000, 0.000}, {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000}, {1.000, 0.000, 0.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000}, {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167}, {0.000, 0.000, 0.333}, {0.000, 0.000, 0.500}, {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000}, {0.000, 0.000, 1.000}, {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571}, {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857}, {0.000, 0.447, 0.741}, {0.314, 0.717, 0.741}, {0.50, 0.5, 0}
};

/* ===================== IOU + NMS ===================== */
static float yolo_iou(const DetectBox& a, const DetectBox& b) {
    float x1 = a.x - a.w / 2, y1 = a.y - a.h / 2;
    float x2 = a.x + a.w / 2, y2 = a.y + a.h / 2;
    float x1b = b.x - b.w / 2, y1b = b.y - b.h / 2;
    float x2b = b.x + b.w / 2, y2b = b.y + b.h / 2;
    float iw = std::max(0.f, std::min(x2, x2b) - std::max(x1, x1b));
    float ih = std::max(0.f, std::min(y2, y2b) - std::max(y1, y1b));
    float inter = iw * ih;
    float ua = (x2 - x1) * (y2 - y1) + (x2b - x1b) * (y2b - y1b) - inter;
    return inter / (ua + 1e-6f);
}

static void yolo_nms(std::vector<DetectBox>& dets, float iou_thresh) {
    if (dets.empty()) return;
    std::sort(dets.begin(), dets.end(),
              [](const DetectBox& a, const DetectBox& b) { return a.conf > b.conf; });
    std::vector<bool> keep(dets.size(), true);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!keep[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                yolo_iou(dets[i], dets[j]) > iou_thresh)
                keep[j] = false;
        }
    }
    std::vector<DetectBox> res;
    for (size_t i = 0; i < dets.size(); ++i)
        if (keep[i]) res.push_back(dets[i]);
    dets.swap(res);
}

/* ===================== 文件读取 ===================== */
static int read_binary_file(const char* path, std::vector<unsigned char>& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return -1;
    std::streamsize sz = f.tellg();
    f.seekg(0);
    data.resize(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return 0;
}

/* ===================== YOLOv5 后处理接口(v2:按输出布局自动分发) ===================== */
typedef enum {
    YOLO_LAYOUT_UNKNOWN = 0,
    YOLO_LAYOUT_CLASSIC = 1,      /* [1, N, 85] 框优先: raw logit, 需 sigmoid+grid+anchor */
    YOLO_LAYOUT_ULTRALYTICS = 2   /* [1, ch, N] 属性优先(转置): 已解码, ch=84 无 obj / ch=85 带 obj */
} YoloOutLayout;

struct YoloOutputLayout {
    YoloOutLayout layout = YOLO_LAYOUT_UNKNOWN;
    int ch = 85;      /* 每框通道数 */
    int n  = 25200;   /* 框数 */
};

/*
 * 检测输出布局:
 *   1) 优先用模型声明的输出 dims (如 {1,84,8400}; 无则传 NULL/0)
 *   2) 兜底用首次推理返回的实际输出字节数 actual_bytes (无则传 0)
 */
static YoloOutputLayout yolo_detect_layout(const int* dims, int num_dims, size_t actual_bytes) {
    YoloOutputLayout lay;
    if (dims && num_dims == 3) {
        int d2 = dims[1], d3 = dims[2];
        if (d2 >= 84 && d2 <= 85 && d3 > d2 * 8) {
            lay.layout = YOLO_LAYOUT_ULTRALYTICS; lay.ch = d2; lay.n = d3;
            return lay;
        }
        if (d3 >= 84 && d3 <= 85 && d2 > d3 * 8) {
            lay.layout = YOLO_LAYOUT_CLASSIC; lay.ch = d3; lay.n = d2;
            return lay;
        }
    }
    if (actual_bytes >= sizeof(float) * 84) {
        size_t nf = actual_bytes / sizeof(float);
        if (nf % 84 == 0 && nf / 84 >= 1000) {
            lay.layout = YOLO_LAYOUT_ULTRALYTICS; lay.ch = 84; lay.n = (int)(nf / 84);
        } else if (nf % 85 == 0 && nf / 85 >= 1000 && (int)(nf / 85) != 25200) {
            lay.layout = YOLO_LAYOUT_ULTRALYTICS; lay.ch = 85; lay.n = (int)(nf / 85);
        } else if (nf % 85 == 0) {
            lay.layout = YOLO_LAYOUT_CLASSIC; lay.ch = 85; lay.n = (int)(nf / 85);
        }
    }
    return lay;
}

static void yolo_print_layout(const YoloOutputLayout& lay) {
    if (lay.layout == YOLO_LAYOUT_ULTRALYTICS) {
        printf("[INFO] Output layout: [1,%d,%d] transposed (ultralytics anchor-free, "
               "boxes pre-decoded, no sigmoid/grid needed)\n", lay.ch, lay.n);
    } else if (lay.layout == YOLO_LAYOUT_CLASSIC) {
        printf("[INFO] Output layout: [1,%d,%d] classic (anchor-based, "
               "raw logits, sigmoid+grid+anchor decode)\n", lay.n, lay.ch);
    } else {
        printf("[WARN] Output layout unknown, fallback to classic 25200x85\n");
    }
}

static void yolo_print_first_det(const std::vector<DetectBox>& dets) {
    if (dets.empty()) return;
    const DetectBox& d = dets[0];
    printf("First det (640坐标) cx=%.2f cy=%.2f w=%.2f h=%.2f conf=%.2f cls=%d(%s)\n",
           d.x, d.y, d.w, d.h, d.conf, d.class_id,
           (d.class_id >= 0 && d.class_id < YOLO_NUM_CLS) ? kClassNames[d.class_id] : "unk");
}

/* 经典布局解码: sigmoid + grid + anchor 还原 (标准 25200 框三导出层) */
static void yolo_decode_classic(const float* output, const YoloOutputLayout& lay,
                                const YoloThresholds& th, std::vector<DetectBox>& dets) {
    const size_t valid = (lay.ch > 0) ? (size_t)lay.n * (size_t)lay.ch
                                      : (size_t)25200 * 85;
    bool overflow = false;
    int ptr = 0;
    for (int layer = 0; layer < 3 && !overflow; layer++) {
        int gs = kGridSizes[layer];
        const float* anc = kAnchors[layer];
        for (int y = 0; y < gs && !overflow; y++) {
            for (int x = 0; x < gs && !overflow; x++) {
                for (int a = 0; a < 3; a++) {
                    if ((size_t)ptr + 85 > valid) { overflow = true; break; }
                    float cx = output[ptr + 0];
                    float cy = output[ptr + 1];
                    float w  = output[ptr + 2];
                    float h  = output[ptr + 3];
                    float obj = output[ptr + 4];
                    float obj_sig = 1.0f / (1.0f + std::exp(-obj));
                    if (obj_sig < th.obj_pass) { ptr += 85; continue; }
                    cx = 1.0f / (1.0f + std::exp(-cx));
                    cy = 1.0f / (1.0f + std::exp(-cy));
                    w  = 1.0f / (1.0f + std::exp(-w));
                    h  = 1.0f / (1.0f + std::exp(-h));
                    float bx = (cx * 2 - 0.5f + x) * (YOLO_INPUT_W / gs);
                    float by = (cy * 2 - 0.5f + y) * (YOLO_INPUT_H / gs);
                    float bw = std::pow(w * 2, 2) * anc[a * 2 + 0];
                    float bh = std::pow(h * 2, 2) * anc[a * 2 + 1];
                    float max_cls = 0; int cls_id = 0;
                    for (int c = 0; c < YOLO_NUM_CLS; c++) {
                        float s = 1.0f / (1.0f + std::exp(-output[ptr + 5 + c]));
                        if (s > max_cls) { max_cls = s; cls_id = c; }
                    }
                    float conf = obj_sig * max_cls;
                    if (conf > th.conf) {
                        dets.push_back({bx, by, bw, bh, conf, cls_id});
                    }
                    ptr += 85;
                }
            }
        }
    }
    yolo_print_first_det(dets);
}

/* ultralytics 转置布局解码:
 * 行 0..3 = cx,cy,w,h (已是 640 尺度绝对像素)
 * ch==85 时行 4 = objectness; 其余行 = 类别分数 (已 sigmoid)
 * 无需 sigmoid / grid / anchor 还原
 */
static void yolo_decode_ultralytics(const float* out, const YoloOutputLayout& lay,
                                    const YoloThresholds& th, std::vector<DetectBox>& dets) {
    const int n = lay.n;
    const bool has_obj = (lay.ch == 85);
    const int cls_off = has_obj ? 5 : 4;
    for (int i = 0; i < n; i++) {
        float cx = out[0 * n + i];
        float cy = out[1 * n + i];
        float w  = out[2 * n + i];
        float h  = out[3 * n + i];
        if (w <= 1.0f || h <= 1.0f) continue;   /* 无效小框 */
        float obj = has_obj ? out[4 * n + i] : 1.0f;
        const float* sc = out + cls_off * n;
        float best = 0.0f; int cls_id = 0;
        for (int c = 0; c < YOLO_NUM_CLS; c++) {
            float s = sc[c * n + i];
            if (s > best) { best = s; cls_id = c; }
        }
        float conf = has_obj ? obj * best : best;
        if (conf > th.conf) {
            dets.push_back({cx, cy, w, h, conf, cls_id});
        }
    }
    yolo_print_first_det(dets);
}

/* 解码总入口: 按 layout 自动分发 */
static void yolo_decode(const float* out, const YoloOutputLayout& lay,
                        const YoloThresholds& th, std::vector<DetectBox>& dets) {
    if (lay.layout == YOLO_LAYOUT_ULTRALYTICS)
        yolo_decode_ultralytics(out, lay, th, dets);
    else
        yolo_decode_classic(out, lay, th, dets);
}

/* 一站式后处理: 解码 + NMS 极大值抑制 */
static void yolo_postprocess(const float* out, const YoloOutputLayout& lay,
                             const YoloThresholds& th, std::vector<DetectBox>& dets) {
    dets.clear();
    yolo_decode(out, lay, th, dets);
    yolo_nms(dets, th.nms);
}

/* 旧接口兼容: 经典布局解码 (默认 25200x85) */
static void decode_yolov5(const float* output, std::vector<DetectBox>& dets,
                          const YoloThresholds& th) {
    YoloOutputLayout lay;
    yolo_decode_classic(output, lay, th, dets);
}

/* ===================== OpenCV 预处理 / 后处理 ===================== */
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>

static cv::Mat letterbox(cv::Mat src, LetterBoxParam& param) {
    int w = src.cols, h = src.rows;
    float scale = std::min((float)YOLO_INPUT_W / w, (float)YOLO_INPUT_H / h);
    int nw = std::round(w * scale), nh = std::round(h * scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), cv::INTER_LINEAR);
    int padw = YOLO_INPUT_W - nw, padh = YOLO_INPUT_H - nh;
    int left = padw / 2, top = padh / 2;
    cv::Mat dst;
    cv::copyMakeBorder(resized, dst, top, padh - top, left, padw - left,
                       cv::BORDER_CONSTANT, cv::Scalar(YOLO_PAD_COLOR, YOLO_PAD_COLOR));
    param.scale = scale; param.dw = left; param.dh = top;
    return dst;
}

static int load_image_tensor(const char* img_path, std::vector<unsigned char>& tensor,
                             LetterBoxParam& param) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) { printf("[ERROR] Read image fail: %s\n", img_path); return -1; }
    cv::Mat lb = letterbox(img, param);
    lb.convertTo(lb, CV_32FC3, 1.0f / 255.f);
    tensor.resize(3 * YOLO_INPUT_W * YOLO_INPUT_H * sizeof(float));
    float* dst = reinterpret_cast<float*>(tensor.data());
    for (int y = 0; y < YOLO_INPUT_H; y++) {
        for (int x = 0; x < YOLO_INPUT_W; x++) {
            cv::Vec3f pix = lb.at<cv::Vec3f>(y, x);
            dst[0 * YOLO_INPUT_W * YOLO_INPUT_H + y * YOLO_INPUT_W + x] = pix[2];
            dst[1 * YOLO_INPUT_W * YOLO_INPUT_H + y * YOLO_INPUT_W + x] = pix[1];
            dst[2 * YOLO_INPUT_W * YOLO_INPUT_H + y * YOLO_INPUT_W + x] = pix[0];
        }
    }
    printf("[INFO] Img %dx%d -> letterbox %dx%d scale=%.3f pad dw=%d dh=%d\n",
           img.cols, img.rows, YOLO_INPUT_W, YOLO_INPUT_H,
           param.scale, param.dw, param.dh);
    return 0;
}

static void save_result_img(const std::string& src_img_path,
                           const std::vector<DetectBox>& dets,
                           const LetterBoxParam& param,
                           const char* out_dir) {
    cv::Mat img = cv::imread(src_img_path);
    if (img.empty()) return;
    float scale = param.scale;
    for (const auto& d : dets) {
        float real_cx = (d.x - param.dw) / scale;
        float real_cy = (d.y - param.dh) / scale;
        float real_w = d.w / scale, real_h = d.h / scale;
        float x1 = std::max(0.0f, real_cx - real_w / 2.0f);
        float y1 = std::max(0.0f, real_cy - real_h / 2.0f);
        float x2 = std::min((float)img.cols - 1.0f, real_cx + real_w / 2.0f);
        float y2 = std::min((float)img.rows - 1.0f, real_cy + real_h / 2.0f);
        int cid = d.class_id; if (cid >= 80) cid = 0;
        cv::Scalar color((int)(kColorList[cid][0] * 255),
                         (int)(kColorList[cid][1] * 255),
                         (int)(kColorList[cid][2] * 255));
        cv::rectangle(img, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), color, 2);
        char buf[128];
        const char* name = (cid < YOLO_NUM_CLS) ? kClassNames[cid] : "unk";
        snprintf(buf, sizeof(buf), "%s %.2f", name, d.conf);
        cv::Size sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
        cv::rectangle(img, cv::Point((int)x1, (int)y1 - sz.height - 4),
                     cv::Point((int)x1 + sz.width, (int)y1), color, -1);
        cv::putText(img, buf, cv::Point((int)x1, (int)y1 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    namespace fs = std::filesystem;
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);
    size_t pos = src_img_path.find_last_of("/\\");
    std::string fname = (pos == std::string::npos) ? src_img_path : src_img_path.substr(pos + 1);
    std::string out_path = std::string(out_dir) + "/" + fname;
    cv::imwrite(out_path, img);
    printf("[INFO] Saved result: %s\n", out_path.c_str());
}
#endif /* USE_OPENCV */

#endif /* YOLO_COMMON_H */
