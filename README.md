# NPU CANN 推理驱动工程

基于华为昇腾 CANN 8.0 SDK 的 NPU 模型推理驱动,运行于 Ascend 310B1 (ARM64),为上层应用提供统一的模型加载、推理、热更新、Profiling 等 C API。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     应用层 (Demo)                            │
│  yolo_demo  resnet_demo  trans_demo  npu_demo  acl_raw_demo │
└──────────────────────┬──────────────────────────────────────┘
                       │  #include "npu_cann_adapter.h"
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               libnpu_driver (静态库)                         │
│                                                             │
│  核心接口:                                                   │
│  ├─ 模型生命周期: Create / Model_Load / Infer / Unload      │
│  ├─ 性能计时:    EventRecord / EventElapsedTime              │
│  ├─ 异步推理:    Model_InferAsync / StreamCreate             │
│  ├─ 资源监控:    Monitor_* (内存峰值/利用率曲线/PNG 导出)    │
│  └─ Profiling:   LayerProfiling_* (算子级耗时 + HTML 报告)   │
└──────────────────────┬──────────────────────────────────────┘
                       │ target_link_libraries PUBLIC
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   底层依赖 (自动链接)                         │
│  libascendcl.so  libmsprofiler.so  libonnxruntime.so        │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
workspace/
├── CMakeLists.txt              # 顶层 CMake: 依赖查找 + 统一构建
├── README.md
│
├── lib/                        # NPU 驱动核心库
│   ├── CMakeLists.txt          # 编译 libnpu_driver.a, 导出 target
│   ├── npu_cann_adapter.h      # 对外 C API 头文件 (~40 个接口)
│   └── npu_cann_adapter.cpp    # 实现 (封装 CANN ACL + msprof)
│
├── demo/                       # 示例程序 (全部通过顶层 CMake 构建)
│   ├── CMakeLists.txt          # add_subdirectory 所有 demo
│   ├── yolo_common.h           # YOLOv5 预处理/后处理 (letterbox/NMS)
│   ├── yolo_demo/              # YOLOv5 目标检测
│   ├── resnet_demo/            # ResNet-18 图像分类
│   ├── trans_demo/             # TinyViT Transformer 分类 (带 Profiling)
│   ├── npu_demo/               # 通用接口测试 (依次调用所有 API)
│   └── acl_raw_demo/           # 纯 ACL 推理 (不依赖 npu_driver, 用于对比验证)
│
└── model/                      # 模型转换工具
    ├── check_onnx_shpae.py     # ONNX 形状检查 + ATC 命令生成
    └── kernel_meta/            # ATC 算子融合中间产物 (可删)
```

## 依赖环境

| 组件 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 22.04 LTS (aarch64) | 目标开发板 |
| NPU | Ascend 310B1 | 昇腾推理卡 |
| CANN | 8.0.0.alpha001 | ascend-toolkit (ACL + ATC + msprof) |
| ONNX Runtime | ≥ 1.15 | CPU 后端 (可选, 用于 .onnx 直接推理) |
| OpenCV | ≥ 4.0 | 图像处理 + 可视化 (可选) |
| GCC | 11.3+ | C++17 |

## 构建

### 前置: 加载环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 一键构建 (推荐)

```bash
cd /root/workspace
rm -rf build && mkdir build && cd build
cmake ..
make -j4
```

产物统一放在 `build/` 下:
```
build/
├── libnpu_driver.a         # 静态库
├── yolo_demo               # YOLOv5 目标检测
├── resnet_demo             # ResNet-18 分类
├── trans_demo              # TinyViT 分类
├── npu_demo                # 通用接口测试
└── acl_raw_infer           # 纯 ACL 推理
```

### CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `-DUSE_ONNXRUNTIME=OFF` | ON | 关闭 ONNX Runtime 支持 |
| `-DUSE_OPENCV=OFF` | ON | 关闭 OpenCV |
| `-DBUILD_DEMOS=OFF` | ON | 只编译静态库, 不编译 demo |
| `-DCANN_ROOT=/path/to/toolkit` | 自动探测 | 手动指定 CANN 路径 |

### 独立构建某个 demo

如果只想编译某个 demo, 也可以进入其目录单独构建 (旧路径仍然兼容):
```bash
cd demo/trans_demo
mkdir -p build && cd build
cmake ..
make -j4
```

## 运行 Demo

### 1. YOLOv5 目标检测

```bash
./build/yolo_demo model/yolov5s.om test.jpg 0
# 参数: [model.om] [image.jpg] [mode: 0=auto 1=NPU 2=ONNX]
# 输出: output/result.jpg (带检测框)
```

### 2. ResNet-18 图像分类

```bash
cd demo/resnet_demo && ../../build/resnet_demo model/resnet18.om infer/dog.jpg 100
# 参数: [model.om] [image.jpg] [loop: benchmark 次数, 默认 100]
# 输出: Top-5 分类结果 + 平均推理耗时
```

### 3. TinyViT Transformer 分类 

```bash
cd demo/trans_demo
../../build/trans_demo model/tiny_vit.om infer/dog.jpg 3 ./prof_out
# 参数: [model.om] [image.jpg] [loop] [labels|prof_dir|monitor[:ms]] ...
#   传 prof_dir 即启用 Profiling; 传 monitor[:采样间隔ms] 开启实时资源监控
# 输出: Top-5 分类 + 算子级耗时统计 + HTML 报告 + 监控曲线 PNG
```

Profiling 产物 (每次运行自动放进对应 PROF_xxx 时间戳目录):
```
prof_out/PROF_000001_<时间戳>_<随机ID>/
├── device_0/
├── host/
├── mindstudio_profiler_output/
│   ├── op_summary_*.csv          # 每个算子详细耗时
│   ├── op_statistic_*.csv        # 按 op_type 聚合统计
│   ├── msprof_*.json             # Chrome Tracing 时间线
│   └── README.txt                # 各文件字段说明
└── profiling_report.html         # 自包含 HTML 可视化报告 ← 浏览器打开
```

监控曲线产物 (传 `monitor[:ms]` 时生成):
```
monitor/monitor_YYYYMMDD_HHMMSS.png   # 内存+利用率曲线图 (Y 轴自适应放大)
```

> 更详细的参数说明、执行流程与注意事项见 [demo/trans_demo/README.md](demo/trans_demo/README.md)。

### 4. 纯 ACL 推理 (acl_raw_demo)

不链接 libnpu_driver, 直接调用 CANN ACL API, 用于对比验证 lib 的正确性:
```bash
cd demo/acl_raw_demo && ../../build/acl_raw_infer ../yolo_demo/model/yolov5s.om ../yolo_demo/test.jpg
```

## 核心 API

### 模型生命周期

| 接口 | 说明 |
|------|------|
| `NPU_GetNpuCount()` | 获取 NPU 数量 |
| `NPU_Create()` / `NPU_Destroy()` | 创建 / 销毁 NPU 实例 |
| `NPU_Model_Load(handle, model_ptr, len)` | 加载 .om 模型 |
| `NPU_Model_Unload(handle)` | 卸载模型 |
| `NPU_Model_Infer(handle, input, size)` | 同步推理 |
| `NPU_Model_InferAsync(handle, stream, input, size)` | 异步推理 |
| `NPU_Model_Update(handle, new_model_ptr, len)` | 模型热更新 |
| `NPU_GetModelShape(data, len, &shape)` | 离线获取模型 I/O 形状 |
| `NPU_GetModelStatus(handle, &status)` | 获取模型状态 (热更新验证用) |
| `NPU_Get_Infer_Result(handle, buf, size)` | 读取最近一次推理结果 |
| `NPU_GetLastError(handle)` | 获取最近错误信息 (线程安全) |
| `NPU_Finalize()` | 全局清理 (调用 aclFinalize) |

### 性能计时 (基于 CANN Event)

| 接口 | 说明 |
|------|------|
| `NPU_EventCreate / Destroy` | 创建 / 销毁 Event |
| `NPU_EventRecord(event, stream)` | 在流中记录时间戳 |
| `NPU_EventSynchronize(event)` | 等待 Event 完成 |
| `NPU_EventElapsedTime(&ms, start, end)` | 计算两个 Event 间耗时 (毫秒) |
| `NPU_StreamCreate / Destroy` | 创建 / 销毁 Stream |
| `NPU_DeviceSynchronize(handle)` | 设备同步 |

### Profiling (算子级耗时)

| 接口 | 说明 |
|------|------|
| `NPU_LayerProfiling_SetOutput(dir)` | 指定 prof 输出目录 (NPU_Create 之前调) |
| `NPU_LayerProfiling_Start(handle)` | 启动算子级采集 (模型加载之后调) |
| `NPU_LayerProfiling_Stop(handle)` | 停止采集, 自动跑 msprof 转 csv |
| `NPU_LayerProfiling_Parse(dir, out, max, &n)` | 解析 csv, 返回按总耗时降序的算子列表 |
| `NPU_LayerProfiling_ExportHtml(dir, html_path)` | 生成自包含 HTML 报告 (饼图+柱状图+表格) |

**Profiling 典型用法**:
```c
#include "npu_cann_adapter.h"

NPU_LayerProfiling_SetOutput("./prof_out");   // 1. 设置输出目录
NPUHandle h = NPU_Create();                    // 2. 创建实例
NPU_Model_Load(h, model_data, model_len);     // 3. 加载模型
NPU_LayerProfiling_Start(h);                   // 4. 启动采集

for (int i = 0; i < 10; i++)
    NPU_Model_Infer(h, input, input_len);      // 5. 跑 N 次推理

NPU_LayerProfiling_Stop(h);                    // 6. 停止采集 (自动转 csv)
NPU_LayerProfiling_ExportHtml("./prof_out", NULL);  // 7. 生成 HTML (NULL=自动放 PROF_xxx 下)
```

### 设备内存管理

| 接口 | 说明 |
|------|------|
| `NPU_Malloc(handle, size, &dev_ptr)` | 分配 device 内存 |
| `NPU_Free(handle, dev_ptr)` | 释放 device 内存 |
| `NPU_Memcpy(handle, dst, src, size, kind)` | 同步内存拷贝 (Host↔Device) |
| `NPU_MemcpyAsync(handle, stream, ...)` | 异步内存拷贝 |

### 运行时资源监控 (内存峰值 / 利用率)

| 接口 | 说明 |
|------|------|
| `NPU_GetDevMemory(handle, &info)` | 设备 DDR 内存快照 (free/total/used/使用率%) |
| `NPU_GetUtilization(handle, &util)` | 实时利用率: Cube / Vector / AI CPU / 内存带宽 (%) |
| `NPU_Monitor_Reset(handle)` | 重置内存峰值追踪基准 |
| `NPU_Monitor_GetPeak(handle, &peak)` | 获取运行期内存峰值 (自动采样+更新) |
| `NPU_Model_Infer_Monitored(handle, ...)` | 一站式监控推理: 前后自动采样内存+利用率打印 |

**监控典型用法**:
```c
NPUMemInfo mem;
NPU_GetDevMemory(h, &mem);          // 模型加载后查询
printf("used=%.1fMB total=%.1fMB (%.1f%%)\n",
       mem.used_bytes/1048576.0, mem.total_bytes/1048576.0, mem.used_ratio);

NPU_Monitor_Reset(h);               // 开始峰值追踪
for (...) NPU_Model_Infer(...);     // 业务推理
NPUMemPeak peak;
NPU_Monitor_GetPeak(h, &peak);      // 取峰值
printf("peak=%.1fMB cur=%.1fMB\n",
       peak.peak_used_bytes/1048576.0, peak.cur_used_bytes/1048576.0);

NPUUtilization util;
NPU_GetUtilization(h, &util);       // 实时利用率
// util.cube / util.vector / util.aicpu / util.memory (百分比, -1=不支持)
```

**实测输出** (310B1 + ViT-S):
```
[Step3.5] Monitor: dev-mem after model load: used=4165.5MB / total=11577.8MB (36.0%)
[Monitor] dev-mem peak: 4540.0 MB (cur 4540.0 MB) @ sample #2
[Monitor] utilization: cube=0% vector=0% aicpu=1% mem=-1%
```

> 注: `aclrtGetDeviceUtilizationRate` 采样的是查询时刻的瞬时值; 推理已结束后 Cube/Vector 显示 0% 属正常现象。`mem=-1` 表示 310B 不上报内存带宽利用率。若需推理期间的真实利用率曲线, 可周期性调用 (如单独监控线程 10ms 间隔)。

### 实时监控曲线 (后台双线程采样)

| 接口 | 说明 |
|------|------|
| `NPU_Monitor_Start(handle, interval_ms)` | 启动后台采样 (环形缓冲最大 `NPU_MONITOR_MAX_SAMPLES`=4096 点) |
| `NPU_Monitor_Stop(handle)` | 停止采样 |
| `NPU_Monitor_IsRunning(handle)` | 是否运行中 |
| `NPU_Monitor_GetHistory(h, buf, max, &n)` | 取采样历史 (NPUSamplePoint 数组: 时间戳/内存/Cube/Vector/AICPU) |
| `NPU_Monitor_PrintCurve(handle, width)` | 终端打印 ASCII 内存曲线 + AI-Core 利用率统计 |
| `NPU_Monitor_SavePng(h, out_dir, buf, len)` | 保存曲线 PNG: 时间戳命名 `monitor_YYYYMMDD_HHMMSS.png`, 目录不存在自动创建, Y 轴自适应放大 |

**采样架构** (310B 实测驱动耗时差异决定双线程设计):
- **内存线程**: `aclrtGetMemInfo` 单次 ~0.5ms, 按 `interval` 高频采样
- **利用率线程**: `aclrtGetDeviceUtilizationRate` 单次 ~120ms, 独立慢速采样后写缓存, 由内存采样点附带记录

**trans_demo 集成** (argv 任意位置加 `monitor[:间隔ms]`):
```bash
./trans_demo model/tiny_vit.om infer/dog.jpg 300 monitor:5
# 也可与 profiling 同时开:
./trans_demo model.om img.jpg 300 ./prof_out monitor:10
```

**输出效果**:
```
[NPU-Monitor] started: interval=20ms
...推理运行中...
[NPU-Monitor] stopped: 43 samples
========== NPU Memory Curve (n=43 samples) ==========
  ...
[NPU-Monitor] PNG saved: ./monitor/monitor_20260901_132054.png
[Monitor] 曲线图已保存: ./monitor/monitor_20260901_132054.png
[Monitor] samples=43  dev-mem max=4233.6MB  AI-Core max=53%
```

PNG 曲线图内容: 内存曲线 (蓝) + AI-Core 利用率散点 (绿) + 峰值红圈标注 (红) + 双 Y 轴 (左 MB / 右 %) + 时间轴与图例; Y 轴范围取 `[min−25%×span, max+25%×span]` 自适应放大, 数据平坦时最小视野为峰值的 2%。

> 关键点: 后台周期采样能捕获**推理运行期间**的 AI-Core 利用率 (如 53%), 而推理结束后的单次查询只能读到 0%。
> 采样密度: 内存点 ≈ 监控时长/interval (interval=20ms 时约 50 点/s); 利用率点受驱动单次 ~120ms 限制, 约 7 点/s, 密度低于内存点属正常。
> 线程注意: ACL context 按线程绑定, 两个采样线程入口都必须先 `aclrtSetDevice(0)` (lib 内部已处理)。
> 无代码替代方案: 秒级粗粒度可用板载工具 `npu-smi info watch -i 0 -d 1 -s m > mem_log.txt` (内存占用导出)。

详细定义参见 [npu_cann_adapter.h](lib/npu_cann_adapter.h)。

## 模型转换 (ONNX → OM)

使用 `model/check_onnx_shpae.py` 检查形状并生成 ATC 命令:

```bash
python3 model/check_onnx_shpae.py model/yolov5s.onnx
# 输出 ATC 参考命令
atc --model=model/yolov5s.onnx \
    --framework=5 \
    --output=model/model_out \
    --soc_version=Ascend310B1 \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --precision_mode=force_fp16
```

## 推理模式

| mode | 含义 |
|------|------|
| 0 (AUTO) | 自动: `.om` → NPU, `.onnx` → 自动 ATC 转换 |
| 1 (NPU) | 强制 NPU 推理 (.om) |
| 2 (ONNX) | 强制 ONNX Runtime CPU 推理 (.onnx) |

## CMake 设计说明

顶层 CMake 统一管理依赖查找, 通过 `target_include_directories` / `target_link_libraries` 的 **PUBLIC** 传播机制, 自动将 CANN / ONNX Runtime / OpenCV 的头文件路径、编译宏、链接库传递给所有依赖 `npu_driver` 的 demo。

关键收益:
- **零硬编码**: 各 demo 不再硬编码 `/root/workspace/lib/build/libnpu_driver.a` 绝对路径
- **自动依赖传播**: `target_link_libraries(xxx npu_driver)` 一行搞定所有依赖
- **独立构建兼容**: 各 demo 目录仍可单独 cmake .. (已移除旧的 find_path / include_directories)
- **可开关**: `-DUSE_ONNXRUNTIME=OFF` / `-DUSE_OPENCV=OFF` 一键裁剪
