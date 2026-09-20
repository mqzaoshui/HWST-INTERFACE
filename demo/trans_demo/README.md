# trans_demo — TinyViT (Vision Transformer) 昇腾 NPU 推理示例

基于自封装库 `libnpu_driver`（ACL / ONNX Runtime 双后端）的 Transformer 分类端到端示例。

模型为 TinyViT：6 层、dim=192、heads=3、patch=16，输入 `1x3x224x224`，输出 1000 类。
权重为随机初始化（由 `gen_vit_onnx.py` 生成），例程目的是验证 Transformer 算子
（MatMul / Softmax / LayerNorm / GELU）在 310B NPU 上的**算子兼容性与性能**，
因此 Top-5 类别无语义（权重非真实训练所得）。

## 目录结构

```
trans_demo/
├── CMakeLists.txt      # 支持顶层构建 / 独立构建
├── trans_demo.cpp      # 主程序
├── gen_vit_onnx.py     # 生成 tiny_vit.onnx（随机权重）
├── classname.txt       # ImageNet 1000 类标签（可选）
├── model/
│   ├── tiny_vit.om     # ATC 转换后的 NPU 模型
│   └── tiny_vit.onnx   # ONNX 原始模型（可自动转 om）
├── infer/
│   └── dog.jpg         # 测试图片
├── build/              # 独立构建产物
└── prof_out/           # profiling 输出目录（运行时生成）
```

## 编译

```bash
# 前置: source CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**方式一：顶层构建**（推荐，在 `workspace/build/` 下统一编译全部目标）

```bash
cmake -S /root/workspace -B /root/workspace/build
cmake --build /root/workspace/build -j4
# 产物: /root/workspace/build/trans_demo
```

**方式二：独立构建**（仅编译本 demo，需已存在 `libnpu_driver.a`）

```bash
cd /root/workspace/demo/trans_demo
cmake -S . -B build && cmake --build build -j4
# 产物: build/trans_demo
```

> 依赖：CANN 8.0（Ascend 310B）、OpenCV 4（必需，图像预处理）、libnpu_driver.a。

## 运行

```bash
cd /root/workspace/demo/trans_demo      # 相对路径 ./model ./infer 依赖此 cwd
/root/workspace/build/trans_demo model/tiny_vit.om infer/dog.jpg [loop] [labels|prof_dir|monitor[:ms]] ...
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `argv[1]` | 模型路径（`.om` 直接推理；`.onnx` 自动 ATC 转 om） | `./model/tiny_vit.om` |
| `argv[2]` | 输入图片 | `./infer/dog.jpg` |
| `argv[3]` | benchmark 循环次数 | `100` |
| `argv[4..6]` | 可选，按关键字识别（顺序任意）：`.txt` 结尾→标签文件；`monitor` 开头→开启资源监控；其他→profiling 输出目录 | 无 |

### 典型用法

```bash
# 基础 benchmark
./trans_demo model/tiny_vit.om infer/dog.jpg 100

# 开启实时资源监控曲线, 采样间隔 20ms（循环次数应足够大, 否则采样点过少）
./trans_demo model/tiny_vit.om infer/dog.jpg 100 monitor:20

# 开启算子级 profiling（输出到 ./prof_out）
./trans_demo model/tiny_vit.om infer/dog.jpg 100 ./prof_out

# profiling + 监控同时开启
./trans_demo model/tiny_vit.om infer/dog.jpg 100 ./prof_out monitor:10

# 指定标签文件
./trans_demo model/tiny_vit.om infer/dog.jpg 100 classname.txt
```

## 执行流程与功能

1. **Step1 预处理**：`imread` → resize 224 → RGB → `/255` → `(x-mean)/std` → NCHW float32
2. **Step2~3 创建句柄 / 加载模型**：`NPU_DRIVER_FORMAT_AUTO`，打印模型 IO 信息
3. **Step3.5 资源监控初始化**：打印模型加载后的设备 DDR 占用，重置峰值基准；若 `monitor` 开启则启动后台采样线程
4. **Step4 Warmup x5**：消除首次推理的初始化开销
5. **Step5 Benchmark**：Event 设备计时 + Host chrono 计时，输出 avg/min/max/p50/p90/p99/FPS
6. **Step5.5 监控结果**（`monitor` 开启时）：
   - 终端打印 ASCII 内存曲线 + AI-Core 利用率统计
   - 保存 PNG 曲线图到当前目录 `./monitor/`，文件名带时间戳：`monitor_YYYYMMDD_HHMMSS.png`（目录不存在会自动创建）
   - 打印采样点数、设备内存峰值、瞬时利用率
7. **Step5.1 算子级 profiling**（指定 `prof_dir` 时）：Stop 后解析 CANN csv，打印 Top-20 算子耗时表，并在 `prof_dir/PROF_xxx/` 下生成完整 csv 与自包含 HTML 可视化报告
8. **Step6 后处理**：softmax → Top-5 分类结果
9. **Step7 清理**：卸载模型、销毁句柄、finalize ACL

## 输出示例（310B1 实测）

```
[Device (Event)  ] n=100  avg= 8.04ms  p50= 7.96  p90= 8.34  FPS= 124.44
[Host   (chrono) ] n=100  avg= 8.00ms  p50= 7.92  p90= 8.31  FPS= 125.04
[NPU-Monitor] stopped: 7 samples
[NPU-Monitor] PNG saved: ./monitor/monitor_20260901_130224.png
[Monitor] samples=7  dev-mem max=4246.4MB  AI-Core max=53%
[Monitor] dev-mem peak: 4247.7 MB (cur 4247.7 MB) @ sample #8

========== Top-5 Classification ==========
  #1: class_id=196   prob= 95.79%  miniature schnauzer
  #2: class_id=198   prob=  3.63%  standard schnauzer
  ...
```

## 注意事项

- 必须先 `source set_env.sh`，否则 ACL 动态库找不到
- 需用 `-DUSE_OPENCV` 编译链（顶层 CMake 默认开启），未开启时直接报错退出
- `monitor` 的采样点数 ≈ 监控时长 / 采样间隔；短循环（如 `loop=5`，约 40ms）配 50ms 间隔只会采到 1 个点，验证曲线时建议 `loop>=100` 且 `monitor:20`
- 内存与利用率由两个线程分别采样：内存按 `interval` 高频采样（单次查询 ~0.5ms）；利用率受驱动限制单次查询约 120ms，曲线上的利用率点密度低于内存点属正常现象
- profiling 目录若指定，会在其下自动生成 `PROF_xxx` 子目录存放 csv / 日志 / HTML 报告
- `aclrtGetDeviceUtilizationRate` 为瞬时值，推理结束后查询到的 Cube/Vector 为 0% 属正常；真实曲线依赖后台线程周期采样（即 `monitor` 功能）
- CANN profiling（`PROF_xxx`）只含算子/API/耗时统计（op_summary 中仅 `memory_bound` 标记访存瓶颈），**没有设备内存占用的时间序列**；如需脱离代码实时监控 NPU 内存，可用板载工具：`npu-smi info watch -i 0 -d 1 -s m > mem_log.txt`（1 秒粒度，stdout 可直接导出）
