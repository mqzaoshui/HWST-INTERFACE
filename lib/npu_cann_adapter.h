/**
 * NPU CANN SDK 驱动接口
 * 
 * 基于华为昇腾CANN SDK，提供NPU模型推理的统一接口
 * 
 * 支持的模型格式：
 * - .om 格式：直接在NPU上加载和推理（高性能）
 * - .onnx 格式：
 *     a) 通过ONNX Runtime在CPU上直接推理（需安装onnxruntime）
 *     b) 自动调用ATC工具转换为.om后在NPU上推理
 * 
 * 使用前需要：
 * 1. source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
 * 2. 如需ONNX直接推理，需安装onnxruntime库
 * 3. 如需ONNX自动转换，需确保atc工具可用
 */

#ifndef NPU_CANN_DRIVER_H
#define NPU_CANN_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*============================================================================
 * 常量定义
 *============================================================================*/

/* 模型格式选项 */
#define NPU_DRIVER_FORMAT_AUTO    0   /* 自动检测：.om→NPU推理，.onnx→自动转换为.om */
#define NPU_DRIVER_FORMAT_NPU     1   /* 强制NPU推理（.om格式，需昇腾设备） */
#define NPU_DRIVER_FORMAT_ONNX    2   /* 强制ONNX Runtime推理（.onnx格式，使用CPU） */

/*============================================================================
 * 数据结构
 *============================================================================*/

/* 模型形状信息 */
typedef struct {
    int dims[8];         /* 各维度大小 */
    int num_dims;        /* 维度数量 */
    int data_type;       /* 0: float32, 1: int32, 2: int8, 3: uint8 */
} NpuShapeInfo;

/* 模型状态信息 */
typedef struct {
    char model_path[256];     /* 当前加载的模型路径 */
    uint32_t model_format;    /* 当前模型格式：NPU_DRIVER_FORMAT_xxx */
    uint32_t backend_type;    /* 后端类型：0=NPU_CANN, 1=ONNX Runtime */
    uint32_t npu_id;          /* 使用的NPU设备ID */
    uint32_t input_size;      /* 输入数据大小（字节）*/
    uint32_t output_size;     /* 输出数据大小（字节）*/
    uint32_t loaded;          /* 是否已加载：0=未加载, 1=已加载 */
    NpuShapeInfo input_shape; /* 输入形状 */
    NpuShapeInfo output_shape;/* 输出形状 */
    uint32_t reserved[4];
} NPUModelStatus;

/* NPU句柄（不透明指针） */
typedef void* NPUHandle;

/* Stream / Event 不透明句柄 */
typedef void* NPUStream;
typedef void* NPUEvent;

/* 内存拷贝方向 */
typedef enum {
    NPU_MEMCPY_HOST_TO_DEVICE = 0,
    NPU_MEMCPY_DEVICE_TO_HOST = 1,
    NPU_MEMCPY_DEVICE_TO_DEVICE = 2,
    NPU_MEMCPY_HOST_TO_HOST = 3
} NPU_CopyDir;

/* 模型加载参数 */
typedef struct {
    uint32_t npu_count;       /* 使用的NPU数量 */
    uint32_t npu_ids[8];      /* NPU设备ID列表 */
    char model_path[256];     /* 模型路径（支持.onnx和.om） */
    uint32_t model_format;    /* 模型格式选项：NPU_MODEL_FORMAT_xxx */
    uint32_t reserved[3];
} NPUModelLoadMsg;

/*============================================================================
 * 接口函数
 *============================================================================*/

/**
 * @brief 获取当前开发板NPU数量
 * @return NPU数量
 */
unsigned int NPU_GetNpuCount(void);

/**
 * @brief 获取模型输入输出形状信息
 * @param data 模型数据（预留，可传NULL）
 * @param len 数据长度（预留）
 * @param input_shape 输出：输入形状信息
 * @param output_shape 输出：输出形状信息
 * @return 0成功，-1失败
 */
int NPU_GetModelShape(const unsigned char* data, unsigned int len,
                      NpuShapeInfo* input_shape, NpuShapeInfo* output_shape);

/**
 * @brief 获取当前模型状态（用于热更新验证）
 * @param handle NPU句柄
 * @param status 输出：模型状态信息
 * @return 0成功，-1失败
 */
int NPU_GetModelStatus(NPUHandle handle, NPUModelStatus* status);

/**
 * @brief 创建NPU实例
 * @return NPU句柄，失败返回NULL
 */
NPUHandle NPU_Create(void);

/**
 * @brief 销毁NPU实例
 * @param handle NPU句柄
 */
void NPU_Destroy(NPUHandle handle);

/**
 * @brief 加载模型
 * @param handle NPU句柄
 * @param pMsg 模型加载参数（NPUModelLoadMsg结构）
 * @param iLen 参数长度
 * @return 0成功，-1失败
 * @note 通过model_format字段选择加载方式：
 *       - NPU_DRIVER_FORMAT_AUTO (0): 自动检测（默认）
 *       - NPU_DRIVER_FORMAT_NPU (1): NPU推理(.om)
 *       - NPU_DRIVER_FORMAT_ONNX (2): ONNX Runtime推理(.onnx，CPU)
 */
int NPU_Model_Load(NPUHandle handle, void* pMsg, uint32_t iLen);

/**
 * @brief 卸载模型
 * @param handle NPU句柄
 * @return 0成功，-1失败
 */
int NPU_Model_Unload(NPUHandle handle);

/**
 * @brief 执行模型推理
 * @param handle NPU句柄
 * @param input_data 输入数据
 * @param input_len 输入数据长度
 * @param output_data 输出数据缓冲区
 * @param output_len [in]缓冲区大小 [out]实际输出长度
 * @return 0成功，-1失败
 */
int NPU_Model_Infer(NPUHandle handle, const unsigned char* input_data,
                    unsigned int input_len, unsigned char* output_data,
                    unsigned int* output_len);

/**
 * @brief 更新模型（热更新）
 * @param handle NPU句柄
 * @param data 新模型数据
 * @param len 数据长度
 * @return 0成功，-1失败
 * @note CANN不支持热更新，需卸载后重新加载
 */
int NPU_Model_Update(NPUHandle handle, const unsigned char* data,
                     unsigned int len);

/**
 * @brief 获取最近一次推理结果
 * @param handle NPU句柄
 * @param result_data 结果数据缓冲区
 * @param result_len [in]缓冲区大小 [out]实际结果长度
 * @return 0成功，-1失败
 */
int NPU_Get_Infer_Result(NPUHandle handle, unsigned char* result_data,
                         unsigned int* result_len);

/**
 * @brief 获取最近错误信息
 * @param handle NPU句柄（可传NULL）
 * @return 错误信息字符串，无错误返回NULL
 */
const char* NPU_GetLastError(NPUHandle handle);

/**
 * @brief 清理全局资源（程序退出前调用）
 */
void NPU_Finalize(void);

/*============================================================================
 * 第一批扩展接口 —— 对标 CUDA Runtime
 *============================================================================*/

/**
 * @brief 在指定 NPU 设备上分配 Device 内存
 * @param handle NPU 句柄
 * @param size 分配字节数
 * @param dev_ptr [out] 设备指针
 * @return 0 成功,-1 失败
 */
int NPU_Malloc(NPUHandle handle, size_t size, void** dev_ptr);

/**
 * @brief 释放 Device 内存
 * @param handle NPU 句柄
 * @param dev_ptr 设备指针
 * @return 0 成功,-1 失败
 */
int NPU_Free(NPUHandle handle, void* dev_ptr);

/**
 * @brief 同步内存拷贝（立即完成）
 * @param handle NPU 句柄
 * @param dst 目的地址
 * @param src 源地址
 * @param size 拷贝字节数
 * @param dir 拷贝方向
 * @return 0 成功,-1 失败
 */
int NPU_Memcpy(NPUHandle handle, void* dst, const void* src,
               size_t size, NPU_CopyDir dir);

/**
 * @brief 异步内存拷贝（投递到指定 stream）
 * @param handle NPU 句柄
 * @param stream 目标 stream,传 NULL 用 handle 默认 stream
 * @param dst 目的地址
 * @param src 源地址
 * @param size 拷贝字节数
 * @param dir 拷贝方向
 * @return 0 成功,-1 失败
 */
int NPU_MemcpyAsync(NPUHandle handle, NPUStream stream,
                    void* dst, const void* src,
                    size_t size, NPU_CopyDir dir);

/**
 * @brief 在 handle 绑定的设备上创建新 stream
 * @param handle NPU 句柄
 * @param stream [out] 新 stream 句柄
 * @return 0 成功,-1 失败
 */
int NPU_StreamCreate(NPUHandle handle, NPUStream* stream);

/**
 * @brief 销毁 stream
 * @param handle NPU 句柄
 * @param stream 要销毁的 stream
 * @return 0 成功,-1 失败
 */
int NPU_StreamDestroy(NPUHandle handle, NPUStream stream);

/**
 * @brief 同步等待某个 stream 上所有任务完成
 * @param handle NPU 句柄
 * @param stream 要同步的 stream,NULL 表示 handle 默认 stream
 * @return 0 成功,-1 失败
 */
int NPU_StreamSynchronize(NPUHandle handle, NPUStream stream);

/**
 * @brief 同步等待整个设备上所有任务完成
 * @param handle NPU 句柄
 * @return 0 成功,-1 失败
 */
int NPU_DeviceSynchronize(NPUHandle handle);

/**
 * @brief 在 handle 绑定的设备上创建 event
 * @param handle NPU 句柄
 * @param event [out] 新 event 句柄
 * @return 0 成功,-1 失败
 */
int NPU_EventCreate(NPUHandle handle, NPUEvent* event);

/**
 * @brief 销毁 event
 * @param handle NPU 句柄
 * @param event 要销毁的 event
 * @return 0 成功,-1 失败
 */
int NPU_EventDestroy(NPUHandle handle, NPUEvent event);

/**
 * @brief 在指定 stream 上记录 event
 * @param handle NPU 句柄
 * @param event 要记录的 event
 * @param stream 目标 stream,NULL 表示 handle 默认 stream
 * @return 0 成功,-1 失败
 */
int NPU_EventRecord(NPUHandle handle, NPUEvent event, NPUStream stream);

/**
 * @brief 阻塞等待 event 完成
 * @param handle NPU 句柄
 * @param event 要等待的 event
 * @return 0 成功,-1 失败
 */
int NPU_EventSynchronize(NPUHandle handle, NPUEvent event);

/**
 * @brief 计算两个 event 之间已完成任务的耗时(ms)
 * @param handle NPU 句柄
 * @param ms [out] 耗时,单位毫秒
 * @param start 起始 event
 * @param end 结束 event
 * @return 0 成功,-1 失败
 */
int NPU_EventElapsedTime(NPUHandle handle, float* ms,
                         NPUEvent start, NPUEvent end);

/**
 * @brief 异步推理（投递到 stream,返回后需自行同步确认完成）
 * @param handle NPU 句柄
 * @param stream 目标 stream,NULL 表示 handle 默认 stream
 * @param input_data 输入数据
 * @param input_len 输入数据长度
 * @param output_data 输出缓冲
 * @param output_len [in]缓冲大小 [out]实际输出长度
 * @return 0 成功(已投递),-1 失败
 */
int NPU_Model_InferAsync(NPUHandle handle, NPUStream stream,
                         const unsigned char* input_data, unsigned int input_len,
                         unsigned char* output_data, unsigned int* output_len);

/*============================================================================
 * 第二批扩展接口 —— OM 模型每层算子推理耗时统计 (ACL Profiling)
 *============================================================================*/

/* 单算子耗时记录 */
typedef struct {
    char op_type[64];     /* 算子类型, 如 "MatMul" / "LayerNorm" / "Softmax" */
    char op_name[160];    /* 算子完整名 */
    uint64_t duration_us; /* 单次耗时(us) */
    uint32_t task_count; /* 调用次数 */
    uint64_t total_us;   /* 累计耗时(us) */
    uint32_t reserved;
} NPULayerInfo;

/**
 * @brief 设置算子级 Profiling 输出目录
 * @note 必须在 NPU_Create 之前调用; 后续 NPU_Create 内部会在 aclInit 之前
 *       自动调用 aclprofInit, 推理结束后 CANN 会在此目录生成 csv
 * @param out_dir 输出目录(将自动创建), 传 NULL 关闭 profiling
 * @return 0 成功, -1 失败
 */
int NPU_LayerProfiling_SetOutput(const char* out_dir);

/**
 * @brief 启动算子耗时采集, 必须在 NPU_Model_Load 之后调用
 * @param handle NPU 句柄
 * @return 0 成功, -1 失败
 */
int NPU_LayerProfiling_Start(NPUHandle handle);

/**
 * @brief 停止采集, 触发 CANN 将算子级耗时写入输出目录
 * @param handle NPU 句柄
 * @return 0 成功, -1 失败
 */
int NPU_LayerProfiling_Stop(NPUHandle handle);

/**
 * @brief 解析 profiling 输出目录, 返回按累计耗时降序排列的算子列表
 * @param out_dir NPU_LayerProfiling_SetOutput 指定的目录
 * @param out 输出数组(由调用方分配)
 * @param max_count 数组容量
 * @param actual_count [out] 实际写入条数
 * @return 0 成功(部分解析), -1 失败
 */
int NPU_LayerProfiling_Parse(const char* out_dir, NPULayerInfo* out,
                             int max_count, int* actual_count);

/**
 * @brief 生成自包含 HTML 可视化报告(饼图+柱状图+表格)
 * @note 调用前必须先 NPU_LayerProfiling_Stop(已自动跑 msprof 导出 csv)
 * @param out_dir NPU_LayerProfiling_SetOutput 指定的目录
 * @param html_path 生成的 html 文件路径
 * @return 0 成功, -1 失败
 */
int NPU_LayerProfiling_ExportHtml(const char* out_dir, const char* html_path);

/*============================================================================
 * 第三批扩展接口 —— 运行时资源监控 (内存 / 利用率)
 *============================================================================*/

/* 设备内存快照 (字节) */
typedef struct {
    size_t free_bytes;    /* DDR 可用内存 */
    size_t total_bytes;   /* DDR 总内存 */
    size_t used_bytes;    /* 已用 = total - free */
    float  used_ratio;    /* 使用率 0~100 (%) */
} NPUMemInfo;

/* 设备利用率快照 (百分比 0~100, -1 表示该指标不支持) */
typedef struct {
    int32_t cube;         /* AI Cube 利用率 (矩阵运算, GEMM/Conv) */
    int32_t vector;       /* AI Vector 利用率 (向量运算) */
    int32_t aicpu;        /* AI CPU 利用率 */
    int32_t memory;       /* 内存带宽利用率 */
    uint64_t timestamp_ms;/* 采样时刻 ( epoch 毫秒) */
} NPUUtilization;

/* 运行时内存峰值追踪 (进程级, 从 NPU_Monitor_Reset 开始累计) */
typedef struct {
    size_t peak_used_bytes;      /* 历史最高已用内存 */
    size_t cur_used_bytes;       /* 当前已用内存 */
    uint64_t peak_timestamp_ms;  /* 峰值出现时刻 */
    uint64_t sample_count;       /* 采样次数 */
} NPUMemPeak;

/**
 * @brief 获取设备 DDR 内存快照
 * @param handle NPU 句柄 (预留, 当前传 NULL 亦可)
 * @param info [out] 内存快照
 * @return 0 成功, -1 失败
 */
int NPU_GetDevMemory(NPUHandle handle, NPUMemInfo* info);

/**
 * @brief 获取设备实时利用率 (Cube/Vector/AICPU/Memory)
 * @note 部分型号 (如 310B) 可能不支持 cube/vector 指标, 不支持时对应字段为 -1
 * @param handle NPU 句柄 (预留, 当前传 NULL 亦可)
 * @param util [out] 利用率快照
 * @return 0 成功, -1 失败
 */
int NPU_GetUtilization(NPUHandle handle, NPUUtilization* util);

/**
 * @brief 重置内存峰值追踪基准 (把当前用量作为新的峰值起点)
 * @param handle NPU 句柄
 * @return 0 成功, -1 失败
 */
int NPU_Monitor_Reset(NPUHandle handle);

/**
 * @brief 获取内存峰值追踪结果 (自动采样: 每次调用先刷新当前值再对比峰值)
 * @param handle NPU 句柄
 * @param peak [out] 峰值信息
 * @return 0 成功, -1 失败
 */
int NPU_Monitor_GetPeak(NPUHandle handle, NPUMemPeak* peak);

/**
 * @brief 一站式推理资源监控: 单次推理前后自动采样内存+利用率
 * @note 内部执行: 采样->NPU_Model_Infer->采样->更新峰值, 适合周期性调用
 * @param handle NPU 句柄
 * @param input_data 输入数据
 * @param input_len 输入长度
 * @param output_data 输出缓冲
 * @param output_len [in]缓冲大小 [out]实际输出长度
 * @param mem_before [out] 推理前内存 (可 NULL)
 * @param mem_after [out] 推理后内存 (可 NULL)
 * @return 0 成功, -1 失败
 */
int NPU_Model_Infer_Monitored(NPUHandle handle,
                              const unsigned char* input_data, unsigned int input_len,
                              unsigned char* output_data, unsigned int* output_len,
                              NPUMemInfo* mem_before, NPUMemInfo* mem_after);

/*============================================================================
 * 实时监控曲线 —— 后台线程周期采样 (内存 + 利用率)
 *   Start -> [业务推理运行中, 后台自动采样] -> Stop -> GetHistory/PrintCurve
 *============================================================================*/

/* 单个采样点 */
typedef struct {
    uint64_t timestamp_ms;   /* epoch 毫秒 */
    size_t   used_bytes;     /* 已用 DDR 内存 */
    size_t   total_bytes;    /* 总 DDR 内存 */
    int32_t  cube;           /* Cube 利用率 % (-1=不支持) */
    int32_t  vector;         /* Vector 利用率 % */
    int32_t  aicpu;          /* AI CPU 利用率 % */
} NPUSamplePoint;

#define NPU_MONITOR_MAX_SAMPLES 4096

/**
 * @brief 启动后台监控线程, 周期采样内存+利用率
 * @param handle NPU 句柄
 * @param interval_ms 采样间隔 (毫秒, 建议 5~50)
 * @return 0 成功, -1 失败 (重复启动返回 -1)
 */
int NPU_Monitor_Start(NPUHandle handle, uint32_t interval_ms);

/**
 * @brief 停止后台监控线程
 * @return 0 成功, -1 失败
 */
int NPU_Monitor_Stop(NPUHandle handle);

/**
 * @brief 是否正在监控
 * @return 1 运行中, 0 未运行
 */
int NPU_Monitor_IsRunning(NPUHandle handle);

/**
 * @brief 获取采样历史 (拷贝到调用者缓冲)
 * @param samples [out] 输出数组
 * @param max_samples 缓冲容量
 * @param out_n [out] 实际采样点数
 * @return 0 成功, -1 失败
 */
int NPU_Monitor_GetHistory(NPUHandle handle,
                           NPUSamplePoint* samples, uint32_t max_samples,
                           uint32_t* out_n);

/**
 * @brief 向终端打印 ASCII 内存监控曲线 (时间轴 + MB), 内含峰值标记
 * @param width 曲线宽度 (字符数, 默认 60 可传 0)
 * @return 0 成功, -1 无数据
 */
int NPU_Monitor_PrintCurve(NPUHandle handle, uint32_t width);

/**
 * @brief 将采样历史绘制为 PNG 曲线图 (需 USE_OPENCV 编译)
 * @param out_dir 输出目录 (不存在自动创建); NULL 则默认 "./monitor"
 * @param path_buf [out] 回写生成的完整文件路径 (可 NULL 不关心)
 * @param buf_len path_buf 容量
 * @return 0 成功; -1 无数据/未编译 OpenCV
 * @note 文件名: monitor_YYYYMMDD_HHMMSS.png, 内容含内存曲线+AI-Core利用率
 */
int NPU_Monitor_SavePng(NPUHandle handle, const char* out_dir,
                        char* path_buf, uint32_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* NPU_CANN_DRIVER_H */