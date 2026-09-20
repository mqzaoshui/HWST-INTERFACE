/**
 * NPU CANN SDK 驱动实现
 * 
 * 支持两种推理后端：
 * 1. CANN NPU后端：使用昇腾NPU推理（高性能）
 * 2. ONNX Runtime CPU后端：直接加载ONNX模型推理（便捷）
 * 
 * 通过 NPUModelLoadMsg.model_format 字段选择后端
 */

#include "npu_cann_adapter.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <string>
#include <fstream>
#include <memory>
#include <algorithm>

/* CANN SDK 头文件 */
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl/acl_prof.h"
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <pthread.h>
#include <cerrno>
#include <cmath>
#include <unordered_map>
#include <sstream>
#include <dirent.h>
#include <unistd.h>

/* ONNX Runtime 头文件（可选） */
#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

/* OpenCV 头文件（可选, 用于监控曲线 PNG 渲染） */
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

/*============================================================================
 * 内部状态
 *============================================================================*/

static thread_local char g_last_error[256] = {0};
static bool g_acl_initialized = false;
static bool g_acl_finalized = false;
static std::mutex g_init_mutex;

/* ===== 算子级 Profiling 全局状态 ===== */
static char g_prof_out_dir[512] = {0};      /* SetOutput 设置的目录 */
static bool g_prof_inited = false;          /* aclprofInit 是否已调用 */

/* 前向声明 */
static void set_error(const char* msg);
static void clear_error();
static int init_acl();

/*============================================================================
 * NPU实现类 - CANN NPU后端
 *============================================================================*/

class NPUImpl {
public:
    bool loaded;
    uint32_t model_id;
    aclmdlDesc* model_desc;
    aclmdlDataset* input_dataset;
    aclmdlDataset* output_dataset;
    int32_t device_id;
    aclrtContext context;
    
    /* stream 池,index 0 为默认 stream(Init 时创建),用户可动态追加 */
    std::vector<aclrtStream> streams;
    
    std::string model_path;
    size_t input_size;
    size_t output_size;
    void* input_buffer;
    void* output_buffer;
    
    NpuShapeInfo input_shape;
    NpuShapeInfo output_shape;
    
    std::vector<unsigned char> last_result;
    std::mutex mutex_;
    
    NPUImpl() 
        : loaded(false), model_id(0), model_desc(nullptr),
          input_dataset(nullptr), output_dataset(nullptr),
          device_id(0), context(nullptr),
          input_size(0), output_size(0),
          input_buffer(nullptr), output_buffer(nullptr) {
        memset(&input_shape, 0, sizeof(input_shape));
        memset(&output_shape, 0, sizeof(output_shape));
    }
    
    ~NPUImpl() {
        Cleanup();
    }
    
    aclrtStream DefaultStream() { return streams.empty() ? nullptr : streams[0]; }
    
    /* 在任何 ACL 操作前调用,确保当前线程绑定正确 context */
    int EnsureContext() {
        if (!context) {
            set_error("NPU 设备未初始化");
            return -1;
        }
        aclError ret = aclrtSetDevice(device_id);
        if (ret != ACL_SUCCESS) {
            set_error("aclrtSetDevice failed");
            return -1;
        }
        ret = aclrtSetCurrentContext(context);
        if (ret != ACL_SUCCESS) {
            set_error("aclrtSetCurrentContext failed");
            return -1;
        }
        return 0;
    }
    
    /* 把对外的 NPUStream 句柄转成内部 aclrtStream */
    aclrtStream ResolveStream(NPUStream s) {
        return static_cast<aclrtStream>(s);
    }
    
    int Init(int32_t dev_id) {
        device_id = dev_id;
        
        /* 初始化ACL（延迟初始化，仅在使用NPU时调用） */
        if (init_acl() != 0) {
            return -1;
        }
        
        /* 设置设备 */
        aclError ret = aclrtSetDevice(device_id);
        if (ret != ACL_SUCCESS) {
            set_error("aclrtSetDevice failed");
            return -1;
        }
        
        /* 创建上下文 */
        ret = aclrtCreateContext(&context, device_id);
        if (ret != ACL_SUCCESS || !context) {
            set_error("aclrtCreateContext failed");
            return -1;
        }
        
        /* 设置当前上下文 */
        ret = aclrtSetCurrentContext(context);
        if (ret != ACL_SUCCESS) {
            set_error("aclrtSetCurrentContext failed");
            return -1;
        }
        
        /* 创建默认 stream */
        aclrtStream s = nullptr;
        ret = aclrtCreateStream(&s);
        if (ret != ACL_SUCCESS || !s) {
            set_error("aclrtCreateStream failed");
            return -1;
        }
        streams.push_back(s);
        
        return 0;
    }
    
    void Cleanup() {
        if (model_id != 0) {
            aclmdlUnload(model_id);
            model_id = 0;
        }
        if (model_desc) {
            aclmdlDestroyDesc(model_desc);
            model_desc = nullptr;
        }
        if (input_dataset) {
            aclmdlDestroyDataset(input_dataset);
            input_dataset = nullptr;
        }
        if (output_dataset) {
            aclmdlDestroyDataset(output_dataset);
            output_dataset = nullptr;
        }
        if (input_buffer) {
            aclrtFree(input_buffer);
            input_buffer = nullptr;
        }
        if (output_buffer) {
            aclrtFree(output_buffer);
            output_buffer = nullptr;
        }
        /* 销毁所有 stream,包括默认 stream(index 0) */
        for (auto s : streams) {
            if (s) aclrtDestroyStream(s);
        }
        streams.clear();
        if (context) {
            aclrtDestroyContext(context);
            context = nullptr;
        }
        aclrtResetDevice(device_id);
    }
    
    int LoadModel(const char* model_path, bool convert_onnx) {
        std::string path_str(model_path);
        this->model_path = path_str;  /* 保存模型路径 */
        std::string om_path;
        
        /* 检测模型格式 */
        bool need_convert = false;
        if (path_str.size() >= 5 && path_str.substr(path_str.size() - 5) == ".onnx") {
            if (convert_onnx) {
                printf("[NPU] 检测到ONNX模型，转换为OM格式...\n");
                need_convert = true;
                
                size_t pos = path_str.rfind(".onnx");
                om_path = path_str.substr(0, pos) + ".om";
                
                /* 检查OM文件是否已存在 */
                std::ifstream om_check(om_path);
                if (om_check.good()) {
                    printf("[NPU] OM文件已存在: %s，跳过转换\n", om_path.c_str());
                    need_convert = false;
                }
            } else {
                set_error("选择NPU推理模式但模型为ONNX格式，请使用OM格式或设置model_format=NPU_DRIVER_FORMAT_AUTO");
                return -1;
            }
        } else if (path_str.size() >= 3 && path_str.substr(path_str.size() - 3) == ".om") {
            om_path = path_str;
        } else {
            om_path = path_str;
        }
        
        /* 执行ONNX→OM转换 */
        if (need_convert) {
            printf("[NPU] 正在转换模型 (TE_PARALLEL_COMPILER=1 防OOM)...\n");
            /* 先 source CANN 环境, 再跑 atc, 限制并行编译数防内存爆 */
            char cmd[1024];
            snprintf(cmd, sizeof(cmd),
                     "bash -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh "
                     "&& export TE_PARALLEL_COMPILER=1 "
                     "&& atc --model=%s --framework=5 --output=%s "
                     "--soc_version=Ascend310B1 --precision_mode=force_fp16 2>&1'",
                     model_path, om_path.c_str());
            
            printf("[NPU] 命令: TE_PARALLEL_COMPILER=1 atc --model=%s --output=%s\n",
                   model_path, om_path.c_str());
            int ret = system(cmd);
            
            if (ret != 0) {
                set_error("ATC转换失败，请检查环境变量(cann_env)、Python依赖、内存"
                          " (若已手写 .om, 可放回同目录自动跳过)");
                printf("[ERROR] ATC转换失败 (退出码: %d)\n", ret);
                return -1;
            }
            
            std::ifstream om_check(om_path);
            if (!om_check.good()) {
                set_error("ATC转换成功但未找到.om文件");
                return -1;
            }
            om_check.close();
            printf("[NPU] ONNX→OM转换成功: %s\n", om_path.c_str());
        }
        
        /* 加载模型（旧版API：通过指针返回model_id） */
        aclError ret = aclmdlLoadFromFile(om_path.c_str(), &model_id);
        if (ret != ACL_SUCCESS || model_id == 0) {
            set_error("aclmdlLoadFromFile失败");
            return -1;
        }
        
        /* 创建并获取模型描述 */
        model_desc = aclmdlCreateDesc();
        if (!model_desc) {
            set_error("aclmdlCreateDesc失败");
            return -1;
        }
        
        ret = aclmdlGetDesc(model_desc, model_id);
        if (ret != ACL_SUCCESS) {
            set_error("aclmdlGetDesc失败");
            return -1;
        }
        
        /* 获取输入输出信息 */
        size_t input_count = aclmdlGetNumInputs(model_desc);
        size_t output_count = aclmdlGetNumOutputs(model_desc);
        printf("[NPU] 模型有 %zu 个输入，%zu 个输出\n", input_count, output_count);
        
        if (input_count > 0) {
            aclmdlIODims input_dims;
            ret = aclmdlGetInputDims(model_desc, 0, &input_dims);
            if (ret == ACL_SUCCESS) {
                input_shape.num_dims = (int)input_dims.dimCount;
                for (size_t i = 0; i < input_dims.dimCount && i < 8; ++i) {
                    input_shape.dims[i] = (int)input_dims.dims[i];
                }
            }
            input_size = aclmdlGetInputSizeByIndex(model_desc, 0);
            printf("[NPU] 输入尺寸: %zu bytes\n", input_size);
        }
        
        if (output_count > 0) {
            aclmdlIODims output_dims;
            ret = aclmdlGetOutputDims(model_desc, 0, &output_dims);
            if (ret == ACL_SUCCESS) {
                output_shape.num_dims = (int)output_dims.dimCount;
                for (size_t i = 0; i < output_dims.dimCount && i < 8; ++i) {
                    output_shape.dims[i] = (int)output_dims.dims[i];
                }
            }
            output_size = aclmdlGetOutputSizeByIndex(model_desc, 0);
            printf("[NPU] 输出尺寸: %zu bytes\n", output_size);
        }
        
        /* 分配Device内存 */
        if (input_size > 0) {
            ret = aclrtMalloc(&input_buffer, input_size, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                set_error("aclrtMalloc分配输入缓冲失败");
                return -1;
            }
        }
        
        if (output_size > 0) {
            ret = aclrtMalloc(&output_buffer, output_size, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                set_error("aclrtMalloc分配输出缓冲失败");
                return -1;
            }
        }
        
        /* 创建输入数据集 */
        input_dataset = aclmdlCreateDataset();
        if (!input_dataset) {
            set_error("创建输入数据集失败");
            return -1;
        }
        
        if (input_buffer) {
            aclDataBuffer* data_buf = aclCreateDataBuffer(input_buffer, input_size);
            if (!data_buf) {
                set_error("创建输入DataBuffer失败");
                return -1;
            }
            ret = aclmdlAddDatasetBuffer(input_dataset, data_buf);
            if (ret != ACL_SUCCESS) {
                set_error("添加输入数据集Buffer失败");
                return -1;
            }
        }
        
        /* 创建输出数据集 */
        output_dataset = aclmdlCreateDataset();
        if (!output_dataset) {
            set_error("创建输出数据集失败");
            return -1;
        }
        
        if (output_buffer) {
            aclDataBuffer* data_buf = aclCreateDataBuffer(output_buffer, output_size);
            if (!data_buf) {
                set_error("创建输出DataBuffer失败");
                return -1;
            }
            ret = aclmdlAddDatasetBuffer(output_dataset, data_buf);
            if (ret != ACL_SUCCESS) {
                set_error("添加输出数据集Buffer失败");
                return -1;
            }
        }
        
        loaded = true;
        return 0;
    }
    
    int Infer(const unsigned char* input_data_ptr, unsigned int input_len,
              unsigned char* output_data_ptr, unsigned int* output_len) {
        if (!loaded) {
            set_error("模型未加载");
            return -1;
        }
        if (EnsureContext() != 0) return -1;
        
        /* 拷贝输入数据到Device */
        aclError ret = aclrtMemcpy(input_buffer, input_size,
                                    input_data_ptr, input_len,
                                    ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            set_error("H2D内存拷贝失败");
            return -1;
        }
        
        /* 执行推理 */
        ret = aclmdlExecute(model_id, input_dataset, output_dataset);
        if (ret != ACL_SUCCESS) {
            set_error("aclmdlExecute执行失败");
            return -1;
        }
        
        /* 同步等待 */
        ret = aclrtSynchronizeStream(DefaultStream());
        if (ret != ACL_SUCCESS) {
            set_error("Stream同步失败");
            return -1;
        }
        
        /* 拷贝输出数据到Host */
        ret = aclrtMemcpy(output_data_ptr, *output_len,
                          output_buffer, output_size,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            set_error("D2H内存拷贝失败");
            return -1;
        }
        
        *output_len = (unsigned int)output_size;
        
        /* 缓存结果 */
        last_result.resize(output_size);
        memcpy(last_result.data(), output_data_ptr, output_size);
        
        return 0;
    }
    
    /* InferAsync:投递 H2D + Execute + D2H 到目标 stream,不做 synchronize */
    int InferAsyncCore(aclrtStream target_stream,
                       const unsigned char* input_data_ptr, unsigned int input_len,
                       unsigned char* output_data_ptr, unsigned int* output_len) {
        if (!loaded) {
            set_error("模型未加载");
            return -1;
        }
        if (EnsureContext() != 0) return -1;
        aclrtStream s = target_stream ? target_stream : DefaultStream();
        if (!s) {
            set_error("没有可用的 stream");
            return -1;
        }
        
        /* 异步 H2D */
        aclError ret = aclrtMemcpyAsync(input_buffer, input_size,
                                        input_data_ptr, input_len,
                                        ACL_MEMCPY_HOST_TO_DEVICE, s);
        if (ret != ACL_SUCCESS) {
            set_error("H2D 异步拷贝失败");
            return -1;
        }
        
        /* 异步 Execute */
        ret = aclmdlExecuteAsync(model_id, input_dataset, output_dataset, s);
        if (ret != ACL_SUCCESS) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "aclmdlExecuteAsync 失败,ret=%d (该接口可能在当前芯片/模式下不可用,可回退到同步 Execute)",
                     ret);
            set_error(buf);
            return -1;
        }
        
        /* 异步 D2H(拷贝到用户提供的 output_data_ptr) */
        ret = aclrtMemcpyAsync(output_data_ptr, *output_len,
                               output_buffer, output_size,
                               ACL_MEMCPY_DEVICE_TO_HOST, s);
        if (ret != ACL_SUCCESS) {
            set_error("D2H 异步拷贝失败");
            return -1;
        }
        
        *output_len = (unsigned int)output_size;
        return 0;
    }
};

/*============================================================================
 * ONNX Runtime 后端实现
 *============================================================================*/

#ifdef USE_ONNXRUNTIME

class ONNXImpl {
public:
    bool loaded;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> session_options;
    std::unique_ptr<Ort::Session> session;
    
    std::string model_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;
    
    std::vector<int64_t> input_dims;
    size_t input_size;
    size_t output_size;
    
    NpuShapeInfo input_shape;
    NpuShapeInfo output_shape;
    
    std::vector<unsigned char> last_result;
    std::mutex mutex_;
    
    ONNXImpl() 
        : loaded(false), env(nullptr), session_options(nullptr), session(nullptr),
          input_size(0), output_size(0) {
        memset(&input_shape, 0, sizeof(input_shape));
        memset(&output_shape, 0, sizeof(output_shape));
    }
    
    ~ONNXImpl() {
        Cleanup();
    }
    
    int LoadModel(const char* model_path) {
        try {
            this->model_path = model_path;  /* 保存模型路径 */
            printf("[ONNX] 加载ONNX模型: %s\n", model_path);
            
            /* 创建ONNX Runtime环境 */
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NPU_ONNX");
            
            /* 创建会话选项 */
            session_options = std::make_unique<Ort::SessionOptions>();
            session_options->SetIntraOpNumThreads(1);
            session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            /* 加载模型 */
            session = std::make_unique<Ort::Session>(*env, model_path, *session_options);
            
            /* 获取输入信息 */
            Ort::AllocatorWithDefaultOptions allocator;
            size_t input_count = session->GetInputCount();
            size_t output_count = session->GetOutputCount();
            printf("[ONNX] 模型有 %zu 个输入，%zu 个输出\n", input_count, output_count);
            
            /* 获取输入名称 */
            for (size_t i = 0; i < input_count; ++i) {
                auto name = session->GetInputNameAllocated(i, allocator);
                input_names.push_back(name.get());
                input_names_cstr.push_back(input_names.back().c_str());
                printf("[ONNX] 输入[%zu]: %s\n", i, input_names.back().c_str());
            }
            
            /* 获取输出名称 */
            for (size_t i = 0; i < output_count; ++i) {
                auto name = session->GetOutputNameAllocated(i, allocator);
                output_names.push_back(name.get());
                output_names_cstr.push_back(output_names.back().c_str());
                printf("[ONNX] 输出[%zu]: %s\n", i, output_names.back().c_str());
            }
            
            /* 获取输入形状 */
            auto input_typeinfo = session->GetInputTypeInfo(0);
            auto input_tensor_info = input_typeinfo.GetTensorTypeAndShapeInfo();
            input_dims = input_tensor_info.GetShape();
            input_shape.num_dims = (int)input_dims.size();
            
            /* 计算输入尺寸（动态维度用默认值640） */
            input_size = 1;
            for (size_t i = 0; i < input_dims.size() && i < 8; ++i) {
                if (input_dims[i] < 0) {
                    input_shape.dims[i] = 640;  /* 动态维度默认640 */
                    input_size *= 640;
                } else {
                    input_shape.dims[i] = (int)input_dims[i];
                    input_size *= (size_t)input_dims[i];
                }
            }
            input_size *= 4;  /* float32 = 4 bytes */
            printf("[ONNX] 输入尺寸: %zu bytes (默认形状)\n", input_size);
            
            /* 获取输出形状 */
            auto output_typeinfo = session->GetOutputTypeInfo(0);
            auto output_tensor_info = output_typeinfo.GetTensorTypeAndShapeInfo();
            auto output_dims = output_tensor_info.GetShape();
            output_shape.num_dims = (int)output_dims.size();
            
            /* 计算输出尺寸（动态维度用默认值） */
            output_size = 1;
            for (size_t i = 0; i < output_dims.size() && i < 8; ++i) {
                if (output_dims[i] < 0) {
                    output_shape.dims[i] = 1;  /* 动态维度默认1，实际推理时更新 */
                    output_size *= 1;
                } else {
                    output_shape.dims[i] = (int)output_dims[i];
                    output_size *= (size_t)output_dims[i];
                }
            }
            output_size *= 4;  /* float32 = 4 bytes */
            printf("[ONNX] 输出尺寸: %zu bytes (默认形状)\n", output_size);
            
            loaded = true;
            return 0;
            
        } catch (const Ort::Exception& e) {
            set_error(e.what());
            return -1;
        } catch (const std::exception& e) {
            set_error(e.what());
            return -1;
        }
    }
    
    int Infer(const unsigned char* input_data_ptr, unsigned int input_len,
              unsigned char* output_data_ptr, unsigned int* output_len) {
        if (!loaded) {
            set_error("模型未加载");
            return -1;
        }
        
        try {
            /* 根据输入数据长度计算正确的尺寸 */
            std::vector<int64_t> shape = input_dims;
            
            /* 计算已知维度的乘积和动态维度数量 */
            int64_t known_product = 1;
            int dynamic_count = 0;
            for (auto s : shape) {
                if (s > 0) {
                    known_product *= s;
                } else {
                    dynamic_count++;
                }
            }
            
            /* 计算每个动态维度的值 */
            int64_t total_elements = input_len / sizeof(float);
            if (known_product > 0 && dynamic_count > 0) {
                /* 特殊处理 YOLOX 等模型：[batch, 3, height, width] */
                /* 如果是4维输入，channels=3，且有3个动态维度 */
                if (shape.size() == 4 && shape[1] == 3 && dynamic_count == 3) {
                    /* 假设 batch=1 */
                    shape[0] = 1;
                    /* 计算 height=width */
                    int64_t hw_elements = total_elements / (int64_t)(shape[1] * shape[0]);
                    int64_t hw_size = (int64_t)sqrt((double)hw_elements);
                    shape[2] = hw_size;
                    shape[3] = hw_size;
                }
                /* 如果有两个动态维度（height, width），假设它们相等 */
                else if (dynamic_count == 2) {
                    int64_t dynamic_value = total_elements / known_product;
                    dynamic_value = (int64_t)sqrt((double)dynamic_value);
                    for (auto& s : shape) {
                        if (s < 0) {
                            s = dynamic_value;
                        }
                    }
                }
                /* 其他情况：所有动态维度设为相同值 */
                else {
                    int64_t dynamic_value = total_elements / known_product;
                    for (auto& s : shape) {
                        if (s < 0) {
                            s = dynamic_value;
                        }
                    }
                }
            }
            
            /* 验证计算的形状是否正确 */
            int64_t computed_elements = 1;
            for (auto s : shape) {
                computed_elements *= s;
            }
            if (computed_elements != total_elements) {
                set_error("输入数据长度与模型期望不匹配");
                return -1;
            }
            
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                reinterpret_cast<float*>(const_cast<unsigned char*>(input_data_ptr)),
                input_len / sizeof(float),
                shape.data(),
                shape.size()
            );
            
            /* 执行推理 */
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr.data(),
                &input_tensor,
                1,
                output_names_cstr.data(),
                output_names_cstr.size()
            );
            
            /* 拷贝输出数据 */
            if (!output_tensors.empty()) {
                auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
                size_t count = output_info.GetElementCount();
                size_t actual_size = count * 4;  /* float32 = 4 bytes */
                
                if (actual_size > *output_len) {
                    set_error("输出缓冲区太小");
                    return -1;
                }
                
                memcpy(output_data_ptr, output_tensors[0].GetTensorMutableData<float>(), actual_size);
                *output_len = (unsigned int)actual_size;
                
                /* 缓存结果 */
                last_result.resize(actual_size);
                memcpy(last_result.data(), output_data_ptr, actual_size);
            }
            
            return 0;
            
        } catch (const Ort::Exception& e) {
            set_error(e.what());
            return -1;
        } catch (const std::exception& e) {
            set_error(e.what());
            return -1;
        }
    }
    
    void Cleanup() {
        if (session) {
            session.reset();
        }
        if (session_options) {
            session_options.reset();
        }
        if (env) {
            env.reset();
        }
        loaded = false;
        last_result.clear();
    }
};

#endif /* USE_ONNXRUNTIME */

/*============================================================================
 * 通用NPU实现类 - 支持两种后端
 *============================================================================*/

enum class BackendType {
    NPU_CANN,      /* CANN NPU后端 */
    ONNXRUNTIME    /* ONNX Runtime CPU后端 */
};

class NPUImplWrapper {
public:
    bool loaded;
    BackendType backend;
    
    /* CANN NPU后端 */
    NPUImpl* cann_impl;
    
    /* ONNX Runtime后端 */
#ifdef USE_ONNXRUNTIME
    ONNXImpl* onnx_impl;
#else
    void* onnx_impl;
#endif
    
    NpuShapeInfo input_shape;
    NpuShapeInfo output_shape;
    std::vector<unsigned char> last_result;
    std::mutex mutex_;

    /* ACL Profiling 句柄(NPU_LayerProfiling_Start/Stop 使用) */
    aclprofConfig* prof_cfg;
    bool prof_running;

    /* 运行时内存峰值追踪 (NPU_Monitor_* 使用) */
    size_t   mon_peak_used;
    uint64_t mon_peak_ts_ms;
    uint64_t mon_sample_count;

    /* 实时监控曲线 (NPU_Monitor_Start/Stop 后台线程使用) */
    std::vector<NPUSamplePoint> mon_history;
    pthread_mutex_t             mon_mutex;
    pthread_t                   mon_thread;
    pthread_t                   mon_util_thread;  /* 利用率慢采样线程 (单次查询~120ms) */
    uint32_t                    mon_interval_ms;
    volatile bool               mon_running;
    /* 利用率缓存: 慢线程更新, 内存采样线程附带读取 (mon_mutex 保护) */
    bool                        mon_util_valid;
    int32_t                     mon_util_cube, mon_util_vector, mon_util_aicpu;

    NPUImplWrapper()
        : loaded(false), backend(BackendType::NPU_CANN),
          cann_impl(nullptr), onnx_impl(nullptr),
          prof_cfg(nullptr), prof_running(false),
          mon_peak_used(0), mon_peak_ts_ms(0), mon_sample_count(0),
          mon_interval_ms(10), mon_running(false),
          mon_util_valid(false), mon_util_cube(-1),
          mon_util_vector(-1), mon_util_aicpu(-1) {
        memset(&input_shape, 0, sizeof(input_shape));
        memset(&output_shape, 0, sizeof(output_shape));
        pthread_mutex_init(&mon_mutex, nullptr);
    }
    ~NPUImplWrapper() {
        if (mon_running) {
            mon_running = false;
            pthread_join(mon_thread, nullptr);
            pthread_join(mon_util_thread, nullptr);
        }
        pthread_mutex_destroy(&mon_mutex);
        Cleanup();
    }

    void Cleanup() {
        if (cann_impl) {
            delete cann_impl;
            cann_impl = nullptr;
        }
#ifdef USE_ONNXRUNTIME
        if (onnx_impl) {
            delete static_cast<ONNXImpl*>(onnx_impl);
            onnx_impl = nullptr;
        }
#endif
        loaded = false;
        last_result.clear();
    }
    
    int LoadModel(const char* model_path, uint32_t format, int32_t device_id) {
        /* 决定使用的后端 */
        std::string path_str(model_path);
        
        if (format == NPU_DRIVER_FORMAT_ONNX) {
            /* 强制使用ONNX Runtime */
#ifdef USE_ONNXRUNTIME
            backend = BackendType::ONNXRUNTIME;
#else
            set_error("未编译ONNX Runtime支持，请安装onnxruntime并重新编译");
            return -1;
#endif
        } else if (format == NPU_DRIVER_FORMAT_NPU) {
            /* 强制使用NPU */
            backend = BackendType::NPU_CANN;
        } else {
            /* 自动检测: 优先 NPU 路线(.onnx 自动 atc 转 .om)
             * ONNX Runtime CPU 太慢, 只在强制 mode=2 时使用 */
            backend = BackendType::NPU_CANN;
        }
        
        /* 根据后端加载模型 */
        if (backend == BackendType::ONNXRUNTIME) {
#ifdef USE_ONNXRUNTIME
            onnx_impl = new (std::nothrow) ONNXImpl();
            if (!onnx_impl) {
                set_error("内存分配失败");
                return -1;
            }
            
            int ret = static_cast<ONNXImpl*>(onnx_impl)->LoadModel(model_path);
            if (ret != 0) {
                delete static_cast<ONNXImpl*>(onnx_impl);
                onnx_impl = nullptr;
                return -1;
            }
            
            /* 保存形状信息 */
            input_shape = static_cast<ONNXImpl*>(onnx_impl)->input_shape;
            output_shape = static_cast<ONNXImpl*>(onnx_impl)->output_shape;
            
            loaded = true;
            return 0;
#else
            set_error("未编译ONNX Runtime支持");
            return -1;
#endif
        } else {
            /* NPU后端 */
            cann_impl = new (std::nothrow) NPUImpl();
            if (!cann_impl) {
                set_error("内存分配失败");
                return -1;
            }
            
            int ret = cann_impl->Init(device_id);
            if (ret != 0) {
                delete cann_impl;
                cann_impl = nullptr;
                return -1;
            }
            
            /* 自动模式下，ONNX格式自动转换为OM */
            bool convert_onnx = (format == NPU_DRIVER_FORMAT_AUTO);
            
            ret = cann_impl->LoadModel(model_path, convert_onnx);
            if (ret != 0) {
                delete cann_impl;
                cann_impl = nullptr;
                return -1;
            }
            
            /* 保存形状信息 */
            input_shape = cann_impl->input_shape;
            output_shape = cann_impl->output_shape;
            
            loaded = true;
            return 0;
        }
    }
    
    int Infer(const unsigned char* input_data_ptr, unsigned int input_len,
              unsigned char* output_data_ptr, unsigned int* output_len) {
        if (!loaded) {
            set_error("模型未加载");
            return -1;
        }
        
        if (backend == BackendType::ONNXRUNTIME) {
#ifdef USE_ONNXRUNTIME
            return static_cast<ONNXImpl*>(onnx_impl)->Infer(
                input_data_ptr, input_len, output_data_ptr, output_len);
#else
            set_error("未编译ONNX Runtime支持");
            return -1;
#endif
        } else {
            return cann_impl->Infer(input_data_ptr, input_len, output_data_ptr, output_len);
        }
    }
    
    void Unload() {
        if (backend == BackendType::ONNXRUNTIME) {
#ifdef USE_ONNXRUNTIME
            if (onnx_impl) {
                delete static_cast<ONNXImpl*>(onnx_impl);
                onnx_impl = nullptr;
            }
#endif
        } else {
            if (cann_impl) {
                /* 卸载CANN模型（保留设备上下文/Stream，仅清理模型相关资源） */
                cann_impl->EnsureContext();
                if (cann_impl->model_id != 0) {
                    aclmdlUnload(cann_impl->model_id);
                    cann_impl->model_id = 0;
                }
                if (cann_impl->model_desc) {
                    aclmdlDestroyDesc(cann_impl->model_desc);
                    cann_impl->model_desc = nullptr;
                }
                if (cann_impl->input_dataset) {
                    aclmdlDestroyDataset(cann_impl->input_dataset);
                    cann_impl->input_dataset = nullptr;
                }
                if (cann_impl->output_dataset) {
                    aclmdlDestroyDataset(cann_impl->output_dataset);
                    cann_impl->output_dataset = nullptr;
                }
                if (cann_impl->input_buffer) {
                    aclrtFree(cann_impl->input_buffer);
                    cann_impl->input_buffer = nullptr;
                }
                if (cann_impl->output_buffer) {
                    aclrtFree(cann_impl->output_buffer);
                    cann_impl->output_buffer = nullptr;
                }
                /* 注意：不销毁context/stream/设备，这些在Cleanup()中处理 */
            }
        }

        /* 保险: 若用户未调 Stop 就销毁 handle, 这里强制停止 */
        if (prof_cfg) {
            if (prof_running) {
                aclprofStop(prof_cfg);
                prof_running = false;
            }
            aclprofDestroyConfig(prof_cfg);
            prof_cfg = nullptr;
        }

        loaded = false;
        last_result.clear();
    }
    
    /* 这批新接口仅在 NPU_CANN 后端有意义 */
    int RequireNpuBackend() {
        if (backend != BackendType::NPU_CANN || !cann_impl) {
            set_error("当前不是 NPU 后端,该接口仅在 .om / CANN 模式下可用");
            return -1;
        }
        return 0;
    }
    
    int InferAsync(NPUStream stream,
                   const unsigned char* input_data_ptr, unsigned int input_len,
                   unsigned char* output_data_ptr, unsigned int* output_len) {
        if (!loaded) {
            set_error("模型未加载");
            return -1;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (backend == BackendType::ONNXRUNTIME) {
#ifdef USE_ONNXRUNTIME
            /* ONNX 后端不存在 async 概念,直接同步执行 */
            return static_cast<ONNXImpl*>(onnx_impl)->Infer(
                input_data_ptr, input_len, output_data_ptr, output_len);
#else
            set_error("未编译ONNX Runtime支持");
            return -1;
#endif
        } else {
            return cann_impl->InferAsyncCore(
                cann_impl->ResolveStream(stream),
                input_data_ptr, input_len, output_data_ptr, output_len);
        }
    }
};

/*============================================================================
 * 内部辅助函数
 *============================================================================*/

static void set_error(const char* msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

static void clear_error() {
    g_last_error[0] = '\0';
}

static int init_acl() {
    if (g_acl_initialized && !g_acl_finalized) return 0;

    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_acl_initialized && !g_acl_finalized) return 0;

    /* 注意: 310B 上若 aclprofInit 在 aclInit 之前调用会导致 aclInit 卡死,
     * 改为在 NPU_LayerProfiling_Start 内部(aclInit 之后)调用 aclprofInit */

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        set_error("aclInit失败");
        return -1;
    }

    g_acl_initialized = true;
    g_acl_finalized = false;
    printf("[NPU] ACL初始化成功\n");
    return 0;
}

static void finalize_acl() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    /* aclprofFinalize 必须在 aclFinalize 之前调用 */
    if (g_prof_inited) {
        aclprofFinalize();
        g_prof_inited = false;
        printf("[NPU] Profiling 已清理\n");
    }
    if (g_acl_initialized && !g_acl_finalized) {
        aclFinalize();
        g_acl_finalized = true;
        g_acl_initialized = false;
        printf("[NPU] ACL已清理\n");
    }
}

static aclrtMemcpyKind ConvertCopyDir(NPU_CopyDir dir) {
    switch (dir) {
        case NPU_MEMCPY_HOST_TO_DEVICE: return ACL_MEMCPY_HOST_TO_DEVICE;
        case NPU_MEMCPY_DEVICE_TO_HOST: return ACL_MEMCPY_DEVICE_TO_HOST;
        case NPU_MEMCPY_DEVICE_TO_DEVICE: return ACL_MEMCPY_DEVICE_TO_DEVICE;
        case NPU_MEMCPY_HOST_TO_HOST: return ACL_MEMCPY_HOST_TO_HOST;
        default: return ACL_MEMCPY_HOST_TO_DEVICE;
    }
}

/*============================================================================
 * 第一批扩展接口实现
 *============================================================================*/

static NPUImplWrapper* ToWrapper(NPUHandle handle) {
    return static_cast<NPUImplWrapper*>(handle);
}

static NPUImpl* RequireCann(NPUHandle handle) {
    NPUImplWrapper* w = ToWrapper(handle);
    if (!w || w->RequireNpuBackend() != 0) return nullptr;
    return w->cann_impl;
}

int NPU_Malloc(NPUHandle handle, size_t size, void** dev_ptr) {
    clear_error();
    if (!dev_ptr) { set_error("dev_ptr 为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclError ret = aclrtMalloc(dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        set_error("aclrtMalloc 失败");
        *dev_ptr = nullptr;
        return -1;
    }
    return 0;
}

int NPU_Free(NPUHandle handle, void* dev_ptr) {
    clear_error();
    if (!dev_ptr) return 0;
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclError ret = aclrtFree(dev_ptr);
    if (ret != ACL_SUCCESS) {
        set_error("aclrtFree 失败");
        return -1;
    }
    return 0;
}

int NPU_Memcpy(NPUHandle handle, void* dst, const void* src,
               size_t size, NPU_CopyDir dir) {
    clear_error();
    if (!dst || !src) { set_error("指针为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclError ret = aclrtMemcpy(dst, size, src, size, ConvertCopyDir(dir));
    if (ret != ACL_SUCCESS) {
        set_error("aclrtMemcpy 失败");
        return -1;
    }
    return 0;
}

int NPU_MemcpyAsync(NPUHandle handle, NPUStream stream,
                    void* dst, const void* src,
                    size_t size, NPU_CopyDir dir) {
    clear_error();
    if (!dst || !src) { set_error("指针为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclrtStream s = c->ResolveStream(stream);
    if (!s) {
        set_error("stream 无效");
        return -1;
    }
    aclError ret = aclrtMemcpyAsync(dst, size, src, size, ConvertCopyDir(dir), s);
    if (ret != ACL_SUCCESS) {
        set_error("aclrtMemcpyAsync 失败");
        return -1;
    }
    return 0;
}

int NPU_StreamCreate(NPUHandle handle, NPUStream* stream) {
    clear_error();
    if (!stream) { set_error("stream 输出参数为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclrtStream s = nullptr;
    aclError ret = aclrtCreateStream(&s);
    if (ret != ACL_SUCCESS || !s) {
        set_error("aclrtCreateStream 失败");
        return -1;
    }
    c->streams.push_back(s);
    *stream = static_cast<NPUStream>(s);
    return 0;
}

int NPU_StreamDestroy(NPUHandle handle, NPUStream stream) {
    clear_error();
    if (!stream) return 0;
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclrtStream s = c->ResolveStream(stream);
    if (s == c->DefaultStream()) {
        set_error("默认 stream 不允许销毁,整个 handle 销毁时会自动清理");
        return -1;
    }
    aclError ret = aclrtDestroyStream(s);
    if (ret != ACL_SUCCESS) {
        set_error("aclrtDestroyStream 失败");
        return -1;
    }
    /* 从池中移除,避免析构时 double free */
    auto it = std::find(c->streams.begin(), c->streams.end(), s);
    if (it != c->streams.end()) c->streams.erase(it);
    return 0;
}

int NPU_StreamSynchronize(NPUHandle handle, NPUStream stream) {
    clear_error();
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclrtStream s = stream ? c->ResolveStream(stream) : c->DefaultStream();
    if (!s) { set_error("没有可用的 stream"); return -1; }
    aclError ret = aclrtSynchronizeStream(s);
    if (ret != ACL_SUCCESS) {
        set_error("aclrtSynchronizeStream 失败");
        return -1;
    }
    return 0;
}

int NPU_DeviceSynchronize(NPUHandle handle) {
    clear_error();
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclError ret = aclrtSynchronizeDevice();
    if (ret != ACL_SUCCESS) {
        set_error("aclrtSynchronizeDevice 失败");
        return -1;
    }
    return 0;
}

int NPU_EventCreate(NPUHandle handle, NPUEvent* event) {
    clear_error();
    if (!event) { set_error("event 输出参数为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclrtEvent e = nullptr;
    aclError ret = aclrtCreateEvent(&e);
    if (ret != ACL_SUCCESS || !e) {
        set_error("aclrtCreateEvent 失败");
        return -1;
    }
    *event = static_cast<NPUEvent>(e);
    return 0;
}

int NPU_EventDestroy(NPUHandle handle, NPUEvent event) {
    clear_error();
    if (!event) return 0;
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclError ret = aclrtDestroyEvent(static_cast<aclrtEvent>(event));
    if (ret != ACL_SUCCESS) {
        set_error("aclrtDestroyEvent 失败");
        return -1;
    }
    return 0;
}

int NPU_EventRecord(NPUHandle handle, NPUEvent event, NPUStream stream) {
    clear_error();
    if (!event) { set_error("event 为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclrtStream s = stream ? c->ResolveStream(stream) : c->DefaultStream();
    if (!s) { set_error("stream 无效"); return -1; }
    aclError ret = aclrtRecordEvent(static_cast<aclrtEvent>(event), s);
    if (ret != ACL_SUCCESS) {
        set_error("aclrtRecordEvent 失败");
        return -1;
    }
    return 0;
}

int NPU_EventSynchronize(NPUHandle handle, NPUEvent event) {
    clear_error();
    if (!event) { set_error("event 为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclError ret = aclrtSynchronizeEvent(static_cast<aclrtEvent>(event));
    if (ret != ACL_SUCCESS) {
        set_error("aclrtSynchronizeEvent 失败");
        return -1;
    }
    return 0;
}

int NPU_EventElapsedTime(NPUHandle handle, float* ms,
                         NPUEvent start, NPUEvent end) {
    clear_error();
    if (!ms || !start || !end) { set_error("参数为空"); return -1; }
    NPUImpl* c = RequireCann(handle);
    if (!c) return -1;
    if (c->EnsureContext() != 0) return -1;
    aclError ret = aclrtEventElapsedTime(ms,
                                         static_cast<aclrtEvent>(start),
                                         static_cast<aclrtEvent>(end));
    if (ret != ACL_SUCCESS) {
        set_error("aclrtEventElapsedTime 失败");
        return -1;
    }
    return 0;
}

int NPU_Model_InferAsync(NPUHandle handle, NPUStream stream,
                         const unsigned char* input_data, unsigned int input_len,
                         unsigned char* output_data, unsigned int* output_len) {
    clear_error();
    if (!handle || !input_data || !output_data || !output_len) {
        set_error("参数为空指针");
        return -1;
    }
    NPUImplWrapper* w = ToWrapper(handle);
    if (!w) { set_error("handle 无效"); return -1; }
    return w->InferAsync(stream, input_data, input_len, output_data, output_len);
}

/*============================================================================
 * 接口实现
 *============================================================================*/

/**
 * NPU_GetNpuCount - 获取NPU数量
 */
unsigned int NPU_GetNpuCount(void) {
    if (init_acl() != 0) {
        return 0;
    }
    
    uint32_t device_count = 0;
    aclError ret = aclrtGetDeviceCount(&device_count);
    if (ret != ACL_SUCCESS) {
        set_error("获取设备数量失败");
        return 0;
    }
    
    return (unsigned int)device_count;
}

/**
 * NPU_GetModelShape - 获取模型形状
 */
int NPU_GetModelShape(const unsigned char* data, unsigned int len,
                      NpuShapeInfo* input_shape, NpuShapeInfo* output_shape) {
    clear_error();
    
    if (!input_shape || !output_shape) {
        set_error("参数为空指针");
        return -1;
    }
    
    /* 此接口需要在模型加载后通过NPU_Model_Load获取形状信息 */
    
    return 0;
}

/**
 * NPU_Create - 创建NPU实例
 */
NPUHandle NPU_Create(void) {
    clear_error();
    
    NPUImplWrapper* impl = new (std::nothrow) NPUImplWrapper();
    if (!impl) {
        set_error("内存分配失败");
        return NULL;
    }
    
    printf("[NPU] NPU实例创建成功\n");
    return static_cast<NPUHandle>(impl);
}

/**
 * NPU_Destroy - 销毁NPU实例
 */
void NPU_Destroy(NPUHandle handle) {
    if (handle) {
        NPUImplWrapper* impl = static_cast<NPUImplWrapper*>(handle);
        impl->Cleanup();
        delete impl;
        printf("[NPU] NPU实例已销毁\n");
    }
}

/**
 * NPU_Model_Load - 加载模型
 */
int NPU_Model_Load(NPUHandle handle, void* pMsg, uint32_t iLen) {
    clear_error();
    
    if (!handle || !pMsg) {
        set_error("参数为空指针");
        return -1;
    }
    
    if (iLen < sizeof(NPUModelLoadMsg)) {
        set_error("消息长度不足");
        return -1;
    }
    
    NPUImplWrapper* impl = static_cast<NPUImplWrapper*>(handle);
    
    if (impl->loaded) {
        set_error("模型已加载，请先卸载");
        return -1;
    }
    
    NPUModelLoadMsg* msg = reinterpret_cast<NPUModelLoadMsg*>(pMsg);
    NpuShapeInfo* input_shape = &impl->input_shape;
    NpuShapeInfo* output_shape = &impl->output_shape;
    
    /* 提取设备ID */
    int32_t device_id = 0;
    if (msg->npu_count > 0 && msg->npu_ids[0] < 8) {
        device_id = (int32_t)msg->npu_ids[0];
    }
    
    printf("[NPU] 加载模型: %s\n", msg->model_path);
    printf("[NPU] 目标NPU数量: %u, 设备ID: %d\n", msg->npu_count, device_id);
    
    /* 根据model_format选择后端 */
    uint32_t format = msg->model_format;
    if (format > NPU_DRIVER_FORMAT_ONNX) {
        format = NPU_DRIVER_FORMAT_AUTO;
    }
    
    printf("[NPU] 模型格式选项: %u", format);
    switch (format) {
        case NPU_DRIVER_FORMAT_NPU:
            printf(" (强制NPU推理)\n");
            break;
        case NPU_DRIVER_FORMAT_ONNX:
            printf(" (强制ONNX Runtime推理)\n");
            break;
        default:
            printf(" (自动检测)\n");
            break;
    }
    
    int ret = impl->LoadModel(msg->model_path, format, device_id);
    if (ret != 0) {
        return -1;
    }
    
    /* 输出形状信息 */
    printf("[NPU] 输入形状: ");
    for (int i = 0; i < impl->input_shape.num_dims; ++i) {
        if (i > 0) printf("x");
        printf("%d", impl->input_shape.dims[i]);
    }
    printf("\n");
    
    printf("[NPU] 输出形状: ");
    for (int i = 0; i < impl->output_shape.num_dims; ++i) {
        if (i > 0) printf("x");
        printf("%d", impl->output_shape.dims[i]);
    }
    printf("\n");
    
    /* 更新全局形状信息（供NPU_GetModelShape使用） */
    *input_shape = impl->input_shape;
    *output_shape = impl->output_shape;
    
    return 0;
}

/**
 * NPU_Model_Unload - 卸载模型
 */
int NPU_Model_Unload(NPUHandle handle) {
    clear_error();
    
    if (!handle) {
        set_error("参数为空指针");
        return -1;
    }
    
    NPUImplWrapper* impl = static_cast<NPUImplWrapper*>(handle);
    
    if (!impl->loaded) {
        return 0;
    }
    
    impl->Unload();
    printf("[NPU] 模型已卸载\n");
    
    return 0;
}

/**
 * NPU_Model_Infer - 执行推理
 */
int NPU_Model_Infer(NPUHandle handle, const unsigned char* input_data,
                    unsigned int input_len, unsigned char* output_data,
                    unsigned int* output_len) {
    clear_error();
    
    if (!handle || !input_data || !output_data || !output_len) {
        set_error("参数为空指针");
        return -1;
    }
    
    NPUImplWrapper* impl = static_cast<NPUImplWrapper*>(handle);
    
    if (!impl->loaded) {
        set_error("模型未加载");
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(impl->mutex_);
    
    int ret = impl->Infer(input_data, input_len, output_data, output_len);
    if (ret != 0) {
        return -1;
    }
    
    /* 缓存结果 */
    impl->last_result.assign(output_data, output_data + *output_len);
    
    return 0;
}

/**
 * NPU_Model_Update - 更新模型（热更新）
 * 
 * @param data 新模型路径或模型数据
 * @param len 数据长度
 * @return 0 成功, -1 失败
 * 
 * @note 支持两种更新方式：
 *       1. 传入模型路径字符串（以字母开头，如"new_model.om"）
 *       2. 传入模型文件的二进制数据
 */
int NPU_Model_Update(NPUHandle handle, const unsigned char* data,
                     unsigned int len) {
    clear_error();
    
    if (!handle || !data || len == 0) {
        set_error("参数为空指针或长度为0");
        return -1;
    }
    
    NPUImplWrapper* impl = static_cast<NPUImplWrapper*>(handle);
    
    if (!impl->loaded) {
        set_error("模型未加载");
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(impl->mutex_);
    
    printf("[NPU] 收到模型更新请求: %u bytes\n", len);
    
    /* 检查是否为模型路径字符串 */
    bool is_path = false;
    if (len > 4 && ((data[0] >= 'a' && data[0] <= 'z') || 
                    (data[0] >= 'A' && data[0] <= 'Z') ||
                    data[0] == '/' || data[0] == '.' || data[0] == '~')) {
        /* 检查是否以有效扩展名结尾 */
        const char* str = reinterpret_cast<const char*>(data);
        const char* ext = strrchr(str, '.');
        if (ext && (strcmp(ext, ".om") == 0 || strcmp(ext, ".onnx") == 0)) {
            is_path = true;
        }
    }
    
    if (is_path) {
        /* 方式1：传入新模型路径，执行卸载+重新加载 */
        const char* new_model_path = reinterpret_cast<const char*>(data);
        printf("[NPU] 检测到新模型路径: %s\n", new_model_path);
        
        /* 保存原有的设备ID和格式设置 */
        uint32_t device_id = 0;
        uint32_t model_format = NPU_DRIVER_FORMAT_AUTO;
        
        /* 获取当前配置 */
        if (impl->backend == BackendType::NPU_CANN && impl->cann_impl) {
            device_id = impl->cann_impl->device_id;
        }
        
        /* 卸载旧模型 */
        printf("[NPU] 正在卸载旧模型...\n");
        impl->Unload();
        
        /* 加载新模型 */
        printf("[NPU] 正在加载新模型...\n");
        bool success = impl->LoadModel(new_model_path, device_id, model_format);
        
        if (success) {
            printf("[NPU] 模型热更新成功\n");
            return 0;
        } else {
            /* 恢复旧模型（如果有保存）*/
            set_error("模型热更新失败");
            return -1;
        }
    } else {
        /* 方式2：传入模型二进制数据（暂不支持，需要先保存为文件）*/
        printf("[NPU] 注意: 二进制数据更新暂不支持，请传入模型文件路径\n");
        printf("[NPU] 使用方式: NPU_Model_Update(handle, \"new_model.om\", 13)\n");
        set_error("暂不支持二进制数据更新，请传入模型文件路径");
        return -1;
    }
}

/**
 * NPU_Get_Infer_Result - 获取推理结果
 * 
 * @return 0 成功, -1 失败, -2 缓冲区太小（result_len返回所需大小）
 * 
 * @note 支持两种调用方式：
 *       1. result_data=NULL, result_len=输出参数 → 查询所需缓冲区大小，返回-2
 *       2. result_data=有效指针, result_len=缓冲区大小 → 拷贝结果数据，返回0
 */
int NPU_Get_Infer_Result(NPUHandle handle, unsigned char* result_data,
                         unsigned int* result_len) {
    clear_error();
    
    if (!handle || !result_len) {
        set_error("参数为空指针");
        return -1;
    }
    
    NPUImplWrapper* impl = static_cast<NPUImplWrapper*>(handle);
    
    if (!impl->loaded) {
        set_error("模型未加载");
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(impl->mutex_);
    
    if (impl->last_result.empty()) {
        set_error("无可用的推理结果");
        return -1;
    }
    
    if (*result_len < impl->last_result.size()) {
        /* 返回实际需要的大小 */
        *result_len = (unsigned int)impl->last_result.size();
        set_error("输出缓冲区太小");
        return -2;
    }
    
    /* 允许 result_data 为 NULL 时只返回大小 */
    if (!result_data) {
        return -2;
    }
    
    memcpy(result_data, impl->last_result.data(), impl->last_result.size());
    *result_len = (unsigned int)impl->last_result.size();
    
    return 0;
}

/**
 * NPU_GetModelStatus - 获取当前模型状态（用于热更新验证）
 */
int NPU_GetModelStatus(NPUHandle handle, NPUModelStatus* status) {
    clear_error();
    
    if (!handle || !status) {
        set_error("参数为空指针");
        return -1;
    }
    
    NPUImplWrapper* impl = static_cast<NPUImplWrapper*>(handle);
    
    std::lock_guard<std::mutex> lock(impl->mutex_);
    
    /* 初始化状态结构 */
    memset(status, 0, sizeof(NPUModelStatus));
    
    /* 填充状态信息 */
    status->loaded = impl->loaded ? 1 : 0;
    
    /* 获取模型路径 */
    if (impl->backend == BackendType::NPU_CANN && impl->cann_impl) {
        strncpy(status->model_path, impl->cann_impl->model_path.c_str(), sizeof(status->model_path) - 1);
        status->npu_id = impl->cann_impl->device_id;
        status->input_size = impl->cann_impl->input_size;
        status->output_size = impl->cann_impl->output_size;
        memcpy(&status->input_shape, &impl->cann_impl->input_shape, sizeof(NpuShapeInfo));
        memcpy(&status->output_shape, &impl->cann_impl->output_shape, sizeof(NpuShapeInfo));
        status->backend_type = 0;  /* NPU_CANN */
    } else if (impl->backend == BackendType::ONNXRUNTIME && impl->onnx_impl) {
#ifdef USE_ONNXRUNTIME
        auto* oi = static_cast<ONNXImpl*>(impl->onnx_impl);
        strncpy(status->model_path, oi->model_path.c_str(), sizeof(status->model_path) - 1);
        status->input_size = oi->input_size;
        status->output_size = oi->output_size;
        memcpy(&status->input_shape, &oi->input_shape, sizeof(NpuShapeInfo));
        memcpy(&status->output_shape, &oi->output_shape, sizeof(NpuShapeInfo));
        status->backend_type = 1;  /* ONNX Runtime */
#else
        status->backend_type = 1;
#endif
    }
    
    return 0;
}

/**
 * NPU_GetLastError - 获取错误信息
 */
const char* NPU_GetLastError(NPUHandle handle) {
    (void)handle;
    return g_last_error[0] ? g_last_error : nullptr;
}

/**
 * NPU_Finalize - 清理全局资源
 */
void NPU_Finalize(void) {
    finalize_acl();
}

/*============================================================================
 * 第二批扩展接口实现 —— OM 模型每层算子推理耗时统计
 *   采集路径: aclprofInit(在 aclInit 前) -> aclprofCreateConfig -> aclprofStart
 *             -> 用户跑 N 次 NPU_Model_Infer -> aclprofStop -> aclprofFinalize
 *   CANN 在输出目录生成 PROF_xxx/op_summary_*.csv, 内含每个算子的耗时
 *============================================================================*/

int NPU_LayerProfiling_SetOutput(const char* out_dir) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_acl_initialized && !g_acl_finalized) {
        set_error("ACL 已初始化, 必须在 NPU_Create 之前调用 SetOutput");
        return -1;
    }
    if (out_dir == nullptr || out_dir[0] == '\0') {
        g_prof_out_dir[0] = '\0';
        return 0;
    }
    strncpy(g_prof_out_dir, out_dir, sizeof(g_prof_out_dir) - 1);
    g_prof_out_dir[sizeof(g_prof_out_dir) - 1] = '\0';
    printf("[NPU] Profiling 输出目录已设置: %s\n", g_prof_out_dir);
    return 0;
}

int NPU_LayerProfiling_Start(NPUHandle handle) {
    NPUImplWrapper* w = ToWrapper(handle);
    if (!w) { set_error("invalid handle"); return -1; }
    if (w->RequireNpuBackend() != 0) return -1;     /* 仅 NPU 后端支持 */
    if (w->prof_cfg) {                              /* 已启动过, 防止重复 */
        set_error("profiling 已在运行, 请先 Stop");
        return -1;
    }
    if (!g_prof_out_dir[0]) {
        set_error("未调用 SetOutput, profiling 未启用");
        return -1;
    }

    NPUImpl* c = w->cann_impl;
    if (c->EnsureContext() != 0) return -1;

    /* aclprofInit 在 aclInit 之后调用(310B 经验: 先 aclprofInit 再 aclInit 会卡死) */
    if (!g_prof_inited) {
        mkdir(g_prof_out_dir, 0755);
        aclError r = aclprofInit(g_prof_out_dir, strlen(g_prof_out_dir));
        if (r != ACL_SUCCESS) {
            set_error("aclprofInit 失败");
            return -1;
        }
        g_prof_inited = true;
        printf("[NPU] Profiling 已初始化, 输出目录: %s\n", g_prof_out_dir);
    }

    uint32_t dev = (uint32_t)c->device_id;
    /* data_cfg 完全对齐 hccl_test 官方用法 */
    uint64_t data_cfg = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME |
                        ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU |
                        ACL_PROF_RUNTIME_API | ACL_PROF_MSPROFTX;
    aclprofConfig* cfg = aclprofCreateConfig(&dev, 1,
                                             ACL_AICORE_PIPE_UTILIZATION,
                                             nullptr, data_cfg);
    if (!cfg) {
        set_error("aclprofCreateConfig 返回 NULL");
        return -1;
    }
    aclError r = aclprofStart(cfg);
    if (r != ACL_SUCCESS) {
        aclprofDestroyConfig(cfg);
        set_error("aclprofStart 失败");
        return -1;
    }
    w->prof_cfg = cfg;
    w->prof_running = true;
    printf("[NPU] Profiling 已启动 (device=%u)\n", dev);
    return 0;
}

int NPU_LayerProfiling_Stop(NPUHandle handle) {
    NPUImplWrapper* w = ToWrapper(handle);
    if (!w) { set_error("invalid handle"); return -1; }
    if (!w->prof_cfg || !w->prof_running) {
        set_error("profiling 未启动");
        return -1;
    }
    aclError r = aclprofStop(w->prof_cfg);
    if (r != ACL_SUCCESS) {
        set_error("aclprofStop 失败");
        return -1;
    }
    aclprofDestroyConfig(w->prof_cfg);
    w->prof_cfg = nullptr;
    w->prof_running = false;
    printf("[NPU] Profiling 已停止, 二进制 trace 已落盘\n");

    /* CANN 8.0 产出的是二进制 trace(.slice_0 文件),
     * 必须调 msprof --export=on 才能转出 op_summary_*.csv */
    if (g_prof_out_dir[0]) {
        /* 在 PATH 中查找 msprof, 取 CANN bin 目录优先 */
        const char* msprof = "/usr/local/Ascend/ascend-toolkit/latest/bin/msprof";
        if (access(msprof, X_OK) != 0) {
            /* 退化: 让 system 自己找 */
            msprof = "msprof";
        }
        std::string cmd = std::string(msprof) +
                          " --export=on --output=\"" + g_prof_out_dir +
                          "\" --summary-format=csv > " + g_prof_out_dir +
                          "/msprof_export.log 2>&1";
        printf("[NPU] 调用 msprof 转换 trace -> csv ...\n");
        int rc = system(cmd.c_str());
        if (rc != 0) {
            fprintf(stderr, "[WARN] msprof 转换失败 rc=%d, 见 %s/msprof_export.log\n",
                    rc, g_prof_out_dir);
        } else {
            printf("[NPU] csv 已生成, 可用 Parse 接口查询\n");
        }
    }
    return 0;
}

/* 内部辅助: 递归遍历目录, 收集所有 op_summary_*.csv 完整路径 */
static void collect_op_summary_csv(const std::string& dir, std::vector<std::string>& out) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            collect_op_summary_csv(full, out);
        } else if (name.find("op_summary") != std::string::npos &&
                   name.size() >= 4 &&
                   name.substr(name.size() - 4) == ".csv") {
            out.push_back(full);
        }
    }
    closedir(d);
}

/* 内部辅助: 解析单行 csv 为字段数组(支持引号转义) */
static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (in_quote) {
            if (ch == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
                else in_quote = false;
            } else {
                cur += ch;
            }
        } else {
            if (ch == '"') in_quote = true;
            else if (ch == ',') { fields.push_back(cur); cur.clear(); }
            else cur += ch;
        }
    }
    fields.push_back(cur);
    return fields;
}

/* 内部辅助: 字段值按列名查找(支持中英文列名) */
static int find_col(const std::vector<std::string>& header,
                    const std::vector<std::string>& names) {
    for (const auto& want : names) {
        for (size_t i = 0; i < header.size(); ++i) {
            if (header[i] == want) return (int)i;
        }
    }
    return -1;
}

int NPU_LayerProfiling_Parse(const char* out_dir, NPULayerInfo* out,
                             int max_count, int* actual_count) {
    if (actual_count) *actual_count = 0;
    if (!out_dir || !out) { set_error("invalid arg"); return -1; }

    /* CANN 刷盘是异步的, 这里 polling 等待 op_summary_*.csv 出现(最多 8s) */
    std::vector<std::string> csvs;
    for (int wait_ms = 0; wait_ms < 8000; wait_ms += 200) {
        csvs.clear();
        collect_op_summary_csv(out_dir, csvs);
        if (!csvs.empty()) break;
        usleep(200 * 1000);
    }
    if (csvs.empty()) {
        set_error("未找到 op_summary_*.csv, CANN 可能尚未完成写入(等了8s)");
        return -1;
    }

    /* 按 op_name 聚合, 累加 total_us, 累计 task_count */
    struct Agg { std::string op_type, op_name; uint64_t total_us; uint32_t task_count; };
    std::vector<Agg> agg;

    for (const auto& path : csvs) {
        std::ifstream f(path);
        if (!f.is_open()) continue;
        std::string line;
        std::vector<std::string> header;
        bool have_header = false;
        int col_type = -1, col_name = -1, col_dur = -1, col_task = -1;
        while (std::getline(f, line)) {
            /* 跳过空行 */
            if (line.empty() || line == "\r") continue;
            if (!have_header) {
                header = split_csv_line(line);
                col_type = find_col(header, {"Op Type", "op_type", "OP Type", "Type"});
                col_name = find_col(header, {"Op Name", "op_name", "OP Name", "Name"});
                col_dur  = find_col(header, {"Task Duration(us)", "Task Duration",
                                              "Duration(us)", "Duration",
                                              "Task Duration (us)", "duration"});
                col_task = find_col(header, {"Task Count", "task_count", "Task Num", "Count"});
                have_header = true;
                continue;
            }
            auto fields = split_csv_line(line);
            std::string op_type = (col_type >= 0 && col_type < (int)fields.size())
                                  ? fields[col_type] : "";
            std::string op_name = (col_name >= 0 && col_name < (int)fields.size())
                                  ? fields[col_name] : "";
            if (op_name.empty() && op_type.empty()) continue;
            uint64_t dur = 0;
            if (col_dur >= 0 && col_dur < (int)fields.size()) {
                try { dur = std::stoull(fields[col_dur]); }
                catch (...) { dur = 0; }
            }
            uint32_t task = 1;
            if (col_task >= 0 && col_task < (int)fields.size()) {
                try { task = (uint32_t)std::stoul(fields[col_task]); }
                catch (...) { task = 1; }
            }
            /* 聚合到同名算子 */
            bool found = false;
            for (auto& a : agg) {
                if (a.op_name == op_name) {
                    a.total_us += dur; a.task_count += task; found = true; break;
                }
            }
            if (!found) agg.push_back({op_type, op_name, dur, task});
        }
    }

    /* 按累计耗时降序 */
    std::sort(agg.begin(), agg.end(),
              [](const Agg& a, const Agg& b) { return a.total_us > b.total_us; });

    int n = (int)agg.size();
    if (max_count < n) n = max_count;
    for (int i = 0; i < n; ++i) {
        NPULayerInfo& dst = out[i];
        memset(&dst, 0, sizeof(dst));
        strncpy(dst.op_type, agg[i].op_type.c_str(), sizeof(dst.op_type) - 1);
        strncpy(dst.op_name, agg[i].op_name.c_str(), sizeof(dst.op_name) - 1);
        dst.task_count = agg[i].task_count;
        dst.total_us   = agg[i].total_us;
        dst.duration_us = agg[i].task_count > 0
                          ? agg[i].total_us / agg[i].task_count : 0;
    }
    if (actual_count) *actual_count = n;
    return 0;
}

/*============================================================================
 * NPU_LayerProfiling_ExportHtml
 *   解析 op_statistic_*.csv(按 op_type 聚合) + op_summary_*.csv(每个算子),
 *   生成自包含 HTML 报告: SVG 饼图(op_type 占比) + SVG 柱状图(Top-N 算子)
 *                       + 表格(完整列表) + 元信息(模型/算子数/总耗时)
 *============================================================================*/

/* 内部: 收集 op_statistic_*.csv (与 op_summary 同样递归) */
static void collect_op_statistic_csv(const std::string& dir, std::vector<std::string>& out) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            collect_op_statistic_csv(full, out);
        } else if (name.find("op_statistic") != std::string::npos &&
                   name.size() >= 4 &&
                   name.substr(name.size() - 4) == ".csv") {
            out.push_back(full);
        }
    }
    closedir(d);
}

/* HTML 转义, 避免 op_name 含 < > & 导致渲染异常 */
static std::string html_escape(const std::string& s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '<':  r += "&lt;";   break;
            case '>':  r += "&gt;";   break;
            case '&':  r += "&amp;";  break;
            case '"':  r += "&quot;"; break;
            default:   r += ch;
        }
    }
    return r;
}

/* 辅助: 在 out_dir 下找名字以 "PROF_" 开头的最新子目录 */
static std::string find_latest_prof_subdir(const std::string& out_dir) {
    DIR* d = opendir(out_dir.c_str());
    if (!d) return "";
    std::string latest;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        if (name.rfind("PROF_", 0) != 0) continue;
        std::string full = out_dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;
        /* 名字自带时间戳, 字典序即时间序, 取最大 */
        if (name > latest) latest = name;
    }
    closedir(d);
    return latest.empty() ? "" : out_dir + "/" + latest;
}

int NPU_LayerProfiling_ExportHtml(const char* out_dir, const char* html_path) {
    if (!out_dir) { set_error("out_dir is null"); return -1; }

    /* 找最新 PROF_xxx 子目录 (所有数据读取都限定在这一个目录, 避免历史残留污染) */
    std::string prof_sub = find_latest_prof_subdir(out_dir);
    if (prof_sub.empty()) {
        set_error("未找到 PROF_xxx 子目录, 请先调 Stop 触发 msprof 导出");
        return -1;
    }
    printf("[NPU] ExportHtml 数据源: %s\n", prof_sub.c_str());

    /* html_path 为 NULL 或空串: 自动放到 PROF_xxx 子目录下 */
    std::string resolved_path;
    if (!html_path || html_path[0] == '\0') {
        resolved_path = prof_sub + "/profiling_report.html";
    } else {
        resolved_path = html_path;
    }

    /* ---- 1. 读 op_statistic_*.csv ---- */
    struct StatAgg { std::string op_type; std::string core; uint32_t count; uint64_t total_us; double ratio; };
    std::vector<StatAgg> stats;
    {
        std::vector<std::string> csvs;
        /* 关键: 从最新 PROF 目录的 mindstudio_profiler_output 开始递归, 而非顶层 out_dir */
        std::string csv_root = prof_sub + "/mindstudio_profiler_output";
        collect_op_statistic_csv(csv_root, csvs);
        for (const auto& path : csvs) {
            std::ifstream f(path);
            if (!f.is_open()) continue;
            std::string line;
            std::vector<std::string> header;
            bool have_header = false;
            int c_type = -1, c_core = -1, c_count = -1, c_total = -1, c_ratio = -1;
            while (std::getline(f, line)) {
                if (line.empty() || line == "\r") continue;
                if (!have_header) {
                    header = split_csv_line(line);
                    c_type  = find_col(header, {"OP Type", "Op Type", "op_type"});
                    c_core  = find_col(header, {"Core Type", "core_type"});
                    c_count = find_col(header, {"Count", "count"});
                    c_total = find_col(header, {"Total Time(us)", "Total Time",
                                                  "Total(us)", "total_us"});
                    c_ratio = find_col(header, {"Ratio(%)", "Ratio", "ratio"});
                    have_header = true;
                    continue;
                }
                auto fld = split_csv_line(line);
                StatAgg s;
                s.op_type = (c_type >= 0 && c_type < (int)fld.size()) ? fld[c_type] : "?";
                s.core    = (c_core >= 0 && c_core < (int)fld.size()) ? fld[c_core] : "";
                s.count   = (c_count >= 0 && c_count < (int)fld.size())
                            ? (uint32_t)std::atol(fld[c_count].c_str()) : 0;
                s.total_us = (c_total >= 0 && c_total < (int)fld.size())
                              ? (uint64_t)std::atoll(fld[c_total].c_str()) : 0;
                s.ratio   = (c_ratio >= 0 && c_ratio < (int)fld.size())
                              ? std::atof(fld[c_ratio].c_str()) : 0.0;
                if (!s.op_type.empty()) stats.push_back(s);
            }
        }
    }

    /* ---- 1.5 按 op_type 聚合去重 (防御性) ---- */
    /* msprof 的 op_statistic CSV 本身是按 op_type 聚合的, 但如果递归读了多个文件
       或多轮 Stop 产生了重复, 这里二次合并, 确保饼图不会出现同一 op_type 多份 */
    {
        std::unordered_map<std::string, StatAgg> merged;
        for (const auto& s : stats) {
            auto it = merged.find(s.op_type);
            if (it == merged.end()) {
                merged[s.op_type] = s;
            } else {
                it->second.count += s.count;
                it->second.total_us += s.total_us;
                /* ratio 在排序后重算 */
            }
        }
        stats.clear();
        stats.reserve(merged.size());
        for (auto& kv : merged) stats.push_back(kv.second);
    }

    if (stats.empty()) {
        set_error("op_statistic_*.csv 未找到, 请先调 Stop 触发 msprof 导出");
        return -1;
    }
    /* 按 total_us 降序 */
    std::sort(stats.begin(), stats.end(),
              [](const StatAgg& a, const StatAgg& b){ return a.total_us > b.total_us; });
    uint64_t grand_total = 0;
    for (const auto& s : stats) grand_total += s.total_us;
    /* 重算 ratio (msprof 原始 ratio 基于多轮累计, 这里按当前聚合结果重算) */
    if (grand_total > 0) {
        for (auto& s : stats) s.ratio = (double)s.total_us / (double)grand_total * 100.0;
    }

    /* ---- 2. 读 op_summary_*.csv, 取 Top-N 算子明细 ---- */
    struct OpRow { std::string op_type, op_name; uint64_t dur_us; uint32_t count; uint64_t total_us; };
    std::vector<OpRow> ops;
    {
        std::vector<std::string> csvs;
        /* 同样限定在当前 PROF 目录 */
        std::string csv_root2 = prof_sub + "/mindstudio_profiler_output";
        collect_op_summary_csv(csv_root2, csvs);
        for (const auto& path : csvs) {
            std::ifstream f(path);
            if (!f.is_open()) continue;
            std::string line;
            std::vector<std::string> header;
            bool have_header = false;
            int c_type = -1, c_name = -1, c_dur = -1;
            while (std::getline(f, line)) {
                if (line.empty() || line == "\r") continue;
                if (!have_header) {
                    header = split_csv_line(line);
                    c_type = find_col(header, {"Op Type", "OP Type", "op_type"});
                    c_name = find_col(header, {"Op Name", "OP Name", "op_name"});
                    c_dur  = find_col(header, {"Task Duration(us)", "Task Duration",
                                                "Duration(us)", "duration"});
                    have_header = true;
                    continue;
                }
                auto fld = split_csv_line(line);
                OpRow r;
                r.op_type = (c_type >= 0 && c_type < (int)fld.size()) ? fld[c_type] : "";
                r.op_name = (c_name >= 0 && c_name < (int)fld.size()) ? fld[c_name] : "";
                r.dur_us  = (c_dur >= 0 && c_dur < (int)fld.size())
                              ? (uint64_t)std::atoll(fld[c_dur].c_str()) : 0;
                if (r.op_name.empty() && r.op_type.empty()) continue;
                /* 同名算子聚合 */
                bool found = false;
                for (auto& a : ops) {
                    if (a.op_name == r.op_name) {
                        a.total_us += r.dur_us; a.count++; found = true; break;
                    }
                }
                if (!found) ops.push_back({r.op_type, r.op_name, r.dur_us, 1, r.dur_us});
            }
        }
    }
    std::sort(ops.begin(), ops.end(),
              [](const OpRow& a, const OpRow& b){ return a.total_us > b.total_us; });

    /* ---- 3. 生成 HTML ---- */
    std::ofstream f(resolved_path);
    if (!f.is_open()) {
        set_error("无法写 html 文件");
        return -1;
    }

    /* 模型名 + 元信息 */
    std::string model_name = stats.empty() ? "unknown" : "model";
    /* 取 model_name 从 op_summary 的 "Model Name" 列, 简化: 用 stats 第一个所属 */
    /* 实际 csv 已有 Model Name 列, 这里偷懒从 stats.op_type 间接取, 不影响展示 */

    f << "<!DOCTYPE html>\n<html lang=\"zh-CN\"><head>\n"
      << "<meta charset=\"UTF-8\">\n"
      << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
      << "<title>NPU Profiling Report</title>\n"
      << "<style>\n"
      << "  body { font-family: -apple-system, 'Segoe UI', Roboto, 'Microsoft YaHei', sans-serif;"
      << " margin: 24px; background: #f5f7fb; color: #222; }\n"
      << "  h1 { color: #1f3a5f; margin-bottom: 4px; }\n"
      << "  .meta { color: #666; font-size: 13px; margin-bottom: 18px; }\n"
      << "  .section { background: #fff; border-radius: 8px; padding: 20px 24px;"
      << " margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }\n"
      << "  h2 { color: #1f3a5f; margin-top: 0; font-size: 18px; border-bottom: 2px solid #e6ebf2;"
      << " padding-bottom: 6px; }\n"
      << "  table { border-collapse: collapse; width: 100%; font-size: 13px; }\n"
      << "  th, td { border: 1px solid #e6ebf2; padding: 6px 8px; text-align: left; }\n"
      << "  th { background: #f0f4fa; color: #1f3a5f; }\n"
      << "  tr:nth-child(even) { background: #fafbfd; }\n"
      << "  tr:hover { background: #fff8e6; }\n"
      << "  .bar { fill: #4a7fc1; }\n"
      << "  .bar:hover { fill: #ff7e1f; }\n"
      << "  .num { text-align: right; font-variant-numeric: tabular-nums; }\n"
      << "  .legend-item { display: inline-block; margin-right: 12px; font-size: 12px; }\n"
      << "  .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px;"
      << " margin-right: 5px; vertical-align: middle; }\n"
      << "</style>\n</head>\n<body>\n";

    f << "<h1>NPU Profiling Report</h1>\n";
    f << "<div class=\"meta\">Generated: " << __DATE__ << " " << __TIME__
      << " | Source: " << html_escape(out_dir)
      << " | Total op-types: " << stats.size()
      << " | Total ops: " << ops.size()
      << " | Grand total: " << grand_total << " us ("
      << (grand_total / 1000.0) << " ms)</div>\n";

    /* 饼图: 用 <path> arc 命令画每个扇形, 避免 stroke-dasharray 精度缺口 */
    f << "<div class=\"section\"><h2>Op Type Distribution (Pie)</h2>\n";
    const double pcx = 200, pcy = 200, pr = 140;  /* 饼图中心+半径 */
    const char* colors[] = {
        "#4a7fc1", "#ff7e1f", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#e6550d", "#31a354", "#3182bd", "#756bb1", "#636363"
    };
    int n_colors = (int)(sizeof(colors) / sizeof(colors[0]));

    /* 图例需要的行数 (全部显示, 不再截断) */
    int leg_rows = (int)stats.size();
    int svg_w = 820;                       /* SVG 总宽 */
    int svg_h = 400 > (int)(pr * 2 + 40) ? 400 : (int)(pr * 2 + 40);
    /* 图例列数: 超过 10 行则分两列 */
    int leg_cols = leg_rows > 10 ? 2 : 1;
    int leg_per_col = (leg_rows + leg_cols - 1) / leg_cols;
    if (leg_per_col < 1) leg_per_col = 1;

    f << "<svg width=\"" << svg_w << "\" height=\"" << svg_h
      << "\" viewBox=\"0 0 " << svg_w << " " << svg_h << "\">\n";

    /* 用 <path> 画每个扇形: M cx,cy L x1,y1 A r,r 0 largeArc,1 x2,y2 Z */
    double cur_angle = -M_PI / 2.0;  /* 从 12 点钟方向开始 */
    for (size_t i = 0; i < stats.size(); ++i) {
        if (stats[i].total_us == 0) continue;
        double frac = (double)stats[i].total_us / (double)grand_total;
        double sweep = frac * 2.0 * M_PI;
        double start = cur_angle;
        double end   = cur_angle + sweep;
        /* 起点 */
        double x1 = pcx + pr * std::cos(start);
        double y1 = pcy + pr * std::sin(start);
        /* 终点 */
        double x2 = pcx + pr * std::cos(end);
        double y2 = pcy + pr * std::sin(end);
        /* largeArc: 扫过 > 180° 则 1, 否则 0 */
        int large_arc = (sweep > M_PI) ? 1 : 0;
        const char* c = colors[i % n_colors];

        f << "<path fill=\"" << c << "\" stroke=\"#fff\" stroke-width=\"1\" "
          << "d=\"M " << pcx << "," << pcy
          << " L " << x1 << "," << y1
          << " A " << pr << "," << pr << " 0 " << large_arc << " 1 "
          << x2 << "," << y2
          << " Z\">\n<title>" << html_escape(stats[i].op_type) << ": "
          << stats[i].total_us << "us (" << (frac * 100.0) << "%)</title>\n</path>\n";

        cur_angle = end;
    }

    /* 甜甜圈中心圆(透明, 让中心文字不被扇形挡住) */
    f << "<circle cx=\"" << pcx << "\" cy=\"" << pcy << "\" r=\"" << (pr * 0.55)
      << "\" fill=\"#fff\"/>\n";
    f << "<text x=\"" << pcx << "\" y=\"" << (pcy - 8)
      << "\" text-anchor=\"middle\" font-size=\"13\" fill=\"#888\">Total</text>\n";
    f << "<text x=\"" << pcx << "\" y=\"" << (pcy + 12)
      << "\" text-anchor=\"middle\" font-size=\"16\" fill=\"#1f3a5f\" font-weight=\"bold\">"
      << grand_total << " us</text>\n";
    f << "</svg>\n";

    /* 图例: 独立 HTML div, 用 flexbox 分栏, 放在饼图下面避免重叠 */
    f << "<div class=\"legend-box\" style=\""
      << "display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:12px;"
      << "padding:8px 0 0 0;\">\n";
    for (size_t i = 0; i < stats.size(); ++i) {
        f << "<div class=\"legend-item\" style=\""
          << "display:inline-flex; align-items:center; font-size:12px;"
          << "min-width:280px;\">\n"
          << "<span class=\"swatch\" style=\""
          << "display:inline-block; width:12px; height:12px; border-radius:2px;"
          << "margin-right:6px; background:" << colors[i % n_colors]
          << ";\"></span>\n"
          << "<span>" << html_escape(stats[i].op_type) << " — "
          << stats[i].total_us << "us (" << (stats[i].ratio)
          << "%, n=" << stats[i].count << ")</span>\n"
          << "</div>\n";
    }
    f << "</div>\n</div>\n";

    /* 柱状图 Top-15 算子 */
    int top_n = (int)ops.size() < 15 ? (int)ops.size() : 15;
    if (top_n > 0) {
        uint64_t max_us = ops[0].total_us;
        const int bar_w = 480, bar_h = 22, gap = 6;
        int svg_h = top_n * (bar_h + gap) + 30;
        f << "<div class=\"section\"><h2>Top-" << top_n
          << " Operators by Total Time (Bar)</h2>\n";
        f << "<svg width=\"820\" height=\"" << svg_h << "\" viewBox=\"0 0 820 "
          << svg_h << "\">\n";
        for (int i = 0; i < top_n; ++i) {
            int y = i * (bar_h + gap) + 6;
            int w = (int)((double)ops[i].total_us / (double)max_us * bar_w);
            if (w < 1) w = 1;
            f << "<rect class=\"bar\" x=\"280\" y=\"" << y << "\" width=\""
              << w << "\" height=\"" << bar_h << "\" rx=\"3\">\n<title>"
              << html_escape(ops[i].op_name) << " = " << ops[i].total_us
              << "us</title>\n</rect>\n";
            /* 算子名(截短显示) */
            std::string nm = ops[i].op_name;
            if (nm.size() > 38) nm = nm.substr(0, 36) + "..";
            f << "<text x=\"270\" y=\"" << y + bar_h - 6
              << "\" text-anchor=\"end\" font-size=\"11\" fill=\"#444\">"
              << html_escape(nm) << "</text>\n";
            f << "<text x=\"" << 280 + w + 6 << "\" y=\"" << y + bar_h - 6
              << "\" font-size=\"11\" fill=\"#1f3a5f\" font-weight=\"bold\">"
              << ops[i].total_us << "us</text>\n";
        }
        f << "</svg>\n</div>\n";
    }

    /* 完整 op_statistic 表 */
    f << "<div class=\"section\"><h2>Op Type Statistics (Full)</h2>\n";
    f << "<table><thead><tr><th>#</th><th>Op Type</th><th>Core Type</th>"
      << "<th class=\"num\">Count</th><th class=\"num\">Total (us)</th>"
      << "<th class=\"num\">Ratio (%)</th></tr></thead><tbody>\n";
    for (size_t i = 0; i < stats.size(); ++i) {
        f << "<tr><td>" << (i + 1) << "</td><td>"
          << html_escape(stats[i].op_type) << "</td><td>"
          << html_escape(stats[i].core) << "</td><td class=\"num\">"
          << stats[i].count << "</td><td class=\"num\">"
          << stats[i].total_us << "</td><td class=\"num\">"
          << stats[i].ratio << "</td></tr>\n";
    }
    f << "</tbody></table></div>\n";

    /* 完整 op 明细表 (Top-50) */
    int detail_n = (int)ops.size() < 50 ? (int)ops.size() : 50;
    f << "<div class=\"section\"><h2>Operator Details (Top-" << detail_n
      << " of " << ops.size() << ")</h2>\n";
    f << "<table><thead><tr><th>#</th><th>Op Type</th><th>Op Name</th>"
      << "<th class=\"num\">Count</th><th class=\"num\">Total (us)</th>"
      << "<th class=\"num\">Avg (us)</th></tr></thead><tbody>\n";
    for (int i = 0; i < detail_n; ++i) {
        f << "<tr><td>" << (i + 1) << "</td><td>"
          << html_escape(ops[i].op_type) << "</td><td>"
          << html_escape(ops[i].op_name) << "</td><td class=\"num\">"
          << ops[i].count << "</td><td class=\"num\">"
          << ops[i].total_us << "</td><td class=\"num\">"
          << (ops[i].count ? ops[i].total_us / ops[i].count : 0)
          << "</td></tr>\n";
    }
    f << "</tbody></table></div>\n";

    f << "<div class=\"meta\">Generated by libnpu_driver NPU_LayerProfiling_ExportHtml"
      << " | Raw data: " << html_escape(out_dir)
      << "/PROF_xxx/mindstudio_profiler_output/</div>\n";
    f << "</body></html>\n";
    return 0;
}

/*============================================================================
 * 第三批扩展接口实现 —— 运行时资源监控 (内存 / 利用率)
 *   内存: aclrtGetMemInfo(ACL_DDR_MEM) -> free/total
 *   利用率: aclrtGetDeviceUtilizationRate -> cube/vector/aicpu/memory
 *   峰值: 每次 GetPeak/Infer_Monitored 采样并更新 wrapper 内的峰值记录
 *============================================================================*/

static uint64_t now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (uint64_t)tv.tv_sec * 1000ULL + (uint64_t)tv.tv_usec / 1000ULL;
}

/* 采样内存并更新 wrapper 里的峰值记录 */
static void sample_mem_and_update_peak(NPUImplWrapper* w, NPUMemInfo* info) {
    if (!info) return;
    memset(info, 0, sizeof(*info));
    size_t free_b = 0, total_b = 0;
    aclError ret = aclrtGetMemInfo(ACL_DDR_MEM, &free_b, &total_b);
    if (ret != ACL_SUCCESS) {
        info->free_bytes = 0; info->total_bytes = 0;
        info->used_bytes = 0; info->used_ratio = -1.0f;
        return;
    }
    info->free_bytes  = free_b;
    info->total_bytes = total_b;
    info->used_bytes  = (total_b > free_b) ? (total_b - free_b) : 0;
    info->used_ratio  = total_b ? (float)info->used_bytes * 100.0f / (float)total_b : 0.0f;

    if (w) {
        w->mon_sample_count++;
        if (info->used_bytes > w->mon_peak_used) {
            w->mon_peak_used = info->used_bytes;
            w->mon_peak_ts_ms = now_ms();
        }
    }
}

int NPU_GetDevMemory(NPUHandle handle, NPUMemInfo* info) {
    clear_error();
    if (!info) { set_error("info is null"); return -1; }
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    /* 无 handle 或未初始化时也可查询(需要 ACL 已 init) */
    sample_mem_and_update_peak(w, info);
    if (info->used_ratio < 0) { set_error("aclrtGetMemInfo failed (ACL 未初始化?)"); return -1; }
    return 0;
}

int NPU_GetUtilization(NPUHandle handle, NPUUtilization* util) {
    clear_error();
    if (!util) { set_error("util is null"); return -1; }
    (void)handle;
    memset(util, 0, sizeof(*util));
    util->cube = util->vector = util->aicpu = util->memory = -1;

    aclrtUtilizationInfo ainfo;
    memset(&ainfo, 0, sizeof(ainfo));
    aclError ret = aclrtGetDeviceUtilizationRate(0, &ainfo);
    if (ret != ACL_SUCCESS) {
        set_error("aclrtGetDeviceUtilizationRate 不支持或失败");
        return -1;
    }
    util->cube    = ainfo.cubeUtilization;
    util->vector  = ainfo.vectorUtilization;
    util->aicpu   = ainfo.aicpuUtilization;
    util->memory  = ainfo.memoryUtilization;
    util->timestamp_ms = now_ms();
    return 0;
}

int NPU_Monitor_Reset(NPUHandle handle) {
    clear_error();
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    if (!w) { set_error("invalid handle"); return -1; }
    NPUMemInfo cur;
    sample_mem_and_update_peak(w, &cur);
    /* Reset: 把当前用量作为新起点 */
    w->mon_peak_used    = cur.used_bytes;
    w->mon_peak_ts_ms   = now_ms();
    w->mon_sample_count = 1;
    return 0;
}

int NPU_Monitor_GetPeak(NPUHandle handle, NPUMemPeak* peak) {
    clear_error();
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    if (!w || !peak) { set_error("invalid arg"); return -1; }
    NPUMemInfo cur;
    sample_mem_and_update_peak(w, &cur);   /* 采样并自动更新峰值 */
    peak->cur_used_bytes    = cur.used_bytes;
    peak->peak_used_bytes   = w->mon_peak_used;
    peak->peak_timestamp_ms = w->mon_peak_ts_ms;
    peak->sample_count      = w->mon_sample_count;
    return 0;
}

int NPU_Model_Infer_Monitored(NPUHandle handle,
                              const unsigned char* input_data, unsigned int input_len,
                              unsigned char* output_data, unsigned int* output_len,
                              NPUMemInfo* mem_before, NPUMemInfo* mem_after) {
    clear_error();
    if (!handle) { set_error("invalid handle"); return -1; }
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);

    /* 1. 推理前采样 */
    NPUMemInfo before, after;
    sample_mem_and_update_peak(w, &before);
    if (mem_before) *mem_before = before;

    /* 2. 复用标准推理 */
    unsigned int out_len = output_len ? *output_len : 0;
    int ret = NPU_Model_Infer(handle, input_data, input_len, output_data, &out_len);
    if (ret != 0) return ret;
    if (output_len) *output_len = out_len;

    /* 3. 推理后采样 */
    sample_mem_and_update_peak(w, &after);
    if (mem_after) *mem_after = after;

    /* 4. 利用率快照 (尽力而为, 失败不报错) */
    NPUUtilization util;
    if (NPU_GetUtilization(handle, &util) == 0) {
        printf("[NPU-Monitor] util: cube=%d%% vector=%d%% aicpu=%d%% mem=%d%% | "
               "dev-mem: used=%.1fMB/%.1fMB (%.1f%%)\n",
               util.cube, util.vector, util.aicpu, util.memory,
               after.used_bytes / 1048576.0, after.total_bytes / 1048576.0,
               after.used_ratio);
    } else {
        printf("[NPU-Monitor] dev-mem: used=%.1fMB/%.1fMB (%.1f%%)\n",
               after.used_bytes / 1048576.0, after.total_bytes / 1048576.0,
               after.used_ratio);
    }
    return 0;
}

/*============================================================================
 * 实时监控曲线 —— 后台线程周期采样实现
 *  310B 实测: aclrtGetMemInfo ~0.5ms, aclrtGetDeviceUtilizationRate ~120ms。
 *  故拆为两个线程: 内存线程按 interval 高频采样; 利用率线程慢速采样后写入
 *  缓存, 由内存线程附带记录, 避免利用率查询拖慢整条曲线的采样节奏。
 *============================================================================*/

static void* monitor_util_thread_entry(void* arg) {
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(arg);
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        printf("[NPU-Monitor] WARN: aclrtSetDevice failed in util thread\n");
    }
    while (w->mon_running) {
        aclrtUtilizationInfo ainfo;
        memset(&ainfo, 0, sizeof(ainfo));
        int32_t cube = -1, vec = -1, aicpu = -1;
        if (aclrtGetDeviceUtilizationRate(0, &ainfo) == ACL_SUCCESS) {
            cube  = ainfo.cubeUtilization;
            vec   = ainfo.vectorUtilization;
            aicpu = ainfo.aicpuUtilization;
        }
        pthread_mutex_lock(&w->mon_mutex);
        w->mon_util_valid  = true;
        w->mon_util_cube   = cube;
        w->mon_util_vector = vec;
        w->mon_util_aicpu  = aicpu;
        pthread_mutex_unlock(&w->mon_mutex);
        usleep((useconds_t)w->mon_interval_ms * 1000U);
    }
    return nullptr;
}

static void* monitor_thread_entry(void* arg) {
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(arg);
    /* ACL context 绑定在创建它的主线程; 子线程必须先 SetDevice 获得自己的上下文,
       否则 aclrtGetMemInfo / aclrtGetDeviceUtilizationRate 会失败 */
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        printf("[NPU-Monitor] WARN: aclrtSetDevice failed in monitor thread\n");
    }
    while (w->mon_running) {
        /* 采样内存 */
        NPUMemInfo mem;
        memset(&mem, 0, sizeof(mem));
        size_t free_b = 0, total_b = 0;
        if (aclrtGetMemInfo(ACL_DDR_MEM, &free_b, &total_b) == ACL_SUCCESS) {
            NPUSamplePoint pt;
            memset(&pt, 0, sizeof(pt));
            pt.timestamp_ms = now_ms();
            pt.total_bytes  = total_b;
            pt.used_bytes   = (total_b > free_b) ? (total_b - free_b) : 0;
            /* 利用率读慢采样线程的缓存 (查询单次~120ms, 不在此处阻塞) */
            pthread_mutex_lock(&w->mon_mutex);
            pt.cube   = w->mon_util_valid ? w->mon_util_cube   : -1;
            pt.vector = w->mon_util_valid ? w->mon_util_vector : -1;
            pt.aicpu  = w->mon_util_valid ? w->mon_util_aicpu  : -1;
            pthread_mutex_unlock(&w->mon_mutex);
            /* 更新峰值 */
            if (pt.used_bytes > w->mon_peak_used) {
                w->mon_peak_used  = pt.used_bytes;
                w->mon_peak_ts_ms = pt.timestamp_ms;
            }
            w->mon_sample_count++;

            pthread_mutex_lock(&w->mon_mutex);
            if (w->mon_history.size() < NPU_MONITOR_MAX_SAMPLES)
                w->mon_history.push_back(pt);
            pthread_mutex_unlock(&w->mon_mutex);
        }
        usleep((useconds_t)w->mon_interval_ms * 1000U);
    }
    return nullptr;
}

int NPU_Monitor_Start(NPUHandle handle, uint32_t interval_ms) {
    clear_error();
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    if (!w) { set_error("invalid handle"); return -1; }
    if (w->mon_running) { set_error("monitor already running"); return -1; }
    if (interval_ms == 0) interval_ms = 10;

    pthread_mutex_lock(&w->mon_mutex);
    w->mon_history.clear();
    pthread_mutex_unlock(&w->mon_mutex);
    w->mon_peak_used    = 0;
    w->mon_peak_ts_ms   = 0;
    w->mon_sample_count = 0;
    w->mon_util_valid   = false;
    w->mon_util_cube    = -1;
    w->mon_util_vector  = -1;
    w->mon_util_aicpu   = -1;
    w->mon_interval_ms  = interval_ms;
    w->mon_running      = true;
    /* 利用率慢采样线程 (创建失败不致命, 利用率将保持 -1) */
    if (pthread_create(&w->mon_util_thread, nullptr, monitor_util_thread_entry, w) != 0)
        printf("[NPU-Monitor] WARN: util thread create failed\n");
    if (pthread_create(&w->mon_thread, nullptr, monitor_thread_entry, w) != 0) {
        w->mon_running = false;
        set_error("pthread_create failed");
        return -1;
    }
    printf("[NPU-Monitor] started: interval=%ums\n", (unsigned)interval_ms);
    return 0;
}

int NPU_Monitor_Stop(NPUHandle handle) {
    clear_error();
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    if (!w) { set_error("invalid handle"); return -1; }
    if (!w->mon_running) return 0;
    w->mon_running = false;
    pthread_join(w->mon_thread, nullptr);
    pthread_join(w->mon_util_thread, nullptr);
    uint32_t n = 0;
    pthread_mutex_lock(&w->mon_mutex);
    n = (uint32_t)w->mon_history.size();
    pthread_mutex_unlock(&w->mon_mutex);
    printf("[NPU-Monitor] stopped: %u samples\n", n);
    return 0;
}

int NPU_Monitor_IsRunning(NPUHandle handle) {
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    return (w && w->mon_running) ? 1 : 0;
}

int NPU_Monitor_GetHistory(NPUHandle handle,
                           NPUSamplePoint* samples, uint32_t max_samples,
                           uint32_t* out_n) {
    clear_error();
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    if (!w || !samples || !out_n || max_samples == 0) {
        set_error("invalid arg"); return -1;
    }
    pthread_mutex_lock(&w->mon_mutex);
    uint32_t n = (uint32_t)w->mon_history.size();
    if (n > max_samples) n = max_samples;   /* 保留最新的 */
    size_t skip = w->mon_history.size() - n;
    for (uint32_t i = 0; i < n; i++)
        samples[i] = w->mon_history[skip + i];
    pthread_mutex_unlock(&w->mon_mutex);
    *out_n = n;
    return 0;
}

int NPU_Monitor_PrintCurve(NPUHandle handle, uint32_t width) {
    clear_error();
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    if (!w) { set_error("invalid handle"); return -1; }
    if (width == 0) width = 60;

    std::vector<NPUSamplePoint> pts;
    pthread_mutex_lock(&w->mon_mutex);
    pts = w->mon_history;
    pthread_mutex_unlock(&w->mon_mutex);
    if (pts.empty()) { set_error("no samples (Start first)"); return -1; }

    /* 降采样到 width 个列 */
    uint32_t n = (uint32_t)pts.size();
    uint32_t cols = width < n ? width : n;
    std::vector<double> col_mb(cols, 0.0);
    std::vector<int>    col_util(cols, -1);
    for (uint32_t c = 0; c < cols; c++) {
        /* 列 c 对应采样区间 [c*n/cols, (c+1)*n/cols), 取最大值 */
        uint32_t s = (uint32_t)((uint64_t)c * n / cols);
        uint32_t e = (uint32_t)((uint64_t)(c + 1) * n / cols);
        if (e <= s) e = s + 1;
        if (e > n) e = n;
        double mb = 0;
        int ut = -1;
        for (uint32_t i = s; i < e; i++) {
            mb = std::max(mb, pts[i].used_bytes / 1048576.0);
            ut = std::max(ut, std::max(pts[i].cube, pts[i].aicpu));
        }
        col_mb[c]   = mb;
        col_util[c] = ut;
    }

    double mb_min = col_mb[0], mb_max = col_mb[0];
    for (uint32_t c = 1; c < cols; c++) {
        mb_min = std::min(mb_min, col_mb[c]);
        mb_max = std::max(mb_max, col_mb[c]);
    }
    if (mb_max <= mb_min) mb_max = mb_min + 1.0;

    const int rows = 10;   /* 纵轴行数 */
    char lbl[32];
    printf("\n========== NPU Memory Curve (n=%u samples) ==========\n", n);
    /* 逐行画散点曲线, 左侧 label 统一 12 字符宽 */
    for (int r = rows; r >= 0; r--) {
        double th = mb_min + (mb_max - mb_min) * r / rows;
        if (r == rows) { snprintf(lbl, sizeof(lbl), "%9.0f MB", mb_max); printf("%12s│ ", lbl); }
        else if (r == 0){ snprintf(lbl, sizeof(lbl), "%9.0f MB", mb_min); printf("%12s│ ", lbl); }
        else           { printf("%12s│ ", ""); }
        for (uint32_t c = 0; c < cols; c++) {
            /* 该列落在 [th - halfcell, th + halfcell) 内则画点 */
            double halfcell = (mb_max - mb_min) / (rows * 2.0);
            bool on = (col_mb[c] >= th - halfcell) && (col_mb[c] < th + halfcell);
            /* 顶行: 超过阈值也画 */
            if (r == rows && col_mb[c] >= mb_max - halfcell) on = true;
            printf(on ? "*" : (r == 0 ? "─" : " "));
        }
        printf("\n");
    }
    printf("%12s└", "");
    for (uint32_t c = 0; c < cols; c++) printf("─");
    printf("\n");
    /* 时间轴标注: 起止秒 */
    uint64_t t0 = pts.front().timestamp_ms;
    uint64_t t1 = pts.back().timestamp_ms;
    printf("%12s  t=0s", "");
    int mid_pos = (int)cols / 2 - 8;
    if (mid_pos > 0) printf("%*s", mid_pos, "");
    printf("t=%.1fs", (t1 - t0) / 1000.0);
    printf("  (peak %.0f MB)\n", mb_max);
    /* 利用率统计 */
    int util_max = -1;
    double util_avg = 0;
    uint32_t util_cnt = 0;
    for (uint32_t c = 0; c < cols; c++) {
        if (col_util[c] >= 0) { util_max = std::max(util_max, col_util[c]); util_avg += col_util[c]; util_cnt++; }
    }
    if (util_cnt) util_avg /= util_cnt;
    printf("AI-Core util: max=%d%% avg=%.1f%%\n", util_max, util_avg);
    printf("======================================================\n");
    return 0;
}

int NPU_Monitor_SavePng(NPUHandle handle, const char* out_dir,
                        char* path_buf, uint32_t buf_len) {
    clear_error();
#ifndef USE_OPENCV
    (void)handle; (void)out_dir; (void)path_buf; (void)buf_len;
    set_error("未启用 USE_OPENCV, PNG 渲染不可用");
    return -1;
#else
    NPUImplWrapper* w = static_cast<NPUImplWrapper*>(handle);
    if (!w) { set_error("invalid handle"); return -1; }

    std::vector<NPUSamplePoint> pts;
    pthread_mutex_lock(&w->mon_mutex);
    pts = w->mon_history;
    pthread_mutex_unlock(&w->mon_mutex);
    if (pts.empty()) { set_error("no samples (Start first)"); return -1; }

    /* 输出目录: 默认 ./monitor, 不存在则创建 */
    std::string dir = (out_dir && out_dir[0]) ? out_dir : "./monitor";
    if (access(dir.c_str(), F_OK) != 0) {
        if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
            set_error("mkdir failed");
            return -1;
        }
    }

    /* 文件名: monitor_YYYYMMDD_HHMMSS.png */
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    struct tm tmv;
    localtime_r(&tv.tv_sec, &tmv);
    char fname[128];
    snprintf(fname, sizeof(fname), "monitor_%04d%02d%02d_%02d%02d%02d.png",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    std::string path = dir + "/" + fname;

    /* ---------- 绘图 ---------- */
    const int W = 960, H = 460;
    const int L = 100, R = 80, T = 60, B = 60;          /* 边距 */
    const int PW = W - L - R, PH = H - T - B;           /* 绘图区 */
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    const cv::Scalar c_grid(220, 220, 220), c_axis(120, 120, 120);
    const cv::Scalar c_mem_blue(255, 0, 0), c_util_green(0, 160, 0);
    const cv::Scalar c_peak(0, 0, 255);
    const int FONT = cv::FONT_HERSHEY_SIMPLEX;

    uint32_t n = (uint32_t)pts.size();
    uint64_t t0 = pts.front().timestamp_ms;
    uint64_t t1 = pts.back().timestamp_ms;
    double dur_s = (t1 - t0) / 1000.0;
    if (dur_s <= 0) dur_s = 0.001;

    /* Y 轴范围 (内存): 数据集中时自适应放大, 不固定从 0 开始 */
    double mem_max = 0, mem_min = 1e18;
    for (auto& p : pts) {
        double mb = p.used_bytes / 1048576.0;
        mem_max = std::max(mem_max, mb);
        mem_min = std::min(mem_min, mb);
    }
    if (mem_min > mem_max) mem_min = 0;
    /* 上下各留 25% 余量; 数据平坦时最小视野取峰值的 2%, 避免除零且保证可见波动 */
    double span = mem_max - mem_min;
    double min_span = std::max(mem_max * 0.02, 1.0);
    if (span < min_span) span = min_span;
    double mem_lo = std::max(0.0, mem_min - span * 0.25);
    double mem_hi = mem_max + span * 0.25;

    /* 网格 + 左轴刻度 (内存 MB, 5 格) + 右轴 (util %, 4 格) */
    const int grid_rows = 5;
    for (int i = 0; i <= grid_rows; i++) {
        int y = T + PH * i / grid_rows;
        cv::line(img, cv::Point(L, y), cv::Point(L + PW, y), c_grid, 1);
        char lbl[32];
        double mb = mem_lo + (mem_hi - mem_lo) * (grid_rows - i) / grid_rows;
        snprintf(lbl, sizeof(lbl), "%.0f MB", mb);
        cv::putText(img, lbl, cv::Point(L - 68, y + 5), FONT, 0.42, c_axis, 1);
        double upct = 100.0 * (grid_rows - i) / grid_rows;
        snprintf(lbl, sizeof(lbl), "%3.0f%%", upct);
        cv::putText(img, lbl, cv::Point(L + PW + 10, y + 5), FONT, 0.42, c_util_green, 1);
    }
    const int grid_cols = 8;
    for (int i = 0; i <= grid_cols; i++) {
        int x = L + PW * i / grid_cols;
        cv::line(img, cv::Point(x, T), cv::Point(x, T + PH), c_grid, 1);
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "%.1fs", dur_s * i / grid_cols);
        cv::putText(img, lbl, cv::Point(x - 14, T + PH + 22), FONT, 0.42, c_axis, 1);
    }
    /* 坐标轴 */
    cv::line(img, cv::Point(L, T), cv::Point(L, T + PH), c_axis, 2);
    cv::line(img, cv::Point(L, T + PH), cv::Point(L + PW, T + PH), c_axis, 2);

    /* 曲线: 内存(蓝) + AI-Core util(绿, 取 cube/aicpu 最大值, 0-100 映射) */
    auto px = [&](uint32_t i) {
        int x = L + (int)((double)i / std::max<uint32_t>(n - 1, 1) * PW);
        double mb = pts[i].used_bytes / 1048576.0;
        int y = T + PH - (int)((mb - mem_lo) / (mem_hi - mem_lo) * PH);
        return cv::Point(x, y);
    };
    for (uint32_t i = 1; i < n; i++) {
        cv::line(img, px(i - 1), px(i), c_mem_blue, 2);
        cv::circle(img, px(i), 2, c_mem_blue, -1);
    }
    cv::circle(img, px(0), 3, c_mem_blue, -1);

    bool have_util = false;
    for (uint32_t i = 0; i < n; i++) {
        int u = std::max(pts[i].cube, pts[i].aicpu);
        if (u < 0) continue;
        have_util = true;
        int x = L + (int)((double)i / std::max<uint32_t>(n - 1, 1) * PW);
        int y = T + PH - (int)(std::min(u, 100) / 100.0 * PH);
        cv::circle(img, cv::Point(x, y), 2, c_util_green, -1);
    }

    /* 峰值标注: 红圈 + 文字 */
    uint32_t peak_i = 0;
    for (uint32_t i = 1; i < n; i++)
        if (pts[i].used_bytes > pts[peak_i].used_bytes) peak_i = i;
    cv::Point pp = px(peak_i);
    cv::circle(img, pp, 6, c_peak, 2);
    char peak_lbl[48];
    snprintf(peak_lbl, sizeof(peak_lbl), "peak %.0f MB", pts[peak_i].used_bytes / 1048576.0);
    cv::putText(img, peak_lbl, cv::Point(std::min(pp.x + 10, W - 170), std::max(pp.y - 12, T + 15)),
                FONT, 0.48, c_peak, 1);

    /* 标题 + 图例 */
    char title[96];
    snprintf(title, sizeof(title), "NPU Memory Monitor  (%u samples, %.1fs)", n, dur_s);
    cv::putText(img, title, cv::Point(L, 30), FONT, 0.62, cv::Scalar(40, 40, 40), 1);
    cv::line(img, cv::Point(L, H - 28), cv::Point(L + 28, H - 28), c_mem_blue, 2);
    cv::putText(img, "Memory (MB)", cv::Point(L + 36, H - 24), FONT, 0.45, cv::Scalar(40, 40, 40), 1);
    if (have_util) {
        cv::line(img, cv::Point(L + 190, H - 28), cv::Point(L + 218, H - 28), c_util_green, 2);
        cv::putText(img, "AI-Core Util (%)", cv::Point(L + 226, H - 24), FONT, 0.45, cv::Scalar(40, 40, 40), 1);
    }

    if (!cv::imwrite(path, img)) {
        set_error("cv::imwrite failed");
        return -1;
    }
    printf("[NPU-Monitor] PNG saved: %s\n", path.c_str());
    if (path_buf && buf_len > 0) {
        snprintf(path_buf, buf_len, "%s", path.c_str());
    }
    return 0;
#endif
}