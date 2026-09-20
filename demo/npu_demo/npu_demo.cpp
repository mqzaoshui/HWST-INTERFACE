/**
 * NPU CANN 驱动接口测试Demo
 * 
 * 测试9个核心接口的基本功能
 * 
 * 使用方法：
 *   ./npu_demo [model_path] [model_format]
 *   
 *   model_format:
 *     0 - 自动检测（默认）
 *     1 - 强制NPU推理（.om）
 *     2 - 强制ONNX Runtime推理（.onnx，CPU）
 *   
 * 示例：
 *   ./npu_demo yolox.onnx
 *   ./npu_demo yolox.onnx 2    // 强制ONNX Runtime
 *   ./npu_demo yolox.om 1       // 强制NPU
 */

#include "npu_cann_adapter.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
    int ret = 0;
    NPUHandle handle = NULL;
    const char* model_path = "yolox.om";
    uint32_t model_format = NPU_DRIVER_FORMAT_AUTO;
    
    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        model_format = (uint32_t)atoi(argv[2]);
        if (model_format > NPU_DRIVER_FORMAT_ONNX) {
            model_format = NPU_DRIVER_FORMAT_AUTO;
        }
    }
    
    printf("========================================\n");
    printf("  NPU CANN Driver Demo\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path);
    printf("Format: %u (", model_format);
    switch (model_format) {
        case NPU_DRIVER_FORMAT_NPU: printf("NPU强制"); break;
        case NPU_DRIVER_FORMAT_ONNX: printf("ONNX Runtime强制"); break;
        default: printf("自动检测"); break;
    }
    printf(")\n\n");
    
    /* Step 1: NPU_GetNpuCount - 获取NPU数量 */
    printf("[Step 1] NPU_GetNpuCount\n");
    unsigned int npu_cnt = NPU_GetNpuCount();
    printf("  NPU数量: %u\n", npu_cnt);
    if (npu_cnt == 0) {
        fprintf(stderr, "  错误: 未检测到NPU设备\n");
        return -1;
    }
    printf("  ✓ 成功\n\n");
    
    /* Step 2: NPU_Create - 创建NPU实例 */
    printf("[Step 2] NPU_Create\n");
    handle = NPU_Create();
    if (!handle) {
        fprintf(stderr, "  错误: 创建NPU实例失败: %s\n", NPU_GetLastError(NULL));
        return -1;
    }
    printf("  ✓ 成功\n\n");
    
    /* Step 3: NPU_Model_Load - 加载模型 */
    printf("[Step 3] NPU_Model_Load\n");
    {
        NPUModelLoadMsg msg;
        memset(&msg, 0, sizeof(msg));
        msg.npu_count = 1;
        msg.npu_ids[0] = 0;
        strncpy(msg.model_path, model_path, sizeof(msg.model_path) - 1);
        msg.model_format = model_format;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        ret = NPU_Model_Load(handle, &msg, sizeof(msg));
        auto t2 = std::chrono::high_resolution_clock::now();
        
        if (ret != 0) {
            fprintf(stderr, "  错误: 模型加载失败: %s\n", NPU_GetLastError(handle));
            NPU_Destroy(handle);
            NPU_Finalize();
            return -1;
        }
        
        double load_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        printf("  加载时间: %.2f ms\n", load_time);
        printf("  ✓ 成功\n\n");
    }
    
    /* Step 4: NPU_GetModelShape - 获取模型形状 */
    printf("[Step 4] NPU_GetModelShape\n");
    {
        NpuShapeInfo input_shape, output_shape;
        memset(&input_shape, 0, sizeof(input_shape));
        memset(&output_shape, 0, sizeof(output_shape));
        
        ret = NPU_GetModelShape(NULL, 0, &input_shape, &output_shape);
        if (ret != 0) {
            fprintf(stderr, "  错误: %s\n", NPU_GetLastError(handle));
        } else {
            printf("  输入形状: ");
            for (int i = 0; i < input_shape.num_dims; ++i) {
                if (i > 0) printf("x");
                printf("%d", input_shape.dims[i]);
            }
            printf("\n");
            printf("  输出形状: ");
            for (int i = 0; i < output_shape.num_dims; ++i) {
                if (i > 0) printf("x");
                printf("%d", output_shape.dims[i]);
            }
            printf("\n");
        }
        printf("  ✓ 完成\n\n");
    }
    
    /* Step 5: NPU_Model_Infer - 执行推理 */
    printf("[Step 5] NPU_Model_Infer\n");
    {
        /* 获取模型状态以获取正确的缓冲区大小 */
        NPUModelStatus model_status;
        memset(&model_status, 0, sizeof(model_status));
        ret = NPU_GetModelStatus(handle, &model_status);
        if (ret != 0) {
            fprintf(stderr, "  获取模型状态失败: %s\n", NPU_GetLastError(handle));
            NPU_Destroy(handle);
            NPU_Finalize();
            return -1;
        }
        
        printf("  模型输出大小: %u bytes\n", model_status.output_size);
        
        /* 准备输入数据：1x3x640x640 float32 */
        unsigned int input_size = model_status.input_size;
        std::vector<unsigned char> input_data(input_size, 0x00);
        
        /* 输出缓冲区 - 使用模型实际大小 */
        unsigned int output_size = model_status.output_size;
        std::vector<unsigned char> output_data(output_size);
        unsigned int actual_output_len = output_size;
        
        /* 预热 */
        printf("  预热中...\n");
        for (int i = 0; i < 3; ++i) {
            NPU_Model_Infer(handle, input_data.data(), input_size,
                          output_data.data(), &actual_output_len);
        }
        
        /* 正式推理 */
        const int num_iterations = 10;
        double total_time = 0;
        
        for (int i = 0; i < num_iterations; ++i) {
            auto t1 = std::chrono::high_resolution_clock::now();
            ret = NPU_Model_Infer(handle, input_data.data(), input_size,
                                output_data.data(), &actual_output_len);
            auto t2 = std::chrono::high_resolution_clock::now();
            
            if (ret != 0) {
                fprintf(stderr, "  错误: 推理失败: %s\n", NPU_GetLastError(handle));
                break;
            }
            
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            total_time += elapsed;
        }
        
        if (ret == 0) {
            double avg_time = total_time / num_iterations;
            printf("  推理次数: %d\n", num_iterations);
            printf("  平均耗时: %.2f ms\n", avg_time);
            printf("  吞吐量: %.2f FPS\n", 1000.0 / avg_time);
            printf("  输出长度: %u bytes\n", actual_output_len);
        }
        printf("  ✓ 完成\n\n");
    }
    
    /* Step 6: NPU_Get_Infer_Result - 获取推理结果 */
    printf("[Step 6] NPU_Get_Infer_Result\n");
    {
        /* 先获取所需缓冲区大小（传入NULL指针） */
        unsigned int result_len = 0;
        ret = NPU_Get_Infer_Result(handle, NULL, &result_len);
        
        if (ret == -2) {
            /* 缓冲区太小，result_len 已返回实际大小 */
            printf("  需要缓冲区大小: %u bytes\n", result_len);
            
            /* 分配足够的缓冲区 */
            std::vector<unsigned char> result_data(result_len);
            ret = NPU_Get_Infer_Result(handle, result_data.data(), &result_len);
            
            if (ret == 0) {
                printf("  结果长度: %u bytes\n", result_len);
                printf("  ✓ 成功\n\n");
            } else {
                fprintf(stderr, "  错误: 获取推理结果失败: %s\n", NPU_GetLastError(handle));
            }
        } else if (ret == 0) {
            printf("  结果长度: %u bytes\n", result_len);
            printf("  ✓ 成功\n\n");
        } else {
            fprintf(stderr, "  错误: %s\n", NPU_GetLastError(handle));
        }
    }
    
    /* Step 7: NPU_Model_Update - 更新模型（热更新） */
    printf("[Step 7] NPU_Model_Update\n");
    {
        /* 热更新前：获取当前模型状态 */
        NPUModelStatus status_before;
        ret = NPU_GetModelStatus(handle, &status_before);
        if (ret == 0) {
            printf("  热更新前状态:\n");
            printf("    模型路径: %s\n", status_before.model_path);
            printf("    后端类型: %s\n", status_before.backend_type == 0 ? "NPU_CANN" : "ONNX Runtime");
            printf("    输入大小: %u bytes\n", status_before.input_size);
            printf("    输出大小: %u bytes\n", status_before.output_size);
        }
        
        /* 热更新方式1：传入新模型路径 */
        const char* new_model_path = model_path;  /* 演示用同一模型 */
        printf("  执行热更新: %s\n", new_model_path);
        ret = NPU_Model_Update(handle, 
                               reinterpret_cast<const unsigned char*>(new_model_path),
                               (unsigned int)strlen(new_model_path));
        
        /* 热更新后：再次获取模型状态 */
        NPUModelStatus status_after;
        ret = NPU_GetModelStatus(handle, &status_after);
        if (ret == 0) {
            printf("  热更新后状态:\n");
            printf("    模型路径: %s\n", status_after.model_path);
            printf("    后端类型: %s\n", status_after.backend_type == 0 ? "NPU_CANN" : "ONNX Runtime");
            printf("    输入大小: %u bytes\n", status_after.input_size);
            printf("    输出大小: %u bytes\n", status_after.output_size);
            
            /* 对比状态 */
            if (status_after.loaded == 1) {
                printf("  ✓ 热更新成功\n");
            } else {
                printf("  ✗ 热更新失败，模型未加载\n");
            }
        } else {
            fprintf(stderr, "  警告: %s\n", NPU_GetLastError(handle));
        }
        printf("  ✓ 完成\n\n");
    }
    
    /* Step 8: NPU_Model_Unload - 卸载模型 */
    printf("[Step 8] NPU_Model_Unload\n");
    ret = NPU_Model_Unload(handle);
    if (ret != 0) {
        fprintf(stderr, "  错误: %s\n", NPU_GetLastError(handle));
    } else {
        printf("  ✓ 成功\n\n");
    }
    
    /* Step 9: NPU_Destroy - 销毁实例 */
    printf("[Step 9] NPU_Destroy\n");
    NPU_Destroy(handle);
    handle = NULL;
    printf("  ✓ 成功\n\n");
    
    /* Step 10: NPU_Finalize - 清理全局资源 */
    printf("[Step 10] NPU_Finalize\n");
    NPU_Finalize();
    printf("  ✓ 成功\n\n");
    
    printf("========================================\n");
    printf("  所有接口测试完成！\n");
    printf("========================================\n");
    
    return 0;
}