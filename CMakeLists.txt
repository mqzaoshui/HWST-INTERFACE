cmake_minimum_required(VERSION 3.10)
project(npu_driver_project LANGUAGES C CXX)

# ==================== 全局 C++ 标准 ====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ==================== 选项 ====================
option(USE_ONNXRUNTIME "Enable ONNX Runtime CPU fallback"  ON)
option(USE_OPENCV      "Enable OpenCV image I/O & viz"      ON)
option(BUILD_DEMOS     "Build all demo executables"         ON)

# ==================== 依赖查找 (仅此处执行一次) ====================

# --- CANN Toolkit ---
# 优先读取环境变量 ASCEND_TOOLKIT_HOME, 其次 ASCEND_HOME_PATH, 否则自动探测
set(_cann_root_candidates "")
if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    list(APPEND _cann_root_candidates "$ENV{ASCEND_TOOLKIT_HOME}")
endif()
if(DEFINED ENV{ASCEND_HOME_PATH})
    list(APPEND _cann_root_candidates "$ENV{ASCEND_HOME_PATH}/ascend-toolkit/latest")
    list(APPEND _cann_root_candidates "$ENV{ASCEND_HOME_PATH}/ascend-toolkit/8.0.0.alpha001")
endif()
list(APPEND _cann_root_candidates
    /usr/local/Ascend/ascend-toolkit/latest
    /usr/local/Ascend/ascend-toolkit/8.0.0.alpha001
)

set(CANN_ROOT "" CACHE PATH "Ascend CANN Toolkit 根目录")
if(NOT CANN_ROOT)
    foreach(_cand ${_cann_root_candidates})
        if(EXISTS "${_cand}/acllib/include/acl/acl.h")
            set(CANN_ROOT "${_cand}")
            break()
        endif()
    endforeach()
endif()

if(NOT CANN_ROOT)
    message(FATAL_ERROR
        "CANN Toolkit 未找到. 请:\n"
        "  1. source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
        "  2. 或 cmake -DCANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest ..")
endif()

set(CANN_INC "${CANN_ROOT}/acllib/include")
set(CANN_LIB "${CANN_ROOT}/acllib/lib64")

# 核心库: ascendcl + acl
find_library(CANN_ASCENDCL  NAMES ascendcl  PATHS ${CANN_LIB} NO_DEFAULT_PATH)
find_library(CANN_ACL       NAMES acl        PATHS ${CANN_LIB} NO_DEFAULT_PATH)
# Profiling 需要 msprofiler
find_library(CANN_MSPROFILER NAMES msprofiler PATHS ${CANN_LIB} NO_DEFAULT_PATH)

if(NOT CANN_ASCENDCL)
    message(FATAL_ERROR "libascendcl.so 未找到于 ${CANN_LIB}")
endif()

# 汇总 CANN 依赖 (供各目标 target_link_libraries 使用)
set(CANN_LIBS "${CANN_ASCENDCL}")
if(CANN_ACL)
    list(APPEND CANN_LIBS "${CANN_ACL}")
endif()
if(CANN_MSPROFILER)
    list(APPEND CANN_LIBS "${CANN_MSPROFILER}")
endif()

message(STATUS "========== CANN ==========")
message(STATUS "CANN_ROOT     : ${CANN_ROOT}")
message(STATUS "CANN_INC      : ${CANN_INC}")
message(STATUS "CANN_LIB      : ${CANN_LIB}")
message(STATUS "AscendCL      : ${CANN_ASCENDCL}")
message(STATUS "msprofiler    : ${CANN_MSPROFILER}")

# --- ONNX Runtime ---
if(USE_ONNXRUNTIME)
    find_path(ONNX_INC
        NAMES onnxruntime_c_api.h
        PATHS /opt/onnxruntime/include
              /usr/local/include
              /usr/include)
    find_library(ONNX_LIB
        NAMES onnxruntime
        PATHS /opt/onnxruntime/lib
              /usr/local/lib
              /usr/lib)
    if(ONNX_INC AND ONNX_LIB)
        message(STATUS "========== ONNX Runtime ==========")
        message(STATUS "ONNX_INC : ${ONNX_INC}")
        message(STATUS "ONNX_LIB : ${ONNX_LIB}")
    else()
        message(WARNING "ONNX Runtime 未找到, 关闭 ONNX 直接推理支持")
        set(USE_ONNXRUNTIME OFF)
    endif()
endif()

# --- OpenCV ---
if(USE_OPENCV)
    find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc highgui imgcodecs)
    if(OpenCV_FOUND)
        message(STATUS "========== OpenCV ==========")
        message(STATUS "OpenCV : ${OpenCV_VERSION}  (${OpenCV_INCLUDE_DIRS})")
    else()
        message(WARNING "OpenCV 未找到, 图像读写/可视化不可用")
        set(USE_OPENCV OFF)
    endif()
endif()

# ==================== 添加子目录 ====================
add_subdirectory(lib)

if(BUILD_DEMOS)
    add_subdirectory(demo)
endif()

message(STATUS "")
message(STATUS "========== 构建配置汇总 ==========")
message(STATUS "CANN Toolkit : ${CANN_ROOT}")
message(STATUS "ONNX Runtime : ${USE_ONNXRUNTIME}")
message(STATUS "OpenCV       : ${USE_OPENCV}")
message(STATUS "Build Demos  : ${BUILD_DEMOS}")
message(STATUS "===================================")
