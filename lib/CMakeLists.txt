# ==================== NPU Driver 核心静态库 ====================
add_library(npu_driver STATIC
    npu_cann_adapter.cpp
)

# --- 编译属性 ---
set_target_properties(npu_driver PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_compile_options(npu_driver PRIVATE -Wall -O2)

# --- 头文件路径 ---
# PUBLIC: 本库编译 + 依赖本库的目标都能找到
target_include_directories(npu_driver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}   # npu_cann_adapter.h
    ${CANN_INC}                   # acl/acl.h
)

# --- 编译宏 ---
if(USE_ONNXRUNTIME)
    target_include_directories(npu_driver PUBLIC ${ONNX_INC})
    target_compile_definitions(npu_driver PUBLIC USE_ONNXRUNTIME)
endif()

if(USE_OPENCV)
    target_include_directories(npu_driver PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(npu_driver PUBLIC USE_OPENCV)
endif()

# --- 链接依赖 ---
# PUBLIC: 依赖本库的目标会自动链接 CANN/ONNX Runtime
target_link_libraries(npu_driver PUBLIC
    ${CANN_LIBS}
    pthread
    dl
    m
)

if(USE_ONNXRUNTIME)
    target_link_libraries(npu_driver PUBLIC ${ONNX_LIB})
endif()

if(USE_OPENCV)
    target_link_libraries(npu_driver PUBLIC ${OpenCV_LIBS})
endif()

# --- RUNTIME RPATH (运行时找 .so) ---
set_target_properties(npu_driver PROPERTIES
    BUILD_RPATH "${CANN_LIB}"
)
if(USE_ONNXRUNTIME)
    get_filename_component(_onnx_lib_dir "${ONNX_LIB}" DIRECTORY)
    set_property(TARGET npu_driver APPEND PROPERTY BUILD_RPATH ";${_onnx_lib_dir}")
endif()

# --- 安装 ---
install(TARGETS npu_driver ARCHIVE DESTINATION lib)
install(FILES npu_cann_adapter.h DESTINATION include)
