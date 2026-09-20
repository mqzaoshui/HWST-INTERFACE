# ==================== resnet_demo: ResNet-18 图像分类 ====================
# 兼容两种构建方式: 顶层 cmake / 独立构建
add_executable(resnet_demo resnet_demo.cpp)

if(TARGET npu_driver)
    target_link_libraries(resnet_demo PRIVATE npu_driver)
else()
    set(_cann_root "")
    foreach(_cand
        "$ENV{ASCEND_TOOLKIT_HOME}"
        "$ENV{ASCEND_HOME_PATH}/ascend-toolkit/latest"
        /usr/local/Ascend/ascend-toolkit/latest
        /usr/local/Ascend/ascend-toolkit/8.0.0.alpha001
    )
        if(EXISTS "${_cand}/acllib/include/acl/acl.h" AND NOT _cann_root)
            set(_cann_root "${_cand}")
        endif()
    endforeach()
    set(CANN_INC "${_cann_root}/acllib/include")
    set(CANN_LIB "${_cann_root}/acllib/lib64")

    get_filename_component(_ws "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(NPU_LIB_SRC "${_ws}/lib")
    set(NPU_STATIC "")
    foreach(_pp "${_ws}/build/lib/libnpu_driver.a"
                "${_ws}/lib/build/libnpu_driver.a")
        if(EXISTS "${_pp}")
            set(NPU_STATIC "${_pp}")
            break()
        endif()
    endforeach()
    if(NOT NPU_STATIC)
        message(FATAL_ERROR "libnpu_driver.a 不存在, 请先编译")
    endif()

    find_library(ONNX_LIB NAMES onnxruntime PATHS /opt/onnxruntime/lib /usr/local/lib /usr/lib)
    find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc highgui imgcodecs)

    target_include_directories(resnet_demo PRIVATE
        ${NPU_LIB_SRC} ${CANN_INC}
    )
    if(OpenCV_FOUND)
        target_include_directories(resnet_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(resnet_demo PRIVATE USE_OPENCV)
    endif()

    find_library(_ascendcl NAMES ascendcl PATHS ${CANN_LIB} NO_DEFAULT_PATH)
    target_link_libraries(resnet_demo PRIVATE
        -Wl,--start-group ${NPU_STATIC} -Wl,--end-group
        ${_ascendcl} ${ONNX_LIB} pthread dl m
    )
    if(OpenCV_FOUND)
        target_link_libraries(resnet_demo PRIVATE ${OpenCV_LIBS})
    endif()
    set_target_properties(resnet_demo PROPERTIES BUILD_RPATH "${CANN_LIB}")
endif()

set_target_properties(resnet_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
