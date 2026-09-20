# ==================== acl_raw_demo: 纯 ACL 推理 ====================
# 不依赖 npu_driver, 两种构建方式兼容
add_executable(acl_raw_infer acl_raw_infer.cpp)

if(NOT DEFINED CANN_INC)
    set(_cann_root "")
    foreach(_cand "$ENV{ASCEND_TOOLKIT_HOME}"
                "$ENV{ASCEND_HOME_PATH}/ascend-toolkit/latest"
                /usr/local/Ascend/ascend-toolkit/latest
                /usr/local/Ascend/ascend-toolkit/8.0.0.alpha001)
        if(EXISTS "${_cand}/acllib/include/acl/acl.h" AND NOT _cann_root)
            set(_cann_root "${_cand}")
        endif()
    endforeach()
    set(CANN_INC "${_cann_root}/acllib/include")
    set(CANN_LIB "${_cann_root}/acllib/lib64")
endif()

target_include_directories(acl_raw_infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # yolo_common.h
    ${CANN_INC}
)

find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc highgui imgcodecs)
if(OpenCV_FOUND)
    target_include_directories(acl_raw_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(acl_raw_infer PRIVATE USE_OPENCV)
endif()

find_library(_ascendcl NAMES ascendcl PATHS ${CANN_LIB} NO_DEFAULT_PATH)
target_link_libraries(acl_raw_infer PRIVATE ${_ascendcl} pthread dl m)
if(OpenCV_FOUND)
    target_link_libraries(acl_raw_infer PRIVATE ${OpenCV_LIBS})
endif()

set_target_properties(acl_raw_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    BUILD_RPATH "${CANN_LIB}"
)
