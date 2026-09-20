import onnx
import sys
import subprocess
import re
import numpy

def check_numpy_version():
    ver = numpy.__version__
    print(f"🔍 NumPy版本: {ver}")
    major, minor, _ = ver.split(".")[:3]
    major = int(major)
    minor = int(minor)
    # CANN建议 numpy 1.21 ~1.23
    if (major == 1 and 21 <= minor <=23):
        print("✅ NumPy版本符合CANN推荐版本\n")
    else:
        print("⚠️ 警告：当前numpy版本不在CANN推荐(1.21~1.23)范围内，可能引发ATC/TBE/ONNX异常！\n")

def get_soc_version():
    """调用npu‑smi info获取板卡soc_version，适配davinci‑mini输出格式"""
    try:
        ret = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        output = ret.stdout + ret.stderr
        # 匹配 310B1 这类设备名
        match = re.search(r"\|\s*0\s*\|\s*(\w+)\s*\|", output)
        if match:
            dev_name = match.group(1).strip()
            soc = f"Ascend{dev_name}"
            return soc
        else:
            return None
    except Exception:
        return None

def dim_to_str(dim):
    if dim.dim_value:
        return str(dim.dim_value), False
    else:
        return "动态", True


def inspect_onnx_model(model_path: str):
    check_numpy_version()

    # 获取真实soc版本，识别失败默认使用 Ascend310B1
    soc_version = get_soc_version()
    if soc_version is None:
        soc_version = "Ascend310B1"
        print("⚠️ 无法通过npu‑smi识别板卡版本，使用默认占位值：Ascend310B1，请核对修改！")
    else:
        print(f"✅ 自动识别板卡soc_version: {soc_version}")

    model = onnx.load(model_path)
    graph = model.graph
    print("=" * 70)
    print(f"模型文件: {model_path}")
    print(f"ONNX opset 版本: {model.opset_import[0].version}")
    print("=" * 70)

    # ===================== 输入解析 =====================
    print("\n📥 模型输入信息")
    print("-" * 70)
    input_names = []
    input_shapes_raw = []   # 保存解析后的维度列表，数字/字符串
    has_any_dynamic = False

    for idx, inp in enumerate(graph.input):
        name = inp.name
        tensor_type = inp.type.tensor_type
        data_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        dims = tensor_type.shape.dim
        dim_str_list = []
        dim_raw_list = []
        has_dynamic = False
        for d in dims:
            s, dyn = dim_to_str(d)
            dim_str_list.append(s)
            if dyn:
                dim_raw_list.append("1")  # 动态维度默认填充1作为参考值
                has_dynamic = True
                has_any_dynamic = True
            else:
                dim_raw_list.append(str(d.dim_value))

        dim_label = ""
        if len(dims) == 4:
            dim_label = "  [N(批次), C(通道), H(高度), W(宽度)]"

        print(f"输入 {idx+1}:  {name}")
        print(f"  数据类型: {data_type}")
        print(f"  形状尺寸: [{', '.join(dim_str_list)}]{dim_label}")
        if has_dynamic:
            print(f"  ⚠️  包含动态维度，ATC 转换必须用 --input_shape 固定")
        print()

        input_names.append(name)
        input_shapes_raw.append(dim_raw_list)

    # ===================== 输出解析 =====================
    print("\n📤 模型输出信息")
    print("-" * 70)
    for idx, out in enumerate(graph.output):
        name = out.name
        tensor_type = out.type.tensor_type
        data_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        dims = tensor_type.shape.dim
        dim_str_list = []
        for d in dims:
            s, _ = dim_to_str(d)
            dim_str_list.append(s)
        print(f"输出 {idx+1}:  {name}")
        print(f"  数据类型: {data_type}")
        print(f"  形状尺寸: [{', '.join(dim_str_list)}]")
        print()

    # ===================== 动态生成ATC命令 =====================
    # 拼接 input_shape 参数，多输入用分号隔开
    input_shape_items = []
    for name, shape_list in zip(input_names, input_shapes_raw):
        input_shape_items.append(f"{name}:{','.join(shape_list)}")
    atc_input_shape_arg = ";".join(input_shape_items)

    print("\n🛠️  ATC 转换参考命令（自动读取ONNX输入shape + 自动识别soc_version）")
    print("-" * 70)

    # 版本1：基础静态 FP32
    cmd_basic = (
        f'atc --model={model_path} --framework=5 --output=model_out.om '
        f'--soc_version={soc_version} '
        f'--input_shape="{atc_input_shape_arg}" '
        f'--input_format=NCHW '
        f'--buffer_optimize=normalize_omit_out'
    )
    print("【基础静态shape FP32命令】")
    print(cmd_basic)
    print()

    # 版本2：force_fp16 精度（无profiler）
    cmd_fp16 = (
        f'atc --model={model_path} --framework=5 --output=model_out_fp16.om '
        f'--soc_version={soc_version} '
        f'--input_shape="{atc_input_shape_arg}" '
        f'--input_format=NCHW '
        f'--precision_mode=force_fp16 '
        f'--buffer_optimize=normalize_omit_out'
    )
    print("【force_fp16 精度版本】")
    print(cmd_fp16)
    print()

    # 版本3：动态batch示例
    if len(input_shapes_raw[0]) >= 1:
        cmd_dyn = (
            f'atc --model={model_path} --framework=5 --output=model_out_dyn.om '
            f'--soc_version={soc_version} '
            f'--input_shape="{input_names[0]}:-1,{",".join(input_shapes_raw[0][1:])}" '
            f'--input_format=NCHW '
            f'--dynamic_batch_size="1,2,3,4" '
            f'--buffer_optimize=normalize_omit_out'
        )
        print("【动态Batch版本示例（batch 1~4）】")
        print(cmd_dyn)
        print()

    # 打印防OOM环境变量提示
    print("💡转换内存不足时，执行导出环境变量再运行atc：")
    print("export TBE_PARALLEL_NUM=1")
    print()

    if has_any_dynamic:
        print("💡提示：原ONNX存在动态维度，脚本默认填1作为参考值，请确认实际业务维度是否需要修改！")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_onnx_shape.py <onnx模型路径>")
        sys.exit(1)
    inspect_onnx_model(sys.argv[1])
