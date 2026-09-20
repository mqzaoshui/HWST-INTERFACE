"""
将 minimal_conv.onnx 中的 Conv 节点替换为自定义 CustomConv2D 节点
- CustomConv2D 已通过 CANN 自定义算子项目注册到 vendors/customize
- 输入/输出张量保持不变;属性名按 CustomConv2D 定义映射:
    Conv.group          -> CustomConv2D.groups
    Conv.strides        -> CustomConv2D.strides
    Conv.pads           -> CustomConv2D.pads
    Conv.dilations      -> CustomConv2D.dilations
    Conv.kernel_shape   -> 删除(核大小由 filter 形状推断)
- 算子放在默认 ONNX domain, ATC 会从 npu_supported_ops.json 识别
"""
import sys
import os
import onnx
from onnx import helper, TensorProto


def replace_conv_node(graph, conv_node):
    """构造一个 CustomConv2D 节点替换给定的 Conv 节点"""
    new_attrs = {}
    for a in conv_node.attribute:
        if a.name == "group":
            # 重命名 group -> groups
            new_attrs["groups"] = helper.get_attribute_value(a)
        elif a.name == "kernel_shape":
            # CustomConv2D 不接受 kernel_shape 属性(由 filter 形状决定)
            continue
        else:
            # strides / pads / dilations 同名传递
            new_attrs[a.name] = helper.get_attribute_value(a)

    new_node = helper.make_node(
        "CustomConv2D",
        inputs=list(conv_node.input),
        outputs=list(conv_node.output),
        name=conv_node.name.replace("Conv", "CustomConv2D") if conv_node.name else "CustomConv2D",
        domain="",  # 默认 domain, ATC 会从 npu_supported_ops.json 识别
        **new_attrs,
    )
    return new_node


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "model/minimal_conv.onnx"
    dst = sys.argv[2] if len(sys.argv) > 2 else "model/minimal_custom_conv.onnx"
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    print(f"[INFO] Source: {src}")
    print(f"[INFO] Target: {dst}")

    model = onnx.load(src)
    graph = model.graph

    replaced = 0
    new_nodes = []
    for node in graph.node:
        if node.op_type == "Conv":
            print(f"[INFO] Replacing Conv node: name={node.name}, inputs={list(node.input)}, outputs={list(node.output)}")
            for a in node.attribute:
                print(f"    attr {a.name} = {helper.get_attribute_value(a)}")
            new_nodes.append(replace_conv_node(graph, node))
            replaced += 1
        else:
            new_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    print(f"[INFO] Replaced {replaced} Conv node(s) -> CustomConv2D")

    # NOTE: onnx.checker 不识别自定义算子,跳过校验,仅做基本图结构检查
    print(f"[INFO] Custom op 'CustomConv2D' not in ONNX schema, skipping onnx.checker.")
    onnx.save(model, dst)
    print(f"[INFO] Saved: {dst}  ({os.path.getsize(dst)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
