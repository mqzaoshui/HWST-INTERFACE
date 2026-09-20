"""
ViT -> ONNX 导出脚本 (预训练版)
- 加载 timm 预训练 ViT-Small/16 (augreg_in1k, ImageNet 1000 类)
- 预训练权重可产生有语义的分类结果, 验证完整 Transformer 推理链路
- 需先安装: pip install timm huggingface_hub safetensors
- 镜像下载: export HF_ENDPOINT=https://hf-mirror.com
用法: python3 gen_vit_onnx.py [输出路径]  (默认 model/tiny_vit.onnx)
"""
import os
import sys
import torch
import timm


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "model/tiny_vit.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # 加载 timm 预训练 ViT-Small/16 (ImageNet-1k, 1000 类)
    # 权重从 hf-mirror 缓存读取, 已下载到 ~/.cache/huggingface/hub/
    print("[INFO] Loading pretrained ViT-Small/16 (augreg_in1k)...")
    model = timm.create_model(
        'vit_small_patch16_224.augreg_in1k', pretrained=True
    ).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] ViT-S params: {n_params / 1e6:.2f} M, classes={model.num_classes}")
    print(f"[INFO] Default input_size: {model.default_cfg.get('input_size')}")
    print(f"[INFO] Mean/std: {model.default_cfg.get('mean')} / {model.default_cfg.get('std')}")

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out,
            input_names=["input"], output_names=["logits"],
            opset_version=13, do_constant_folding=True,
            dynamic_axes=None,
        )
    print(f"[INFO] Exported: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

    # 快速自检
    import onnx
    m = onnx.load(out)
    onnx.checker.check_model(m)
    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:8]
    print("[INFO] Op summary:", ", ".join(f"{k}x{v}" for k, v in top))

    # CPU 参考推理验证: 用标准预处理跑一张图, 看输出是否合理
    print("\n[INFO] CPU reference inference test:")
    cfg = model.default_cfg
    mean = cfg.get('mean', (0.5, 0.5, 0.5))
    std = cfg.get('std', (0.5, 0.5, 0.5))
    img = torch.rand(1, 3, 224, 224)
    # 归一化
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    x = (img - mean_t) / std_t
    with torch.no_grad():
        logits = model(x)
        # 加载 ImageNet 类名(timm 自带)
        labels = cfg.get('label_names')
    top5 = torch.topk(logits[0], 5)
    print("  Top-5 (随机输入, 仅验证前向能跑):")
    for i in range(5):
        idx = top5.indices[i].item()
        name = labels[idx] if labels else f'class_{idx}'
        print(f"    #{i+1}: {idx:4d} {name}  prob~{torch.softmax(logits[0], 0)[idx].item()*100:.2f}%")


if __name__ == "__main__":
    main()
