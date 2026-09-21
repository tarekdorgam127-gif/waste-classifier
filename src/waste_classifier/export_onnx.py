from pathlib import Path

import onnx
import torch

from waste_classifier.models.resnet import WasteResNet18


MODEL_PATH = Path("models/resnet18_best.pth")
ONNX_PATH = Path("models/resnet18.onnx")

DEVICE = torch.device("cpu")


def main():
    model = WasteResNet18(num_classes=9).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )

    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    print(f"Exported: {ONNX_PATH}")
    print("ONNX validation: OK")


if __name__ == "__main__":
    main()