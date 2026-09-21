import numpy as np
import onnxruntime as ort
import torch

from waste_classifier.models.resnet import WasteResNet18


MODEL_PATH = "models/resnet18_best.pth"
ONNX_PATH = "models/resnet18.onnx"


def main():
    # PyTorch
    model = WasteResNet18(num_classes=9)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        pytorch_output = model(x).numpy()

    # ONNX Runtime
    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["logits"],
        {"images": x.numpy()},
    )[0]

    pytorch_prediction = pytorch_output.argmax(axis=1)[0]
    onnx_prediction = onnx_output.argmax(axis=1)[0]

    max_difference = np.max(
        np.abs(pytorch_output - onnx_output)
    )

    print(f"PyTorch prediction: {pytorch_prediction}")
    print(f"ONNX prediction:    {onnx_prediction}")
    print(f"Max difference:     {max_difference:.8f}")

    if pytorch_prediction == onnx_prediction:
        print("Prediction match: OK")
    else:
        print("Prediction match: FAILED")


if __name__ == "__main__":
    main() 