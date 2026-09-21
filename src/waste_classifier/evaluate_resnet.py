from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from waste_classifier.data.dataset import CLASS_NAMES
from waste_classifier.data.loaders import create_dataloaders
from waste_classifier.models.resnet import WasteResNet18


DATA_DIR = Path("data/raw/realwaste-main/RealWaste")
MODEL_PATH = Path("models/resnet18_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    _, _, test_loader = create_dataloaders(
        DATA_DIR,
        batch_size=32,
    )

    model = WasteResNet18(num_classes=9).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )


if __name__ == "__main__":
    main()