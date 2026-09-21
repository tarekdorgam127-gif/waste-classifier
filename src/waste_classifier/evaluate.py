from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from waste_classifier.data.dataset import CLASS_NAMES
from waste_classifier.data.loaders import create_dataloaders
from waste_classifier.models.cnn import WasteCNN


DATA_DIR = Path("data/raw/realwaste-main/RealWaste")
MODEL_PATH = Path("models/best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    _, _, test_loader = create_dataloaders(
        DATA_DIR,
        batch_size=32,
    )

    model = WasteCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

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
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()