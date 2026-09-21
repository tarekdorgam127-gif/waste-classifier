from pathlib import Path

import matplotlib.pyplot as plt
import torch

from waste_classifier.data.dataset import CLASS_NAMES
from waste_classifier.data.loaders import create_dataloaders
from waste_classifier.models.cnn import WasteCNN


DATA_DIR = Path("data/raw/realwaste-main/RealWaste")
MODEL_PATH = Path("models/best_model.pth")
OUTPUT_PATH = Path("models/error_analysis.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    _, _, test_loader = create_dataloaders(
        DATA_DIR,
        batch_size=32,
    )

    model = WasteCNN(num_classes=9).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    errors = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu()

            for image, true_label, predicted_label in zip(
                images, labels, predictions
            ):
                if true_label != predicted_label:
                    errors.append(
                        (
                            image,
                            true_label.item(),
                            predicted_label.item(),
                        )
                    )

    print(f"Total errors: {len(errors)}")

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    for ax, (image, true_label, predicted_label) in zip(
        axes.flat,
        errors[:12],
    ):
        image = image.permute(1, 2, 0)
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)

        ax.imshow(image)
        ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[predicted_label]}"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()