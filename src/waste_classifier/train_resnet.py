from pathlib import Path

import mlflow
import torch
from torch import nn, optim

from waste_classifier.data.loaders import create_dataloaders
from waste_classifier.models.resnet import WasteResNet18


DATA_DIR = Path("data/raw/realwaste-main/RealWaste")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    mlflow.set_experiment("waste-classifier-resnet18")

    with mlflow.start_run(run_name="resnet18-baseline"):

        batch_size = 32
        learning_rate = 1e-4
        epochs = 5

        train_loader, val_loader, test_loader = create_dataloaders(
            DATA_DIR,
            batch_size=batch_size,
        )

        model = WasteResNet18(num_classes=9).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        mlflow.log_params({
            "model": "ResNet18",
            "num_classes": 9,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "optimizer": "Adam",
            "device": str(DEVICE),
            "image_size": 224,
        })

        best_val_loss = float("inf")

        print(f"Device: {DEVICE}")

        # -------------------------
        # Training
        # -------------------------
        for epoch in range(epochs):
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    model.state_dict(),
                    MODEL_PATH,
                )

                mlflow.log_metric(
                    "best_val_loss",
                    best_val_loss,
                    step=epoch + 1,
                )

                print("  ✓ saved best model")

        # -------------------------
        # Test
        # -------------------------
        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE,
            )
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
        )

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        })

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model",
        )

        print("\nFinal Test Results:")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()