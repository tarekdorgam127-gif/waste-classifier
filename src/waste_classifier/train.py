from pathlib import Path

import torch
from torch import nn, optim

from waste_classifier.data.loaders import create_dataloaders
from waste_classifier.models.cnn import WasteCNN


DATA_DIR = Path("data/raw/realwaste-main/RealWaste")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train():
    train_loader, val_loader, _ = create_dataloaders(
        DATA_DIR,
        batch_size=32,
    )

    model = WasteCNN(num_classes=9).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    best_val_loss = float("inf")

    print(f"Device: {DEVICE}")

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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
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
                MODEL_DIR / "best_model.pth",
            )
            print("  ✓ saved best model")


    model.load_state_dict(
        torch.load(
            MODEL_DIR / "best_model.pth",
            map_location=DEVICE,
        )
    )

    test_loader = create_dataloaders(
        DATA_DIR,
        batch_size=32,
    )[2]

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train()