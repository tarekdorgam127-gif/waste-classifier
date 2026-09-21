from pathlib import Path

from torch.utils.data import DataLoader

from waste_classifier.data.dataset import (
    WasteDataset,
    train_transform,
    eval_transform,
)
from waste_classifier.data.split import collect_images, stratified_split


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
):
    records = collect_images(data_dir)

    train_records, val_records, test_records = stratified_split(records)

    train_dataset = WasteDataset(train_records, train_transform)
    val_dataset = WasteDataset(val_records, eval_transform)
    test_dataset = WasteDataset(test_records, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader