from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def collect_images(data_dir: Path) -> list[ImageRecord]:
    records = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file():
                records.append(
                    ImageRecord(
                        path=image_path,
                        label=class_dir.name,
                    )
                )

    return records


def stratified_split(
    records: list[ImageRecord],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    paths = [record.path for record in records]
    labels = [record.label for record in records]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=test_size + val_size,
        stratify=labels,
        random_state=random_state,
    )

    relative_val_size = val_size / (test_size + val_size)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=1 - relative_val_size,
        stratify=temp_labels,
        random_state=random_state,
    )

    train = [
        ImageRecord(path, label)
        for path, label in zip(train_paths, train_labels)
    ]

    val = [
        ImageRecord(path, label)
        for path, label in zip(val_paths, val_labels)
    ]

    test = [
        ImageRecord(path, label)
        for path, label in zip(test_paths, test_labels)
    ]

    return train, val, test