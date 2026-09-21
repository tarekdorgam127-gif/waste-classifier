from collections import Counter
from pathlib import Path

from waste_classifier.data.split import collect_images, stratified_split


DATA_DIR = Path("data/raw/realwaste-main/RealWaste")


def test_stratified_split():
    records = collect_images(DATA_DIR)

    train, val, test = stratified_split(records)

    assert len(records) == 4752
    assert len(train) + len(val) + len(test) == 4752

    assert len(train) == 3326
    assert len(val) == 713
    assert len(test) == 713

    train_counts = Counter(record.label for record in train)
    val_counts = Counter(record.label for record in val)
    test_counts = Counter(record.label for record in test)

    assert set(train_counts) == set(val_counts) == set(test_counts) == {
        "Cardboard",
        "Food Organics",
        "Glass",
        "Metal",
        "Miscellaneous Trash",
        "Paper",
        "Plastic",
        "Textile Trash",
        "Vegetation",
    }