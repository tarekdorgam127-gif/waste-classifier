from fastapi.testclient import TestClient

from waste_classifier.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    with open("data/raw/realwaste-main/RealWaste/Cardboard/Cardboard_1.jpg", "rb") as image:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200

    data = response.json()

    assert "class_name" in data
    assert "confidence" in data   

def test_prediction_response():
    from waste_classifier.api.schemas import PredictionResponse

    result = PredictionResponse(
        class_name="Plastic",
        confidence=0.95,
    )

    assert result.class_name == "Plastic"
    assert result.confidence == 0.95


def test_resnet_forward():
    import torch

    from waste_classifier.models.resnet import WasteResNet18

    model = WasteResNet18(num_classes=9)
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 9)    
def test_class_names():
    from waste_classifier.data.dataset import CLASS_NAMES

    assert len(CLASS_NAMES) == 9
    assert "Plastic" in CLASS_NAMES
    assert "Metal" in CLASS_NAMES 

def test_waste_dataset(tmp_path):
    from PIL import Image

    from waste_classifier.data.dataset import (
        WasteDataset,
        eval_transform,
    )
    from waste_classifier.data.split import ImageRecord

    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    record = ImageRecord(
        path=image_path,
        label="Plastic",
    )

    dataset = WasteDataset(
        records=[record],
        transform=eval_transform,
    )

    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 224, 224)
    assert label == 6    

def test_predict_invalid_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid image file"}    