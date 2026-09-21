from pydantic import BaseModel, Field, field_validator


CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class FeedbackRequest(BaseModel):
    image_name: str = Field(min_length=1)
    predicted_class: str
    actual_class: str

    @field_validator("predicted_class", "actual_class")
    @classmethod
    def validate_class(cls, value: str) -> str:
        if value not in CLASS_NAMES:
            raise ValueError(f"Unknown class: {value}")
        return value