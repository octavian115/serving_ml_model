from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):
    predicted_category: str = Field(
        ...,
        description="The predicted insurance premium category",  
        example= "High"
    )
    confidence: float = Field(
        ...,
        description= "Model's confidence score for the predicted class (range: 0 to )",
        example = 0.8435
    )
    class_probablities: Dict[str,float] = Field(
        ...,
        description="Probablity Distribution across all possible classes",
        example={"Low": 0.01, "Medium": 0.15, "High": 0.84}
    )
    