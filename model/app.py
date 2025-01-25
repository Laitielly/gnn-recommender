from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import warnings
from model.dataloader import dataset
from model.model import Model
from model.parse import config

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.")
from prometheus_fastapi_instrumentator import Instrumentator


class RecommendationModel:
    def __init__(self):
        self.device = config.device
        self.model = Model(dataset).to(self.device)
        self.model.load_state_dict(
            torch.load('model/model.pth', map_location=self.device, weights_only=True)
        )
        self.model.computer()

    def predict_for_user(self, user_id: int):
        return self.model.predict_user(user_id)

recommendation_model = RecommendationModel()

app = FastAPI()
Instrumentator().instrument(app).expose(app)

class RecommendationRequest(BaseModel):
    user_id: int

class RecommendationResponse(BaseModel):
    recommendations: List[int]

@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    user_id = request.user_id

    try:
        recommendations = recommendation_model.predict_for_user(user_id)
        return RecommendationResponse(recommendations=recommendations.tolist()[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
