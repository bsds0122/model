from fastapi import APIRouter

from schemas.model import PatientRequest

from services.model import PredictionService

router = APIRouter(
    prefix="/model",
    tags=["Cardiovascular Prediction"],
)


@router.post("/predict")
def predict(patient: PatientRequest):

    return PredictionService.predict(patient)