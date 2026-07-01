from pydantic import BaseModel


class PatientRequest(BaseModel):

    sex: int
    age: int
    cigsPerDay: float
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float