from model.ml_model import cardio_model


class PredictionService:

    @staticmethod
    def predict(patient):

        return cardio_model.predict(patient.model_dump())