from src.usecase.predict_usecase.dtos import RequestPrediction, ResponsePrediction
from src.factory.predict import MakePrediction


class PredictModel():
    """Classe responsável por realizar a predição
    """
    def __init__(self, request: RequestPrediction) -> None:
        self.__request = request
    
    def predict(self) -> ResponsePrediction:
        """Realiza a predição

        Returns:
            ResponsePrediction: Resultado da Predição
        """
        ress = MakePrediction(ticker=self.__request.ticker, days=self.__request.days_to_predict).predict()
        return ResponsePrediction(predictions=ress)