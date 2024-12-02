from src.usecase.train_usecase.dtos import ResponseTrain, RequestTrain
from src.factory.train import MakeModel

class TrainModel():


    def __init__(self, request: RequestTrain):
            self.__request = request

    def train(self) -> ResponseTrain:
        """Realiza o treinamento do modelo

        Returns:
            ResponseTrain: Métricas do modelo treinado
        """
        make_model_instance = MakeModel(self.__request.ticker, self.__request.start_date, self.__request.end_date)
        ress: RequestTrain = make_model_instance.make_model(ticker=self.__request.ticker,
                                                            lookback=self.__request.lookback,
                                                            # future_steps=self.__request.future_steps,
                                                            hidden_size=self.__request.hidden_size,
                                                            epochs=self.__request.epoch)
        return ress
