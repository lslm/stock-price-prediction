from pydantic import BaseModel, Field


class RequestPrediction(BaseModel):
    ticker: str = Field(title="Ação", description="Ação que será usada para o treinamento")
    days_to_predict: int 

class ResponsePrediction(BaseModel):
    predictions: list[float] = Field(title="Lista de predições")