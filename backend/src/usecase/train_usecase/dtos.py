from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import Literal
from dateutil.relativedelta import relativedelta


class ResponseTrain(BaseModel):
    """Modelo de resposta da rota api/train com as métricas no modelo treinado
    """
    rmse: float
    mae: float
    mape: float

class RequestTrain(BaseModel):
    """Modelo de request da rota api/train com as variáveis necessárias para o treinamento
    """
    ticker: str = Field(title="Ação", description="Ação que será usada para o treinamento")
    start_date: date = Field(title="Data Início", description="Data de Início dos dados da Ação", default=(date.today() - relativedelta(years=1)))
    end_date: date = Field(title="Data Fim", description="Data de Fim dos dados da Ação", default=date.today())
    lookback: int = Field(title="LookBack", description="Quantidade de dados que serão utilizados para prever o próximmo passo", ge=15, le=365, default=15)
    epoch: int = Field(title="Epochs", description="Quantidade de Epochs utilizadas para o treinamento", ge=10, le=500, default=500)
    hidden_size: int = Field(title="Hidden Size", description="Fefine a dimensionalidade do vetor de estado oculto (o vetor que mantém a informação relevante da sequência até o ponto atual)", default=50)
