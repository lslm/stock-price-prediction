import logging
from usecase.train_usecase import RequestTrain, ResponseTrain, TrainModel
from usecase.predict_usecase import RequestPrediction, ResponsePrediction, PredictModel
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder
from api.core.dtos import ErrorModel
from fastapi import APIRouter
from http import HTTPStatus

# Configuração básica do logger
logging.basicConfig(
    level=logging.DEBUG, ## ALTERAR PARA INFO QUANDO FINALIZADO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Você pode adicionar um arquivo, se necessário
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=['Stock Price Prediction'])

@router.post("/predict")
async def make_prediction(request: RequestPrediction):
    try:
        res: ResponsePrediction = PredictModel(request=request).predict()
        return JSONResponse(content=jsonable_encoder(res), status_code=HTTPStatus.OK)
    except Exception as ex:
        ex.with_traceback
        error = ErrorModel.from_exception(ex)
        return JSONResponse(content=jsonable_encoder(error), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

@router.post("/train")
async def make_model(request: RequestTrain):
    """Realiza o treinamento do modelo
    """
    try:
        res: ResponseTrain = TrainModel(request).train()
        return JSONResponse(content=jsonable_encoder(res), status_code=HTTPStatus.OK)
    except Exception as ex:
        ex.with_traceback
        error = ErrorModel.from_exception(ex)
        return JSONResponse(content=jsonable_encoder(error), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)