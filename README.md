# Stock Prediction

Essa é sistema criado para prever o valor de fechamento de ações em até 15 dias no futuro.

## Overview
Esse sistema é composto de três projetos:
- `experiments`: notebooks para avaliar a usabilidade da rede neural LSTM e testar parâmetros
- `backend`: projeto em FastAPI que disponibiliza duas rotas: `POST /api/predict` e `POST /api/train`. O projeto pode ser facilmente executado com o Docker ao executar `docker-compose up`
- `frontend`: projeto para iOS escrito com SwiftUI

O racional por tráz do projeto se encontra neste vídeo: https://youtu.be/9OYyXhkgpNY

## Documentação

### Executar aplicação:
```
docker-compose up
```

### Rotas
Treinar modelo para uma ação:
```
curl -H "Content-type: application/json" -d '{
    "ticker": "GOOG",
    "start_date": "2019-11-29",
    "end_date": "2024-11-29"
}' 'localhost:8000/api/train'
```

Prever valor do fechamento de uma ação para os próximos 15 dias:
```
curl -H "Content-type: application/json" -d '{
    "ticker": "GOOG",
    "days_to_predict": 15
}' 'localhost:8000/api/predict'
```
