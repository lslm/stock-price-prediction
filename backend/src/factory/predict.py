import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
from src.factory.utils import get_dataset_by_ticker
from src.factory.train import create_sequences, inverse_transform


class MakePrediction:
    def __init__(self, ticker: str, days: int):
        self.__days_to_predict = days  # Quantos dias você deseja prever
        self.__dataset_to_predict = self.__get_dataset_by_ticker(ticker)
        self.__bundle_model = self.__load_bundle_model_from_pkl(ticker=ticker)

    def __get_dataset_by_ticker(self, ticker: str) -> pd.DataFrame:
        start_date = (date.today() - relativedelta(years=4))  # 2 anos atrás
        dataset = get_dataset_by_ticker(ticker, start_date, date.today())
        return dataset

    def __load_bundle_model_from_pkl(self, ticker: str):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_DIR = os.path.join(BASE_DIR, "model")
        if not os.path.exists(f"{MODEL_DIR}/{ticker}.joblib"):
            raise Exception("Não foi possível carregar o modelo, verifique se ele foi gerado!")

        file_path = os.path.join(MODEL_DIR, f"{ticker}.joblib")
        bundle = joblib.load(file_path)
        return bundle

    def predict(self):
        # Carregar o modelo e o scaler
        bundle = self.__bundle_model
        model_state = bundle['model_state']
        scaler = bundle['scaler']
        input_size = bundle['input_size']
        hidden_size = bundle['hidden_size']
        future_steps = bundle['future_steps']

        # Escalar os dados completos
        dataset = self.__dataset_to_predict
        # Normalizar os dados
        data_scaled = scaler.transform(dataset)
        data_scaled = pd.DataFrame(data_scaled, columns=dataset.columns, index=dataset.index)

        # Criar sequências com base no lookback e nos dados recentes
        sequences, _ = create_sequences(data_scaled, 100, future_steps)

        # Preparar o modelo para previsão
        model = LSTMModel(input_size, hidden_size, 1)  # Prever apenas uma coluna
        model.load_state_dict(model_state)
        model.eval()

        # Previsão para os próximos dias
        predictions = []
        last_sequence = sequences[-1]  # Usando a última sequência para fazer a previsão
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Previsão para cada dia futuro
        with torch.no_grad():
            for _ in range(self.__days_to_predict):
                X_input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)  # Adiciona a dimensão batch
                X_input_tensor = X_input_tensor.to(device)

                # Fazendo a previsão
                predicted = model(X_input_tensor)
                predictions.append(predicted.item())

                # Atualizar a sequência para a próxima previsão (deslizar a janela)
                last_sequence = np.roll(last_sequence, -1, axis=0)  # Desliza a sequência
                last_sequence[-1, 0] = predicted.item()  # Atualiza com a previsão

        # Inverter as previsões para os valores reais
        predictions_real = inverse_transform(np.array(predictions).reshape(-1, 1), scaler, dataset.columns.get_loc('close')).round(4)

        return predictions_real


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
