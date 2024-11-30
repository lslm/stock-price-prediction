import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from src.factory.utils import get_dataset_by_ticker
from src.usecase.train_usecase.dtos import ResponseTrain
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MakeModel():
    def __init__(self, ticker: str, start_date: date, end_date: date) -> None:
        try:
            self.__dataset = self.__get_dataset_by_ticker(ticker, start_date, end_date)
        except Exception as ex:
            raise Exception(f"Não foi possível carregar os dados de treinamento: {ex.__annotations__}")

    def __get_dataset_by_ticker(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        dataset = get_dataset_by_ticker(ticker, start_date, end_date)
        return dataset

    def __mean_absolute_percentage_error(self, y_true, y_pred):
        """Calcula o erro percentual absoluto médio (MAPE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_indices = y_true != 0
        y_true = y_true[non_zero_indices]
        y_pred = y_pred[non_zero_indices]
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def make_model(self, lookback: int, hidden_size: int, epochs: int):
        data = self.__dataset
        future_steps = [1]  # Only predict the next step
        split_ratio = 0.8
        split_index = int(len(data) * split_ratio)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        scaler = MinMaxScaler()
        scaler.fit(train_data)
        data_scaled = scaler.transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

        # Create sequences for only the 'close' column
        sequences, targets = create_sequences(data_scaled, lookback, future_steps, target_column='close')

        print(f"Shape of sequences: {sequences.shape}")
        print(f"Shape of targets: {targets.shape}")

        sequence_split_index = split_index - lookback - max(future_steps) + 1

        X_train = sequences[:sequence_split_index]
        y_train = targets[:sequence_split_index]
        X_test = sequences[sequence_split_index:]
        y_test = targets[sequence_split_index:]

        print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
        print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")

        # Model configuration
        input_size = X_train.shape[2]
        output_size = 1  # Predict only one column
        num_layers = 2

        model = LSTMModel(input_size, hidden_size, output_size, num_layers=num_layers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = model.to(device)

        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)

        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            test_loss = criterion(predictions, y_test_tensor)
            print(f"Test Loss: {test_loss.item():.4f}")

        predictions_np = predictions.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()

        close_index = data.columns.get_loc('close')
        predictions_real = inverse_transform(predictions_np, scaler, close_index)
        y_test_real = inverse_transform(y_test_np, scaler, close_index)

        mae = mean_absolute_error(y_test_real, predictions_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
        mape = self.__mean_absolute_percentage_error(y_test_real, predictions_real)

        # Save model and scalers
        bundle = {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "lookback": lookback,
            "future_steps": future_steps,
        }

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_DIR = os.path.join(BASE_DIR, "model")
        os.makedirs(MODEL_DIR, exist_ok=True)
        file_path = os.path.join(MODEL_DIR, "model_and_scaler.joblib")
        joblib.dump(bundle, file_path)

        response = ResponseTrain(
            mae=mae,
            rmse=rmse,
            mape=mape
        )

        return response


def inverse_transform(predictions, scaler, feature_index):
    n_samples, n_steps = predictions.shape
    n_features = scaler.scale_.shape[0]
    inv_predictions = np.zeros((n_samples, n_steps))
    for i in range(n_steps):
        full_data = np.zeros((n_samples, n_features))
        full_data[:, feature_index] = predictions[:, i]
        inv_data = scaler.inverse_transform(full_data)
        inv_predictions[:, i] = inv_data[:, feature_index]
    return inv_predictions

def create_sequences(data, lookback, future_steps, target_column='close'):
        sequences = []
        targets = []
        data_values = data.values
        target_idx = data.columns.get_loc(target_column)
        max_future_step = max(future_steps)
        for i in range(len(data_values) - lookback - max_future_step + 1):
            seq = data_values[i:i + lookback]
            target_indices = [i + lookback + step - 1 for step in future_steps]
            target = data_values[target_indices, target_idx]
            sequences.append(seq)
            targets.append(target)
        sequences = np.array(sequences)
        targets = np.array(targets)
        targets = targets.reshape(-1, len(future_steps))
        return sequences, targets

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
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
