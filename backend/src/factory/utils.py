import pandas as pd
import yfinance as yf
from datetime import date

def get_dataset_by_ticker(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Busca os dados conforme ação e datas estipuladas

    Args:
        ticker (str): Ação
        start_date (date): data de início do dados a serem retornados
        end_date (date): data de fim do dados a serem retornados

    Returns:
        data (pd.DataFrame): Dataset com os dados da Ação inforamda
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data is None:
            raise Exception("Não há dados")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
        data.rename(columns={'Adj Close': 'adj_close', 'Close': 'close', 'High': 'high',
                     'Low': 'low', 'Open': 'open', 'Volume': 'volume'}, inplace=True)
        return data
    except Exception as ex:
        raise Exception(f"Não foi possível buscar os dados referente à ação {ticker}: {str(ex)}")