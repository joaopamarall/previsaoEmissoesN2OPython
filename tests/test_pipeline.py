import numpy as np
from sklearn.metrics import mean_squared_error

def gerar_baseline(y_true):
    """
    Cria uma baseline simples usando a média dos valores reais como predição.
    Retorna o MSE (mean squared error) dessa baseline.
    """
    media = np.mean(y_true)
    previsoes = [media] * len(y_true)
    mse = mean_squared_error(y_true, previsoes)
    return mse
