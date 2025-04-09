from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.config import TEST_SIZE, RANDOM_STATE
import numpy as np

def treinar_modelo_random_forest(X, y):
    """
    Treina um modelo Random Forest e retorna o modelo treinado, previsões, métricas e importâncias das features.
    """

    # Usar apenas 30% dos dados para acelerar testes (remova isso para usar o dataset completo)
    X, _, y, _ = train_test_split(X, y, train_size=0.3, random_state=RANDOM_STATE)

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Inicializar o modelo com todos os núcleos
    modelo = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1  # Utiliza todos os núcleos disponíveis para paralelizar
    )

    # Treinar o modelo
    modelo.fit(X_train, y_train)

    # Prever com os dados de teste
    y_pred = modelo.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Importância das variáveis
    importancias = modelo.feature_importances_

    return modelo, y_pred, mse, r2, importancias
