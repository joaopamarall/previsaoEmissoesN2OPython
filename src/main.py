import os
import shutil
import joblib
import numpy as np
from src.config import CSV_FILE, RESULTS_FILE, MODEL_FILE, OUTPUT_DIR, MODEL_DIR, GRAPHICS_DIR
from src.preprocessing.limpeza import carregar_dados_filtrados
from src.modeling.treino import treinar_modelo_random_forest
from src.modeling.baseline import gerar_baseline
from src.visualization.graficos import gerar_graficos

def limpar_outputs():
    """
    Limpa a pasta de saída (outputs/) e recria a estrutura de subpastas.
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # Recria estrutura de diretórios
    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_FILE.parent, exist_ok=True)

def main():
    # 0. Limpeza da pasta de saída
    limpar_outputs()

    # 1. Carregamento e pré-processamento dos dados
    print(" Carregando e processando dados...")
    df = carregar_dados_filtrados(CSV_FILE)

    if df.empty:
        raise ValueError(" O DataFrame está vazio após o pré-processamento.")

    print(f" Dados carregados com {len(df)} registros.")

    # 2. Separação das features e variável alvo
    X = df.drop(columns=["emissao"])
    y = df["emissao"]

    # 3. Treinamento do modelo Random Forest
    print("\n Treinando modelo Random Forest...")
    modelo, y_pred_rf, mse_rf, r2_rf, importancias = treinar_modelo_random_forest(X, y)

    # 4. Geração da baseline
    print("\n Gerando baseline...")
    _, _, mse_base = gerar_baseline(df)

    # 5. Comparação de desempenho
    ganho = ((mse_base - mse_rf) / mse_base) * 100

    # 6. Impressão de métricas
    print("\n Resultados:")
    print(f"   • MSE da baseline: {mse_base:.2f}")
    print(f"   • MSE do modelo  : {mse_rf:.2f}")
    print(f"   • R² do modelo   : {r2_rf:.4f}")
    print(f"   • Ganho sobre baseline: {ganho:.2f}%")

    # 7. Salvando as métricas no arquivo de resultados
    with open(RESULTS_FILE, "w") as f:
        f.write(f"Baseline MSE: {mse_base:.2f}\n")
        f.write(f"Random Forest MSE: {mse_rf:.2f}\n")
        f.write(f"Random Forest R²: {r2_rf:.4f}\n")
        f.write(f"Ganho sobre o baseline: {ganho:.2f}%\n")

    # 8. Salvando o modelo treinado
    joblib.dump(modelo, MODEL_FILE)
    print(f"\n Modelo salvo em: {MODEL_FILE}")

    # 9. Geração dos gráficos com visualização aprimorada
    print("\n Gerando gráficos finais...")
    gerar_graficos(df, modelo, X.columns, importancias)
    print("Gráficos salvos com sucesso.")

if __name__ == "__main__":
    main()
