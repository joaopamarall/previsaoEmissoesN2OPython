import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.config import GRAPHICS_DIR
import shutil

def gerar_graficos(df: pd.DataFrame, modelo, feature_names, importancias):
    """
    Gera gráfico de emissão real vs prevista por ano e salva no diretório definido.
    Limpa gráficos antigos antes de salvar os novos.
    """
    # Limpar diretório de gráficos
    if GRAPHICS_DIR.exists():
        shutil.rmtree(GRAPHICS_DIR)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

    # Garantir que há colunas necessárias
    if "ano" not in df.columns or "emissao" not in df.columns:
        print("Colunas 'ano' e 'emissao' são necessárias para o gráfico.")
        return

    # Limitar dados para amostragem, se necessário
    df = df.copy()
    df = df.sample(n=min(150, len(df)), random_state=42).reset_index(drop=True)

    # Prever emissões
    df["emissao_prevista"] = modelo.predict(df[feature_names])

    # Agrupar por ano
    df_ano = df.groupby("ano").agg({
        "emissao": "mean",
        "emissao_prevista": "mean"
    }).reset_index()

    # Gerar gráfico
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_ano, x="ano", y="emissao", label="Emissão Real", marker="o")
    sns.lineplot(data=df_ano, x="ano", y="emissao_prevista", label="Emissão Prevista", marker="o")

    plt.title("Emissão Real vs Prevista por Ano")
    plt.xlabel("Ano")
    plt.ylabel("Emissão")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHICS_DIR / "emissao_por_ano.png")
    plt.close()
