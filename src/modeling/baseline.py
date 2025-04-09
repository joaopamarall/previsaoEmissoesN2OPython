import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Caminho base atualizado para os gráficos
GRAPHICS_DIR = Path("outputs") / "resultgrafics"

def gerar_baseline(df: pd.DataFrame, target_col: str = "emissao") -> tuple:
    """
    Gera uma predição baseline usando a média da emissão,
    cria gráficos mais informativos e retorna as métricas.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        target_col (str): Nome da coluna alvo.

    Returns:
        tuple: (y_real, y_pred, mse)
    """
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no DataFrame.")

    # Define níveis de emissão com base em 't' (toneladas)
    if "t" in df.columns and df["t"].nunique() >= 3:
        df["nivel_emissao_geral"] = pd.qcut(
            df["t"], q=3, labels=["Baixo", "Médio", "Alto"], duplicates="drop"
        )
        print("[INFO] Níveis de emissão classificados em Baixo, Médio e Alto.")
    else:
        df["nivel_emissao_geral"] = "Desconhecido"
        print("[WARN] Coluna 't' ausente ou com poucos valores distintos — classificação de nível indisponível.")

    # Baseline com a média
    media_emissao = df[target_col].mean()
    y_pred = [media_emissao] * len(df)
    y_real = df[target_col].tolist()
    mse = mean_squared_error(y_real, y_pred)
    print(f"[INFO] Média da emissão usada como baseline: {media_emissao:.2f}")
    print(f"[INFO] Erro quadrático médio (MSE) do baseline: {mse:.2f}")

    # Criação do diretório para gráficos de baseline
    baseline_dir = GRAPHICS_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Diretório de gráficos criado em: {baseline_dir.resolve()}")

   # Gráfico 1: Emissão média por tipo de gás
    colunas_gas = [col for col in df.columns if "gas_" in col.lower()]
    if colunas_gas:
        try:
            df_gas = df.copy()
            df_gas["gas"] = df_gas[colunas_gas].idxmax(axis=1).str.replace("gas_", "")
            media_emissao_gas = df_gas.groupby("gas")[target_col].mean().sort_values(ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=media_emissao_gas.values, y=media_emissao_gas.index, palette="Set2")
            plt.title("Emissão Média por Tipo de Gás")
            plt.xlabel("Emissão média (t)")
            plt.ylabel("Tipo de Gás")
            plt.tight_layout()
            file_path = baseline_dir / "emissao_media_por_gas.png"
            plt.savefig(file_path)
            plt.close()
            print(f"[INFO] Gráfico de emissão média por tipo de gás salvo em: {file_path}")
        except Exception as e:
            print(f"[ERROR] Erro ao gerar o gráfico de emissão por gás: {e}")
    else:
        print("[WARN] Nenhuma coluna com prefixo 'gas_' foi encontrada para gerar o gráfico por tipo de gás.")


    # Gráfico 2: Linha da emissão por atividade econômica
    atividades = [col for col in df.columns if "atividade_economica_" in col.lower()]
    if atividades and "ano" in df.columns:
        try:
            df_atividades = df.copy()
            df_atividades["atividade"] = df_atividades[atividades].idxmax(axis=1).str.replace("atividade_economica_", "")
            top_atividades = df_atividades["atividade"].value_counts().index[:5]
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=df_atividades[df_atividades["atividade"].isin(top_atividades)],
                x="ano", y=target_col, hue="atividade", palette="tab10"
            )
            plt.title("Emissão por Ano nas Principais Atividades Econômicas")
            plt.xlabel("Ano")
            plt.ylabel("Emissão (t)")
            plt.tight_layout()
            file_path = baseline_dir / "emissao_por_atividade.png"
            plt.savefig(file_path)
            plt.close()
            print(f"[INFO] Gráfico de emissão por atividade econômica salvo em: {file_path}")
        except Exception as e:
            print(f"[ERROR] Erro ao gerar o gráfico por atividade econômica: {e}")
    else:
        if not atividades:
            print("[WARN] Nenhuma coluna com prefixo 'atividade_economica_' foi encontrada.")
        if "ano" not in df.columns:
            print("[WARN] Coluna 'ano' não encontrada — necessário para o gráfico de linha por atividade.")

    # Gráfico 3: Matriz de correlação simples com 3 variáveis (3x3)
    try:
        df_encoded = df.copy()

        # Extrai a principal atividade econômica
        atividades = [col for col in df_encoded.columns if "atividade_economica_" in col.lower()]
        if atividades:
            df_encoded["atividade"] = df_encoded[atividades].idxmax(axis=1).str.replace("atividade_economica_", "")
        else:
            df_encoded["atividade"] = "Desconhecido"

        # Verifica se ano e emissao estão presentes
        colunas_correlacao = ["emissao", "ano", "atividade"]
        colunas_existentes = [col for col in colunas_correlacao if col in df_encoded.columns]

        # Aplica LabelEncoder nas colunas categóricas
        for col in colunas_existentes:
            if df_encoded[col].dtype == "object":
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        # Gera a matriz de correlação 3x3
        corr_matrix = df_encoded[colunas_existentes].corr()

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
        plt.title("Matriz de Correlação (Emissão x Ano x Atividade)")
        plt.tight_layout()
        file_path = baseline_dir / "matriz_correlacao_3x3.png"
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"[INFO] Matriz de correlação 3x3 salva em: {file_path}")
    except Exception as e:
        print(f"[ERROR] Erro ao gerar matriz de correlação 3x3: {e}")



    return y_real, y_pred, mse
