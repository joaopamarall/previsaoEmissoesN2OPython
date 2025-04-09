import pandas as pd

def carregar_dados_filtrados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)

    # Corrigir espaços nos nomes das colunas
    df.columns = df.columns.str.strip()

    # Filtrar apenas linhas com gás N2O ou CH4
    df = df[df["gas"].str.upper().str.contains("N2O") | df["gas"].str.upper().str.contains("CH4")]

    # Criar a coluna combinada de níveis de emissão antes de remover colunas
    niveis = [col for col in df.columns if col.lower().startswith("nivel_")]
    if len(niveis) >= 2:
        df["nivel_emissao_geral"] = df[niveis].astype(str).agg(" > ".join, axis=1)

    # Remover colunas com muitos valores nulos
    limite_nulos = 0.3
    limite_colunas = df.isnull().mean() < limite_nulos
    df = df.loc[:, limite_colunas]

    # Remover linhas com valores nulos
    df = df.dropna()

    # Separar variável alvo
    if "emissao" in df.columns:
        emissao = df["emissao"]
        df = df.drop(columns=["emissao"])
    else:
        emissao = None

    # Preservar 'gas' e 'atividade_economica' até o one-hot
    colunas_categoricas = [
        col for col in df.columns
        if df[col].dtype == "object"
    ]

    df = pd.get_dummies(df, columns=colunas_categoricas)

    # Reanexar a emissão
    if emissao is not None:
        df["emissao"] = emissao

        # Remover outliers
        q1 = df["emissao"].quantile(0.25)
        q3 = df["emissao"].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        df = df[(df["emissao"] >= limite_inferior) & (df["emissao"] <= limite_superior)]

    return df
