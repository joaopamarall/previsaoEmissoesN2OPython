import os
from pathlib import Path

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Caminhos dos diretórios
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
GRAPHICS_DIR = OUTPUT_DIR / "graficos"
MODEL_DIR = OUTPUT_DIR / "modelos"

# Arquivos de entrada e saída
CSV_FILE = DATA_DIR / "base.csv"
RESULTS_FILE = OUTPUT_DIR / "resultados.txt"
MODEL_FILE = MODEL_DIR / "modelo_random_forest.pkl"  # Novo caminho para salvar o modelo

# Configurações do modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
