# 🎯 Projeto: Previsão de Emissões de N₂O com Random Forest

Este projeto tem como objetivo prever emissões de óxido nitroso (N₂O) utilizando técnicas de aprendizado de máquina, com foco no modelo Random Forest. Os dados são tratados, analisados e visualizados para melhor compreensão das variáveis que mais influenciam nas emissões.

## ⚙️ Tecnologias Utilizadas

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn & Matplotlib
- Jupyter Notebook
- Joblib

## 🧠 Modelagem e Avaliação

- Pré-processamento de dados com imputação e normalização
- Treinamento do modelo com Random Forest
- Comparação com modelo baseline (média)
- Métricas de avaliação: MSE, RMSE, R²

## 📊 Visualizações

- Gráficos de importância das variáveis
- Análise de correlação
- Gráficos por tipo de gás e atividade econômica
- Comparação entre predições e valores reais

## 📁 Estrutura do Projeto

```bash
previsaoEmissoesN2OPython/
├── data/                   # Dados brutos
│   └── base.csv
├── src/                    # Código-fonte principal
│   ├── config.py
│   ├── main.py
│   ├── preprocessing/
│   │   └── limpeza.py
│   ├── modeling/
│   │   ├── baseline.py
│   │   └── treino.py
│   └── visualization/
│       └── graficos.py
├── outputs/                # Resultados e gráficos gerados
│   ├── modelos/
│   │   └── modelo_random_forest.pkl
│   ├── graficos/
│   ├── resultgrafics/
│   │   └── baseline/
│   └── resultados.txt
├── tests/                  # Testes automatizados
│   └── test_pipeline.py
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

## 🧪 Como Executar

1. Clone o repositório
```bash
git clone https://github.com/joaopamarall/previsaoEmissoesN2OPython.git
cd previsaoEmissoesN2OPython
```

2. Instale as dependências
```bash
pip install -r requirements.txt
```

3. Execute o script principal
```bash
python src/main.py
```

## ✍️ Autores

- Daniel Rodrigues RA: 2022101144
- Yuri Richter Andolfato RA :2018102376
- João Pedro Amaral 2022100255
- Yan Percegona Weiss 2022101667
