# 💳 Predição de Risco de Crédito (Give Me Some Credit)

Este projeto tem como objetivo construir um modelo de **análise de risco de crédito** para prever se uma pessoa terá dificuldades financeiras nos próximos dois anos.

O fluxo cobre **limpeza de dados, análise exploratória, entendimento das variáveis, modelagem, avaliação e criação de um sistema de predição**.

<sub>Fonte de dados: https://www.kaggle.com/competitions/GiveMeSomeCredit/data</sub>

---

# 📊 Objetivo

Prever a variável target:

```
SeriousDlqin2yrs
```

* `0` → Sem risco de inadimplência
* `1` → Com risco de inadimplência

---

# 📁 Descrição do Dataset

Principais variáveis:

| Variável                             | Descrição                 |
| ------------------------------------ | ------------------------- |
| RevolvingUtilizationOfUnsecuredLines | Uso do limite de crédito  |
| age                                  | Idade do cliente          |
| DebtRatio                            | Relação dívida/renda      |
| MonthlyIncome                        | Renda mensal              |
| NumberOfTimes90DaysLate              | Atrasos graves (90+ dias) |
| NumberOfTime60-89DaysPastDueNotWorse | Atrasos médios            |
| NumberOfTime30-59DaysPastDueNotWorse | Atrasos leves             |
| NumberOfDependents                   | Número de dependentes     |

---

# 🧹 Pré-processamento de Dados

## Tratamento de valores nulos

```python
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
```

---

## Remoção de valores inválidos

```python
df = df[df['age'] >= 18]
```

---

## Tratamento de outliers (capping)

```python
upper = df['DebtRatio'].quantile(0.95)
df['DebtRatio'] = df['DebtRatio'].clip(upper=upper)
```

---

# 📊 Análise Exploratória

Comparação entre classes:

```python
df.groupby('SeriousDlqin2yrs')[col].mean()
```

### 🔍 Principais insights:

* Maior uso de crédito → maior risco
* Mais atrasos → forte indicador de inadimplência
* Menor renda → maior probabilidade de risco
* Idade → clientes mais jovens tendem a maior risco

---

# ⚠️ Observação Importante

O dataset é **desbalanceado**:

* Classe 0 (sem risco) é maioria
* Classe 1 (risco) é minoria

Isso impacta diretamente os modelos.

---

# 🤖 Modelos Utilizados

---

## 1. Regressão Logística (baseline)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
```

### ✔ Vantagens:

* Simples e interpretável
* Ajuste de peso melhora detecção da classe minoritária

---

## 2. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')
```

### ✔ Vantagens:

* Captura relações não lineares
* Robusto a outliers

---

## 3. XGBoost (melhor desempenho)

```python
from xgboost import XGBClassifier

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb = XGBClassifier(scale_pos_weight=scale_pos_weight)
```

### ✔ Vantagens:

* Alta performance
* Muito usado em problemas reais de crédito

---

# 📏 Métricas de Avaliação

Principais métricas:

* **ROC AUC** → capacidade de separar classes
* **Recall (classe 1)** → detectar clientes de risco
* **Precision** → qualidade das previsões positivas

---

## ⚠️ Sobre Accuracy

A acurácia pode ser enganosa em datasets desbalanceados.

Exemplo:

* Prever sempre “sem risco” pode gerar ~93% de acurácia

---

# 📊 Resultados

| Métrica        | Valor                 |
| -------------- | --------------------- |
| ROC AUC        | ~0.79                 |
| Recall (risco) | melhorou de 2% → ~29% |
| Accuracy       | ~92%                  |

---

# 🎯 Ajuste de Threshold

Threshold padrão:

```
0.5
```

Ajuste utilizado:

```python
threshold = 0.3
```

---

## ⚖️ Trade-off

| Threshold | Recall | Precision |
| --------- | ------ | --------- |
| Menor     | ↑      | ↓         |
| Maior     | ↓      | ↑         |

---

# ⚙️ Função de Predição

```python
def prever_risco(model, input_dict, features, threshold=0.3):
    import pandas as pd
    
    df_input = pd.DataFrame([input_dict])
    df_input = df_input[features]
    
    prob = model.predict_proba(df_input)[0][1]
    classe = int(prob >= threshold)
    
    return prob, classe
```

---

# 🧪 Exemplo de Uso

```python
cliente = {
    'RevolvingUtilizationOfUnsecuredLines': 0.8,
    'age': 35,
    'DebtRatio': 1.5,
    'MonthlyIncome': 4000,
    'NumberOfOpenCreditLinesAndLoans': 5,
    'NumberOfTimes90DaysLate': 1,
    'NumberRealEstateLoansOrLines': 1,
    'NumberOfTime60-89DaysPastDueNotWorse': 0,
    'NumberOfTime30-59DaysPastDueNotWorse': 2,
    'NumberOfDependents': 2
}

prob, classe = prever_risco(xgb, cliente, features)
```

---

# ⚠️ Problemas Comuns Resolvidos

## 1. Diferença de colunas

```python
df_input = df_input[features]
```

---

## 2. Coluna indesejada (`Unnamed: 0`)

```python
df = df.drop('Unnamed: 0', axis=1)
```

---

## 3. Valores nulos na target

```python
df = df.dropna(subset=['SeriousDlqin2yrs'])
```

---

# 🔄 Re-treinamento com novos dados

```python
df_total = pd.concat([df_antigo, df_novo])
df_total = df_total.dropna(subset=['SeriousDlqin2yrs'])

X = df_total.drop('SeriousDlqin2yrs', axis=1)
y = df_total['SeriousDlqin2yrs']

model.fit(X, y)
```

---

# 🚀 Próximos Passos

* Ajuste de hiperparâmetros (GridSearch)
* Engenharia de features
* Deploy (API ou app web)
* Monitoramento do modelo

---

# 💡 Principais Aprendizados

* Dados desbalanceados exigem tratamento especial
* Acurácia não é suficiente
* Recall é essencial em risco de crédito
* Consistência de features é crítica
* Ajustar threshold melhora muito o modelo

---

# 📌 Conclusão

Este projeto demonstra um pipeline completo de análise de dados e machine learning aplicado a risco de crédito, com práticas próximas ao mercado.

---

# 👨‍💻 Autor

Gabriel Bifon
SAP Security Analyst & Data Enthusiast
