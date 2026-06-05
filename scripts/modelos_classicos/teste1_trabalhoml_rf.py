# Pré processamentos sem stem

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score



data_col_import = pd.read_csv('data/dados_limpos_sem_stem.csv', sep=None, engine='python')
data_col_import["text"] = data_col_import["text"].fillna("")

"""#Vetorização"""

from sklearn.feature_extraction.text import TfidfVectorizer

# vetorização TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data_col_import["text"])

y = data_col_import['CRIME']

X_treino, X_teste, y_treino, y_teste = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)


rf_model = RandomForestClassifier(random_state=13)


# Ajuste de hiperparâmetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

print("\nBuscando os melhores parâmetros via Grid Search...")
grid = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid.fit(X_treino, y_treino)

best_rf = grid.best_estimator_
melhor_indice = grid.best_index_
acuracia_treino_cv = grid.cv_results_['mean_train_score'][melhor_indice]

print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Acurácia média de TREINO (Validação Cruzada): {acuracia_treino_cv:.4f}")
print(f"Acurácia média de VALIDAÇÃO (Validação Cruzada): {grid.best_score_:.4f}")
