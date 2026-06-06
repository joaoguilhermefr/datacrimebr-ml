#Pré processamento com stem

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


data_col_import = pd.read_csv('data/dados_limpos_com_stem.csv', sep=None, engine='python')
data_col_import["text"] = data_col_import["text"].fillna("")

"""#Vetorização"""

from sklearn.feature_extraction.text import TfidfVectorizer

y = data_col_import['CRIME']

X_treino, X_teste, y_treino, y_teste = train_test_split(data_col_import["text"], y, test_size=0.2, random_state=42, stratify=y)

# vetorização TF-IDF
tfidf = TfidfVectorizer()
X_treino = tfidf.fit_transform(X_treino)
X_teste = tfidf.transform(X_teste)

svm = SVC(random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  
}

#5-fold
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid.fit(X_treino, y_treino)

melhor_indice = grid.best_index_
acuracia_treino_cv = grid.cv_results_['mean_train_score'][melhor_indice]

print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Acurácia média de TREINO (Validação Cruzada): {acuracia_treino_cv:.4f}")
print(f"Acurácia média de VALIDAÇÃO (Validação Cruzada): {grid.best_score_:.4f}")
