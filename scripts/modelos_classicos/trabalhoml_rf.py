# Pré processamentos sem stem

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data_col_import = pd.read_csv('data/dados_limpos_sem_stem.csv', sep=None, engine='python')
data_col_import["text"] = data_col_import["text"].fillna("")

"""#Vetorização"""

from sklearn.feature_extraction.text import TfidfVectorizer

# vetorização TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data_col_import["text"])

#X = dados.drop(columns=['CRIME'])
y = data_col_import['CRIME']

X_treino, X_teste, y_treino, y_teste = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Paramentros com melhores resultados na validação
rf_model = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=200,random_state=42)

# Treino do modelo
rf_model.fit(X_treino, y_treino)


# Após definir hiperparâmetros finais, avaliação com conjunto de teste
y_pred = rf_model.predict(X_teste)

print("\n--- Avaliação Final com Conjunto de Teste ---")
print(f"Acurácia no Teste: {accuracy_score(y_teste, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_teste, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, y_pred))
