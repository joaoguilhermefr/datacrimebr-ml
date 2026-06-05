# Pré processamentos sem stem

from pathlib import Path
from joblib import dump

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

filepath_result = Path("results" , "modelos_classicos")


data_col_import = pd.read_csv('data/dados_limpos_sem_stem.csv', sep=None, engine='python')
data_col_import["text"] = data_col_import["text"].fillna("")

"""#Vetorização"""

# vetorização TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data_col_import["text"])


#X = dados.drop(columns=['CRIME'])
y = data_col_import['CRIME']

X_treino, X_teste, y_treino, y_teste = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)


# Parametros com melhor resultado na validação
svm = SVC(kernel='rbf', C=1, gamma='scale')

#treino do modelo
svm.fit(X_treino, y_treino)

dump(svm, filepath_result / "svm_model.joblib")
dump(tfidf, filepath_result / "svm_tfidf.joblib")

# Utilizado após ajustes utilizando a validação cruzada e obter um bom resultado
y_pred = svm.predict(X_teste)

print("\n--- Resultados da Avaliação ---")
print(f"Acurácia Geral: {accuracy_score(y_teste, y_pred):.2%}\n")
print("Relatório de Classificação:")
print(classification_report(y_teste, y_pred))
print(f"\nModelos salvos em: {filepath_result}")