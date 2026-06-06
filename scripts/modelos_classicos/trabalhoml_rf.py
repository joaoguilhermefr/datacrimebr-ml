# Pré processamentos sem stem

from pathlib import Path
import time
from joblib import dump

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support


filepath_result = Path("results" , "modelos_classicos")
filepath_result.mkdir(parents=True, exist_ok=True)


data_col_import = pd.read_csv('data/dados_limpos_sem_stem.csv', sep=None, engine='python')
data_col_import["text"] = data_col_import["text"].fillna("")

"""#Vetorização"""

# vetorização TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data_col_import["text"])

#X = dados.drop(columns=['CRIME'])
y = data_col_import['CRIME']

X_treino, X_teste, y_treino, y_teste = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Paramentros com melhores resultados na validação
rf_model = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=100, random_state=42)

# Treino do modelo
rf_model.fit(X_treino, y_treino)

dump(rf_model, filepath_result / "rf_model.joblib")
dump(tfidf, filepath_result / "rf_tfidf.joblib")


# Após definir hiperparâmetros finais, avaliação com conjunto de teste
start_time_inf = time.time()
y_pred = rf_model.predict(X_teste)
inf_time = time.time() - start_time_inf

accuracy = accuracy_score(y_teste, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(
	y_teste,
	y_pred,
	average="weighted",
	zero_division=0,
)
tamanho_teste = len(y_teste)

print("\n--- Avaliação Final com Conjunto de Teste ---")
print(f"Acurácia no Teste: {accuracy:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_teste, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, y_pred))
print(f"\nModelos salvos em: {filepath_result}")

arquivo_resultado = filepath_result / "resultados_teste_rf.txt"
with open(arquivo_resultado, "w", encoding="utf-8") as f:
	f.write("--- Avaliação Final com Conjunto de Teste (Random Forest) ---\n")
	f.write(f"Acurácia: {accuracy:.6f}\n")
	f.write(f"Precisão (weighted): {precision:.6f}\n")
	f.write(f"Recall (weighted): {recall:.6f}\n")
	f.write(f"F1-score (weighted): {f1_score:.6f}\n")
	f.write(f"Tempo de inferência total (s): {inf_time:.6f}\n")
	f.write(f"Tamanho do conjunto de teste: {tamanho_teste}\n\n")
	f.write("Relatório de Classificação:\n")
	f.write(classification_report(y_teste, y_pred))
	f.write("\nMatriz de Confusão:\n")
	f.write(f"{confusion_matrix(y_teste, y_pred)}\n")

print(f"Resultados do teste salvos em: {arquivo_resultado}")
