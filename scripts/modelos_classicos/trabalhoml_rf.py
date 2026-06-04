# Pré processamentos sem stem

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


caminho_arq = 'crime_tweets_in_portuguese.csv'

data = pd.read_csv(caminho_arq)

data_col_import = data.drop(["INSULT","IDENTITY_ATTACK",	"SEVERE_TOXICITY",	"THREAT", "PROFANITY",	"TOXICITY",	"POSITIVE",	"NEUTRAL",	"NEGATIVE"], axis=1)
#data_col_import.tail()


data_col_import = data_col_import.drop(data_col_import.index[10001:61715])

#data_col_import.tail()

"""#Remoção de stop words"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')

nltk.download('rslp')
stop_words = set(stopwords.words('portuguese'))

import string

substituicoes = {
    "vc": "você",
    "ce": "você",
    "vcs": "vocês",
    "tds": "todos",
    "cmg": "comigo",
    "aki": "aqui",
    "dnv": "denovo",
    "n": "nao",
    "nn": "nao",
    "d": "de",
    "hj": "hoje",
    "mt": "muito",
    "mto":"muito",
    "brazil": "brasil",
    "br": "brasil",
    "ola": "oi",
    "cel": "celular",
    "q": "que",
    "oq": "que",
    "oque": "que",
    "qq": "que",
    "av":"avenida",
    "c":"com",
    "tb":"tambem",
    "tmb":"tambem",
    "quartafeira":"quarta",
    "quintafeira":"quinta",
    "sextafeira":"sexta",
    "tt":"twitter",
    "x":"twitter",
    "tt":"twitter",
    "tweet":"twitter",
    "dms":"dm",
    "pm":"dp",
    "c":"com"
}

import re
regex = {
   "risada": re. compile(r'k{2,}$')
}

texto = data_col_import['text'].astype(str)

def limpeza(text):

    tokens = word_tokenize(text)

    palavras_limpas = []

    for palavra in tokens:
        for nome, padrao in regex.items():
            if padrao.fullmatch(palavra):
                palavra = nome
                break
        palavra = substituicoes.get(
            palavra,
            palavra
        )
        if palavra not in stop_words:
            palavras_limpas.append(palavra)

    return palavras_limpas

data_col_import["text"] = texto.apply(lambda x: " ".join(limpeza(x)))
print(data_col_import["text"].tail())

"""#Vetorização"""

from sklearn.feature_extraction.text import TfidfVectorizer

# vetorização TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data_col_import["text"])

#X = dados.drop(columns=['CRIME'])
y = data_col_import['CRIME']

X_treino, X_teste, y_treino, y_teste = train_test_split(X_tfidf, y, test_size=0.2, random_state=13, stratify=y)

# Paramentros com melhores resultados na validação
rf_model = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=200,random_state=13)

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
