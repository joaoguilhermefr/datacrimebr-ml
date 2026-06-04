#Pré processamento com stem

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


caminho_arq = '/content/drive/MyDrive/ColabNotebooks/crime_tweets_in_portuguese.csv'

data = pd.read_csv(caminho_arq)

data.head()

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

"""#Dicionário e REGEX"""

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

    #tokenização:
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

"""#Stemming"""

def stemming(text):
    tokens = word_tokenize(text)
    stemmer = nltk.stem.RSLPStemmer()
    stemming = [stemmer.stem(word) for word in tokens]
    return stemming

# Aplicando a função linha por linha
data_col_import["text"] = data_col_import["text"].apply(lambda x: " ".join(stemming(x)))


# Exibir parte dos resultados
print(data_col_import["text"].head())
print(data_col_import.head())

"""#Vetorização"""

from sklearn.feature_extraction.text import TfidfVectorizer

# vetorização TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data_col_import["text"])

y = data_col_import['CRIME']

X_treino, X_teste, y_treino, y_teste = train_test_split(X_tfidf, y, test_size=0.2, random_state=13, stratify=y)


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
