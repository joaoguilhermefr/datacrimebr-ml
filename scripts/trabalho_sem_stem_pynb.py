import pandas as pd
data = pd.read_csv('data/crime_tweets_in_portuguese.csv')
#data.head()

data_col_import = data.drop(["INSULT","IDENTITY_ATTACK",	"SEVERE_TOXICITY",	"THREAT", "PROFANITY",	"TOXICITY",	"POSITIVE",	"NEUTRAL",	"NEGATIVE"], axis=1)
#data_col_import.tail()


data_col_import = data_col_import[data_col_import['KEYWORD'] == 1]

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

# Salvar os dados limpos para uso nos modelos
data_col_import.to_csv('data/dados_limpos_sem_stem.csv', index=False)
print("Dados limpos salvos com sucesso em 'dados_limpos_sem_stem.csv'!")