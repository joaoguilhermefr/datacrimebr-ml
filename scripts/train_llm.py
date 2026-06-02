import pandas as pd
import time
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print(f"GPU disponível? {'Sim' if torch.cuda.is_available() else 'Não'}")

print("Carregando dados...")
df = pd.read_csv('data/dados_limpos_sem_stem.csv')
df = df.dropna(subset=['text', 'label']) 

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

hg_dataset = DatasetDict({
    'train': Dataset.from_pandas(df_train),
    'test': Dataset.from_pandas(df_test)
})

print("Baixando/carregando BERTimbau...")
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizando os textos...")
tokenized_datasets = hg_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./resultados_bertimbau",
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    num_train_epochs=3, 
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)


print("\n" + "="*40)
print("Iniciando fine-tuning do BERTimbau")
start_time_train = time.time()

trainer.train()

train_time = time.time() - start_time_train
print(f"\nTempo total de treinamento = {train_time:.2f} segundos")


print("\n" + "="*40)
print("Avaliando o modelo na base de teste")
start_time_inf = time.time()

eval_results = trainer.evaluate()

inf_time = time.time() - start_time_inf
print(f"Tempo de inferência (teste): {inf_time:.4f} segundos")

print("\nResultados finais BERTimbau")
for key, value in eval_results.items():
    print(f"{key}: {value}")


trainer.save_model("./meu_bertimbau_crimes")
tokenizer.save_pretrained("./meu_bertimbau_crimes")
print("\nModelo salvo em './meu_bertimbau_crimes'")