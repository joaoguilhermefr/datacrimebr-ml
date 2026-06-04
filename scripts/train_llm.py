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
df = pd.read_csv('data/dados_limpos_sem_stem.csv', sep=None, engine='python')

df.columns = df.columns.str.strip().str.lower()

if 'crime' in df.columns:
    df.rename(columns={'crime': 'label'}, inplace=True)

df['text'] = df['text'].astype(str)
df = df.dropna(subset=['text', 'label'])

df['label'] = df['label'].astype(int)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

hg_dataset = DatasetDict({
    'train': Dataset.from_pandas(df_train),
    'test': Dataset.from_pandas(df_test)
})

print("Baixando/carregando BERTimbau...")
model_name = "neuralmind/bert-base-portuguese-cased"
#model_name = "neuralmind/bert-large-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizando os textos...")
tokenized_datasets = hg_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir="./results/bertimbau_base",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("\nIniciando fine-tuning do BERTimbau")
start_time_train = time.time()
trainer.train()
train_time = time.time() - start_time_train
print(f"\nTempo total de Treinamento: {train_time:.2f} segundos")

print("\nAvaliando o modelo na base de teste")
start_time_inf = time.time()
eval_results = trainer.evaluate()
inf_time = time.time() - start_time_inf
print(f"Tempo de Inferência (Teste): {inf_time:.4f} segundos")

print("\nResultados Finais BERTimbau:")
for key, value in eval_results.items():
    print(f"{key}: {value}")


trainer.save_model("./results/meu_bertimbau_base_crimes")
tokenizer.save_pretrained("./results/meu_bertimbau_base_crimes")
print("\nModelo treinado e salvo com sucesso")