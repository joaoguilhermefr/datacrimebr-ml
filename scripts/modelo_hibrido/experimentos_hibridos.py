from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "dados_limpos_sem_stem.csv"
CLASSIC_RESULTS_DIR = BASE_DIR / "results" / "modelos_classicos"
HYBRID_RESULTS_DIR = BASE_DIR / "results" / "modelo_hibrido"
MAX_LENGTH = 128
THRESHOLD = 0.70
BATCH_SIZE = 16


EXPERIMENTS = {
    "rf_bert_base": {
        "layer1_name": "rf",
        "layer2_name": "bert_base",
        "layer2_path": BASE_DIR / "results" / "meu_bertimbau_base_crimes",
    },
    "rf_bert_large": {
        "layer1_name": "rf",
        "layer2_name": "bert_large",
        "layer2_path": BASE_DIR / "results" / "meu_bertimbau_large_crimes",
    },
    "svm_bert_base": {
        "layer1_name": "svm",
        "layer2_name": "bert_base",
        "layer2_path": BASE_DIR / "results" / "meu_bertimbau_base_crimes",
    },
    "svm_bert_large": {
        "layer1_name": "svm",
        "layer2_name": "bert_large",
        "layer2_path": BASE_DIR / "results" / "meu_bertimbau_large_crimes",
    },
}


def load_dataset():
    data = pd.read_csv(DATA_PATH, sep=None, engine="python")
    data.columns = data.columns.str.strip().str.lower()

    data = data.rename(columns={"crime": "label"})

    data = data.fillna({"text": ""})

    return data


def load_classic_artifacts(layer1_name):
    model_path = CLASSIC_RESULTS_DIR / f"{layer1_name}_model.joblib"
    tfidf_path = CLASSIC_RESULTS_DIR / f"{layer1_name}_tfidf.joblib"

    return load(model_path), load(tfidf_path)


def load_llm_artifacts(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _classic_confidence(model, X_test):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        return probabilities.max(axis=1)

    raise AttributeError("O modelo clássico precisa expor predict_proba.")


def predict_classic(model, X_test):
    predictions = np.asarray(model.predict(X_test))
    confidence = _classic_confidence(model, X_test)
    return predictions, confidence


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def predict_llm(tokenizer, model, device, texts):
    predictions: list[int] = []
    confidences: list[float] = []

    with torch.no_grad():
        for batch_texts in _batched(texts, BATCH_SIZE):
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            batch_predictions = probabilities.argmax(dim=-1)
            batch_confidences = probabilities.max(dim=-1).values

            predictions.extend(batch_predictions.detach().cpu().numpy().tolist())
            confidences.extend(batch_confidences.detach().cpu().numpy().tolist())

    return np.asarray(predictions), np.asarray(confidences)


def combine_predictions(classic_predictions, classic_confidence, llm_predictions, llm_confidence, threshold):
    use_llm = classic_confidence < threshold
    hybrid_predictions = np.where(use_llm, llm_predictions, classic_predictions)
    hybrid_confidence = np.where(use_llm, llm_confidence, classic_confidence)
    return hybrid_predictions, hybrid_confidence


def build_predictions_frame(texts, labels, classic_predictions, classic_confidence, llm_predictions, llm_confidence, threshold):
    hybrid_predictions, hybrid_confidence = combine_predictions(
        classic_predictions=classic_predictions,
        classic_confidence=classic_confidence,
        llm_predictions=llm_predictions,
        llm_confidence=llm_confidence,
        threshold=threshold,
    )

    return pd.DataFrame(
        {
            "text": texts,
            "label_real": labels,
            "predicao_camada1": classic_predictions,
            "confianca_camada1": classic_confidence,
            "predicao_camada2": llm_predictions,
            "confianca_camada2": llm_confidence,
            "predicao_hibrida": hybrid_predictions,
            "confianca_hibrida": hybrid_confidence,
            "roteado_para_llm": classic_confidence < threshold,
        }
    )


def build_metrics(y_true, y_pred):
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_score,
    }


def run_experiment(experiment_name, threshold):
    config = EXPERIMENTS[experiment_name]
    output_dir = HYBRID_RESULTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    data["text"] = data["text"].fillna("")

    _, test_texts, _, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    
    classic_model, tfidf = load_classic_artifacts(config["layer1_name"])

    X_test_tfidf = tfidf.transform(test_texts)

    tokenizer, llm_model, device = load_llm_artifacts(config["layer2_path"])

    test_classic_predictions, test_classic_confidence = predict_classic(classic_model, X_test_tfidf)
    test_llm_predictions, test_llm_confidence = predict_llm(tokenizer, llm_model, device, list(test_texts))
    test_predictions = build_predictions_frame(
        texts=test_texts,
        labels=y_test,
        classic_predictions=test_classic_predictions,
        classic_confidence=test_classic_confidence,
        llm_predictions=test_llm_predictions,
        llm_confidence=test_llm_confidence,
        threshold=threshold,
    )

    classic_predictions = test_predictions["predicao_camada1"].to_numpy()
    llm_predictions = test_predictions["predicao_camada2"].to_numpy()
    hybrid_predictions = test_predictions["predicao_hibrida"].to_numpy()
    classic_confidence = test_predictions["confianca_camada1"].to_numpy()
    llm_confidence = test_predictions["confianca_camada2"].to_numpy()
    hybrid_confidence = test_predictions["confianca_hibrida"].to_numpy()
    routed_to_llm = float(test_predictions["roteado_para_llm"].mean())

    classic_metrics = build_metrics(y_test, classic_predictions)
    llm_metrics = build_metrics(y_test, llm_predictions)
    hybrid_metrics = build_metrics(y_test, hybrid_predictions)

    report_path = output_dir / "relatorio.txt"
    

    with open(report_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"Experimento: {experiment_name}\n")
        file_handle.write(f"Camada 1: {config['layer1_name']}\n")
        file_handle.write(f"Camada 2: {config['layer2_name']}\n")
        file_handle.write(f"Threshold de roteamento: {threshold:.2f}\n")
        file_handle.write(f"Fração roteada para LLM: {routed_to_llm:.4f}\n\n")

        file_handle.write("--- Camada 1 ---\n")
        file_handle.write(f"Acurácia: {classic_metrics['accuracy']:.6f}\n")
        file_handle.write(f"Precisão ponderada: {classic_metrics['precision_weighted']:.6f}\n")
        file_handle.write(f"Recall ponderado: {classic_metrics['recall_weighted']:.6f}\n")
        file_handle.write(f"F1 ponderado: {classic_metrics['f1_weighted']:.6f}\n\n")

        file_handle.write("--- Camada 2 ---\n")
        file_handle.write(f"Acurácia: {llm_metrics['accuracy']:.6f}\n")
        file_handle.write(f"Precisão ponderada: {llm_metrics['precision_weighted']:.6f}\n")
        file_handle.write(f"Recall ponderado: {llm_metrics['recall_weighted']:.6f}\n")
        file_handle.write(f"F1 ponderado: {llm_metrics['f1_weighted']:.6f}\n\n")

        file_handle.write("--- Híbrido ---\n")
        file_handle.write(f"Acurácia: {hybrid_metrics['accuracy']:.6f}\n")
        file_handle.write(f"Precisão ponderada: {hybrid_metrics['precision_weighted']:.6f}\n")
        file_handle.write(f"Recall ponderado: {hybrid_metrics['recall_weighted']:.6f}\n")
        file_handle.write(f"F1 ponderado: {hybrid_metrics['f1_weighted']:.6f}\n\n")
        file_handle.write("Relatório de classificação do híbrido:\n")
        file_handle.write(classification_report(y_test, hybrid_predictions, zero_division=0))

    return {
        "experimento": experiment_name,
        "camada_1": config["layer1_name"],
        "camada_2": config["layer2_name"],
        "threshold": threshold,
        "fracao_llm": routed_to_llm,
        "classic_accuracy": classic_metrics["accuracy"],
        "llm_accuracy": llm_metrics["accuracy"],
        "hybrid_accuracy": hybrid_metrics["accuracy"],
        "classic_f1": classic_metrics["f1_weighted"],
        "llm_f1": llm_metrics["f1_weighted"],
        "hybrid_f1": hybrid_metrics["f1_weighted"],
        "report_path": str(report_path),
    }



HYBRID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

experiment_names = list(EXPERIMENTS.keys())
summaries = [run_experiment(experiment_name, THRESHOLD) for experiment_name in experiment_names]

summary_frame = pd.DataFrame(summaries)
summary_path = HYBRID_RESULTS_DIR / "resumo_experimentos.csv"
summary_frame.to_csv(summary_path, index=False, encoding="utf-8")

print(summary_frame.to_string(index=False))
print(f"\nResumo salvo em: {summary_path}")
