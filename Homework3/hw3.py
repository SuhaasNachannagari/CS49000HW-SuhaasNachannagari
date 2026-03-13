from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import stanza
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, __version__ as transformers_version

# ---------------------------
# Reproducibility
# ---------------------------

SEED = 51

def set_global_reproducibility(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


set_global_reproducibility(SEED)

# ---------------------------
# Paths / config
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_CANDIDATES = [
    BASE_DIR / "data.jsonl",
]

# Required output files
PERF_CSV = BASE_DIR / "perf.csv"
COMPLEX_CSV = BASE_DIR / "complex.csv"

# Helpful debug / report support files
COMPLEXITY_JSONL = BASE_DIR / "data_with_complexity.jsonl"
COMPLEXITY_SUMMARY_JSON = BASE_DIR / "complexity_summary.json"
BASELINE_RESULTS_JSON = BASE_DIR / "baseline_results.json"
PREDICTIONS_JSONL = BASE_DIR / "predictions.jsonl"
FAILURE_CASES_JSONL = BASE_DIR / "failure_cases.jsonl"
SCATTER_PLOT_DIR = BASE_DIR / "scatterplots"
RUN_METADATA_JSON = BASE_DIR / "run_metadata.json"

DEVICE = "cpu"
BATCH_SIZE = 16
MAX_EXAMPLES = None  # use None for exact file contents, or set to 15000

METRIC_NAMES = [
    "tree_depth",
    "subject_verb_distance",
    "dependent_clause_count",
    "t_unit_count",
    "dependent_clauses_per_t_unit",
]

MODEL_SPECS = [
    {
        "name": "roberta_large_mnli",
        "family": "encoder_only_transformer",
        "backend": "hf",
        "model_id": "FacebookAI/roberta-large-mnli",
    },
    {
        "name": "bart_large_mnli",
        "family": "encoder_decoder_transformer",
        "backend": "hf",
        "model_id": "facebook/bart-large-mnli",
    },
    {
        "name": "deberta_base_mnli",
        "family": "disentangled_attention_transformer",
        "backend": "hf",
        "model_id": "microsoft/deberta-base-mnli",
    },
    {
        "name": "electra_base_mnli",
        "family": "electra_discriminator_transformer",
        "backend": "hf",
        "model_id": "TehranNLP-org/electra-base-mnli",
    },
]

LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# ---------------------------
# Label handling
# ---------------------------

def normalize_label(label: str) -> str:
    label = label.strip().lower()

    mapping = {
        "entailment": "entailment",
        "entails": "entailment",
        "entail": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
        "contradictory": "contradiction",
        "contradict": "contradiction",
        "contradicts": "contradiction",
        "not_entailment": "contradiction",
        "contradiction.": "contradiction",
    }

    if label not in mapping:
        raise ValueError(f"Unrecognized label: {label}")
    return mapping[label]


def normalize_hf_label(label: str) -> str:
    return normalize_label(label)


# ---------------------------
# Data loading / saving
# ---------------------------

def resolve_input_path() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any input file in: {[str(p) for p in INPUT_CANDIDATES]}"
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be a list of examples.")

    raise ValueError(f"Unsupported file type: {path.suffix}")


def save_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------
# Part 1: complexity metrics
# ---------------------------

def ensure_stanza_resources() -> None:
    stanza.download(
        "en",
        processors="tokenize,pos,lemma,depparse,constituency",
        verbose=False,
    )


def build_parser():
    # Stanza does not use Apple's MPS, so on Mac this runs on CPU.
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma,depparse,constituency",
        use_gpu=False,
        verbose=False,
    )


def tree_depth(node) -> int:
    if not getattr(node, "children", None):
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def subject_verb_distance(words) -> int | None:
    root = None
    subject = None

    for word in words:
        if word.head == 0:
            root = word
            break

    if root is None:
        return None

    for word in words:
        if word.head == root.id and word.deprel in {
            "nsubj",
            "nsubj:pass",
            "csubj",
            "csubj:pass",
        }:
            subject = word
            break

    if subject is None:
        return None

    return abs(root.id - subject.id)


def is_finite_like_verb(word) -> bool:
    if word.upos in {"VERB", "AUX"}:
        return True

    feats = word.feats or ""
    if "VerbForm=Fin" in feats:
        return True
    if "Tense=" in feats:
        return True
    if "Mood=" in feats:
        return True

    return False


def dependent_clause_count(words) -> int:
    clause_rels = {
        "acl",
        "acl:relcl",
        "advcl",
        "ccomp",
        "xcomp",
        "csubj",
        "csubj:pass",
    }

    count = 0
    for word in words:
        if word.deprel in clause_rels:
            count += 1
    return count


def t_unit_count(words) -> int:
    if not words:
        return 0

    count = 1
    for word in words:
        if word.deprel == "conj" and is_finite_like_verb(word):
            count += 1
    return count


def dependent_clauses_per_t_unit(words) -> float | None:
    t_units = t_unit_count(words)
    if t_units == 0:
        return None
    return dependent_clause_count(words) / t_units


def analyze_sentence(sent) -> dict[str, Any]:
    words = sent.words
    return {
        "sentence": sent.text,
        "tree_depth": tree_depth(sent.constituency),
        "subject_verb_distance": subject_verb_distance(words),
        "dependent_clause_count": dependent_clause_count(words),
        "t_unit_count": t_unit_count(words),
        "dependent_clauses_per_t_unit": dependent_clauses_per_t_unit(words),
    }


def parse_single_sentence(nlp, text: str) -> dict[str, Any]:
    doc = nlp(text)
    if not doc.sentences:
        return {
            "sentence": text,
            "tree_depth": None,
            "subject_verb_distance": None,
            "dependent_clause_count": None,
            "t_unit_count": None,
            "dependent_clauses_per_t_unit": None,
        }

    return analyze_sentence(doc.sentences[0])


def compute_complexity(rows: list[dict[str, Any]], nlp) -> list[dict[str, Any]]:
    enriched = []

    for row in tqdm(rows, desc="Computing complexity"):
        premise_metrics = parse_single_sentence(nlp, row["premise"])
        hypothesis_metrics = parse_single_sentence(nlp, row["hypothesis"])

        new_row = dict(row)
        new_row["premise_complexity"] = premise_metrics
        new_row["hypothesis_complexity"] = hypothesis_metrics
        enriched.append(new_row)

    return enriched


def summarize_complexity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_examples": len(rows),
        "by_perturbation_method": {},
    }

    methods = sorted({row["perturbation_method"] for row in rows})

    for method in methods:
        method_rows = [r for r in rows if r["perturbation_method"] == method]
        summary["by_perturbation_method"][method] = {
            "premise": {},
            "hypothesis": {},
        }

        for side in ["premise", "hypothesis"]:
            for metric in METRIC_NAMES:
                vals = []
                for row in method_rows:
                    val = row[f"{side}_complexity"][metric]
                    if val is not None:
                        vals.append(val)

                summary["by_perturbation_method"][method][side][metric] = {
                    "mean": mean(vals) if vals else None,
                    "count_non_null": len(vals),
                }

    return summary


def build_complex_csv_rows(rows_with_complexity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    methods = sorted({row["perturbation_method"] for row in rows_with_complexity})

    for method in methods:
        subset = [r for r in rows_with_complexity if r["perturbation_method"] == method]

        for metric in METRIC_NAMES:
            vals = []
            for row in subset:
                val = row["premise_complexity"][metric]
                if val is not None:
                    vals.append(val)

            out_rows.append(
                {
                    "perturbation method": method,
                    "metric type": metric,
                    "value": mean(vals) if vals else "",
                }
            )

    return out_rows


# ---------------------------
# Model prediction helpers
# ---------------------------

def accuracy(preds: list[str], golds: list[str]) -> float:
    correct = sum(p == g for p, g in zip(preds, golds))
    return correct / len(golds) if golds else 0.0


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

    id2label = model.config.id2label
    normalized_id2label = {
        int(idx): normalize_hf_label(raw_label)
        for idx, raw_label in id2label.items()
    }
    return tokenizer, model, normalized_id2label


def predict_with_loaded_model(
    tokenizer,
    model,
    normalized_id2label: dict[int, str],
    rows: list[dict[str, Any]],
    batch_size: int = 16,
) -> list[str]:
    preds: list[str] = []

    for start in tqdm(range(0, len(rows), batch_size), desc="Predicting"):
        batch = rows[start:start + batch_size]
        premises = [r["premise"] for r in batch]
        hypotheses = [r["hypothesis"] for r in batch]

        enc = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            pred_ids = logits.argmax(dim=-1).tolist()

        preds.extend(normalized_id2label[i] for i in pred_ids)

    return preds


# ---------------------------
# Evaluation
# ---------------------------

def evaluate_models(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = [r for r in rows if r["perturbation_method"] == "original"]
    methods = sorted({r["perturbation_method"] for r in rows})

    baseline_results: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    method_to_rows: dict[str, list[dict[str, Any]]] = {
        method: [r for r in rows if r["perturbation_method"] == method]
        for method in methods
    }

    for spec in MODEL_SPECS:
        print(f"\nEvaluating: {spec['name']} ({spec['family']})")

        try:
            tokenizer, model, normalized_id2label = load_model_and_tokenizer(spec["model_id"])

            baseline_golds = [normalize_label(r["label"]) for r in baseline_rows]
            baseline_preds = predict_with_loaded_model(
                tokenizer,
                model,
                normalized_id2label,
                baseline_rows,
                batch_size=BATCH_SIZE,
            )
            baseline_acc = accuracy(baseline_preds, baseline_golds)

            baseline_results.append(
                {
                    "model_name": spec["name"],
                    "architecture_family": spec["family"],
                    "backend": spec["backend"],
                    "checkpoint": spec["model_id"],
                    "baseline_accuracy": baseline_acc,
                    "n_examples": len(baseline_rows),
                    "status": "ok",
                }
            )

            for method in methods:
                method_rows = method_to_rows[method]
                method_golds = [normalize_label(r["label"]) for r in method_rows]

                if method == "original":
                    method_preds = baseline_preds
                    method_acc = baseline_acc
                else:
                    method_preds = predict_with_loaded_model(
                        tokenizer,
                        model,
                        normalized_id2label,
                        method_rows,
                        batch_size=BATCH_SIZE,
                    )
                    method_acc = accuracy(method_preds, method_golds)

                perf_rows.append(
                    {
                        "model": spec["name"],
                        "perturbation method": method,
                        "performance": method_acc,
                    }
                )

                for row, pred in zip(method_rows, method_preds):
                    gold = normalize_label(row["label"])
                    prediction_rows.append(
                        {
                            "pair_id": row["pair_id"],
                            "source_split": row["source_split"],
                            "perturbation_method": row["perturbation_method"],
                            "label": gold,
                            "prediction": pred,
                            "correct": int(pred == gold),
                            "model": spec["name"],
                            "architecture_family": spec["family"],
                            "premise": row["premise"],
                            "hypothesis": row["hypothesis"],
                        }
                    )

            del model
            if DEVICE == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        except Exception as e:
            baseline_results.append(
                {
                    "model_name": spec["name"],
                    "architecture_family": spec["family"],
                    "backend": spec["backend"],
                    "checkpoint": spec["model_id"],
                    "baseline_accuracy": None,
                    "n_examples": len(baseline_rows),
                    "status": "error",
                    "error": str(e),
                }
            )

    return baseline_results, perf_rows, prediction_rows


# ---------------------------
# Error analysis helpers
# ---------------------------

def build_failure_cases(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}

    for row in prediction_rows:
        key = (row["model"], row["pair_id"], row["source_split"])
        indexed.setdefault(key, {})
        indexed[key][row["perturbation_method"]] = row

    failures: list[dict[str, Any]] = []

    for (model_name, pair_id, source_split), method_map in indexed.items():
        original_row = method_map.get("original")
        if original_row is None or original_row["correct"] != 1:
            continue

        for method, pert_row in method_map.items():
            if method == "original":
                continue
            if pert_row["correct"] == 0:
                failures.append(
                    {
                        "model": model_name,
                        "pair_id": pair_id,
                        "source_split": source_split,
                        "perturbation_method": method,
                        "gold_label": original_row["label"],
                        "original_prediction": original_row["prediction"],
                        "perturbed_prediction": pert_row["prediction"],
                        "original_premise": original_row["premise"],
                        "perturbed_premise": pert_row["premise"],
                        "hypothesis": pert_row["hypothesis"],
                    }
                )

    return failures


# ---------------------------
# Scatter plots
# ---------------------------

def pretty_model_name(model_name: str) -> str:
    mapping = {
        "roberta_large_mnli": "RoBERTa-large-MNLI",
        "bart_large_mnli": "BART-large-MNLI",
        "deberta_base_mnli": "DeBERTa-base-MNLI",
        "electra_base_mnli": "ELECTRA-base-MNLI",
    }
    return mapping.get(model_name, model_name.replace("_", " ").title())


def method_color(method: str) -> str:
    mapping = {
        "original": "tab:blue",
        "adjective_relative_clause": "tab:orange",
        "irrelevant_relative_clause": "tab:green",
        "entity_linked_appositive": "tab:red",
    }
    return mapping.get(method, "tab:gray")


def pretty_metric_name(metric: str) -> str:
    mapping = {
        "tree_depth": "Tree Depth",
        "subject_verb_distance": "Subject-Verb Distance",
        "dependent_clause_count": "Dependent Clause Count",
        "t_unit_count": "T-Unit Count",
        "dependent_clauses_per_t_unit": "Dependent Clauses per T-Unit",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def pretty_method_name(method: str) -> str:
    mapping = {
        "original": "Original",
        "adjective_relative_clause": "Adjective Relative Clause",
        "irrelevant_relative_clause": "Irrelevant Relative Clause",
        "entity_linked_appositive": "Entity-Linked Appositive",
    }
    return mapping.get(method, method.replace("_", " ").title())


def make_scatterplots(
    rows_with_complexity: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({r["perturbation_method"] for r in rows_with_complexity})
    splits = sorted({r["source_split"] for r in rows_with_complexity})
    models = sorted({r["model"] for r in prediction_rows})

    complexity_summary: dict[tuple[str, str, str], float] = {}
    for method in methods:
        for split in splits:
            subset = [
                r for r in rows_with_complexity
                if r["perturbation_method"] == method and r["source_split"] == split
            ]
            if not subset:
                continue

            for metric in METRIC_NAMES:
                vals = [
                    r["premise_complexity"][metric]
                    for r in subset
                    if r["premise_complexity"][metric] is not None
                ]
                if vals:
                    complexity_summary[(method, split, metric)] = mean(vals)

    error_summary: dict[tuple[str, str, str], float] = {}
    for model in models:
        for method in methods:
            for split in splits:
                subset = [
                    r for r in prediction_rows
                    if r["model"] == model
                    and r["perturbation_method"] == method
                    and r["source_split"] == split
                ]
                if not subset:
                    continue

                error_rate = 1.0 - (sum(r["correct"] for r in subset) / len(subset))
                error_summary[(model, method, split)] = error_rate

    for model in models:
        for metric in METRIC_NAMES:
            xs = []
            ys = []

            for method in methods:
                for split in splits:
                    x = complexity_summary.get((method, split, metric))
                    y = error_summary.get((model, method, split))

                    if x is None or y is None:
                        continue

                    xs.append(x)
                    ys.append(y)

            if not xs:
                continue

            plt.figure(figsize=(14, 5))
            seen_labels = set()

            for method in methods:
                for split in splits:
                    x = complexity_summary.get((method, split, metric))
                    y = error_summary.get((model, method, split))

                    if x is None or y is None:
                        continue

                    color = method_color(method)

                    if split == "matched":
                        facecolors = color
                        edgecolors = color
                        marker_label = f"{pretty_method_name(method)} (matched)"
                    else:
                        facecolors = "none"
                        edgecolors = color
                        marker_label = f"{pretty_method_name(method)} (mismatched)"

                    label = marker_label if marker_label not in seen_labels else None
                    if label is not None:
                        seen_labels.add(marker_label)

                    plt.scatter(
                        x,
                        y,
                        s=35,
                        facecolors=facecolors,
                        edgecolors=edgecolors,
                        linewidths=1.5,
                        alpha=0.9,
                        label=label,
                    )

            plt.xlabel(f"Average Premise {pretty_metric_name(metric)}")
            plt.ylabel("Model Error Rate")
            plt.title(f"{pretty_model_name(model)}: {pretty_metric_name(metric)} vs. Error Rate")
            plt.ylim(0, max(0.05, max(ys) + 0.02) if ys else 0.05)
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.08, 1),
                borderaxespad=0,
                fontsize=8,
                frameon=True,
            )
            plt.tight_layout(rect=[0, 0, 0.8, 1])

            out_path = output_dir / f"{model}_{metric}_aggregate.png"
            plt.savefig(out_path, dpi=200)
            plt.close()


# ---------------------------
# Metadata
# ---------------------------

def save_run_metadata(input_path: Path, n_rows: int) -> None:
    metadata = {
        "seed": SEED,
        "device": DEVICE,
        "input_file": str(input_path),
        "n_rows": n_rows,
        "batch_size": BATCH_SIZE,
        "max_examples": MAX_EXAMPLES,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "stanza_version": stanza.__version__,
        "model_specs": MODEL_SPECS,
    }

    with open(RUN_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------
# Main
# ---------------------------

def main():
    input_path = resolve_input_path()
    rows = load_rows(input_path)

    if MAX_EXAMPLES is not None:
        rows = rows[:MAX_EXAMPLES]

    save_run_metadata(input_path, len(rows))

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    print("\nDownloading/loading Stanza resources...")
    ensure_stanza_resources()

    print("\nBuilding Stanza parser...")
    nlp = build_parser()

    print("\nComputing complexity metrics...")
    rows_with_complexity = compute_complexity(rows, nlp)
    save_jsonl(rows_with_complexity, COMPLEXITY_JSONL)

    complexity_summary = summarize_complexity(rows_with_complexity)
    with open(COMPLEXITY_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(complexity_summary, f, indent=2)

    complex_csv_rows = build_complex_csv_rows(rows_with_complexity)
    save_csv(
        complex_csv_rows,
        COMPLEX_CSV,
        fieldnames=["perturbation method", "metric type", "value"],
    )

    print(f"Saved complexity-enriched data to: {COMPLEXITY_JSONL}")
    print(f"Saved complexity summary to: {COMPLEXITY_SUMMARY_JSON}")
    print(f"Saved required CSV to: {COMPLEX_CSV}")
    print(f"Saved run metadata to: {RUN_METADATA_JSON}")

    print("\nEvaluating models...")
    baseline_results, perf_rows, prediction_rows = evaluate_models(rows_with_complexity)

    with open(BASELINE_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    save_csv(
        perf_rows,
        PERF_CSV,
        fieldnames=["model", "perturbation method", "performance"],
    )
    save_jsonl(prediction_rows, PREDICTIONS_JSONL)

    failure_cases = build_failure_cases(prediction_rows)
    save_jsonl(failure_cases, FAILURE_CASES_JSONL)

    print(f"Saved baseline results to: {BASELINE_RESULTS_JSON}")
    print(f"Saved required CSV to: {PERF_CSV}")
    print(f"Saved per-example predictions to: {PREDICTIONS_JSONL}")
    print(f"Saved failure cases to: {FAILURE_CASES_JSONL}")

    print("\nMaking scatter plots...")
    make_scatterplots(rows_with_complexity, prediction_rows, SCATTER_PLOT_DIR)
    print(f"Saved scatter plots to: {SCATTER_PLOT_DIR}")

    print("\nBaseline summary (ORIGINAL rows only):")
    for item in baseline_results:
        print(item)


if __name__ == "__main__":
    main()