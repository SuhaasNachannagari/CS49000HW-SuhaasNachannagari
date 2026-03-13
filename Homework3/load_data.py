from pathlib import Path
import json
import numpy as np
from datasets import load_dataset

SAMPLE_SIZE_PER_SPLIT = 7500
SEED = 51

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSONL = BASE_DIR / "data.jsonl"

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

def load_multinli_dev():
    ds = load_dataset("glue", "mnli")
    return ds["validation_matched"], ds["validation_mismatched"]


def standardize_split(dataset, source_split):
    rows = []

    for ex in dataset:
        if ex["label"] == -1:
            continue

        rows.append(
            {
                "pair_id": None,
                "premise": ex["premise"].strip(),
                "hypothesis": ex["hypothesis"].strip(),
                "label": LABEL_MAP[ex["label"]],
                "source_split": source_split,
            }
        )

    return rows


def sample_rows(rows, sample_size, rng):
    if len(rows) < sample_size:
        raise ValueError(f"Requested {sample_size} rows, but only found {len(rows)}.")

    indices = rng.choice(len(rows), size=sample_size, replace=False)
    return [rows[i] for i in indices]


def load_and_prepare_multinli():
    matched_ds, mismatched_ds = load_multinli_dev()

    matched_rows = standardize_split(matched_ds, "matched")
    mismatched_rows = standardize_split(mismatched_ds, "mismatched")

    rng = np.random.default_rng(SEED)

    matched_sample = sample_rows(matched_rows, SAMPLE_SIZE_PER_SPLIT, rng)
    mismatched_sample = sample_rows(mismatched_rows, SAMPLE_SIZE_PER_SPLIT, rng)

    all_rows = matched_sample + mismatched_sample
    for i, row in enumerate(all_rows):
        row["pair_id"] = i

    return matched_sample, mismatched_sample, all_rows


def save_jsonl(rows, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    matched_rows, mismatched_rows, all_rows = load_and_prepare_multinli()

    print("matched sampled examples:", len(matched_rows))
    print("mismatched sampled examples:", len(mismatched_rows))
    print("total sampled examples:", len(all_rows))
    print()
    print("first matched example:")
    print(matched_rows[0])
    print()
    print("first mismatched example:")
    print(mismatched_rows[0])

    save_jsonl(all_rows, OUTPUT_JSONL)
    print(f"\nSaved to {OUTPUT_JSONL}")