from __future__ import annotations
import json
from pathlib import Path
import stanza
from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSONL = BASE_DIR / "multinli_dev_all.jsonl"

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


def load_and_prepare_multinli():
    matched_ds, mismatched_ds = load_multinli_dev()

    matched_rows = standardize_split(matched_ds, "matched")
    mismatched_rows = standardize_split(mismatched_ds, "mismatched")

    all_rows = matched_rows + mismatched_rows

    for i, row in enumerate(all_rows):
        row["pair_id"] = i

    return matched_rows, mismatched_rows, all_rows


def save_jsonl(rows, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser():
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
        if word.head == root.id and word.deprel in {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}:
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


def analyze_sentence(sent) -> dict:
    words = sent.words

    return {
        "sentence": sent.text,
        "tree_depth": tree_depth(sent.constituency),
        "subject_verb_distance": subject_verb_distance(words),
        "dependent_clause_count": dependent_clause_count(words),
        "t_unit_count": t_unit_count(words),
        "dependent_clauses_per_t_unit": dependent_clauses_per_t_unit(words),
    }


if __name__ == "__main__":
    matched_rows, mismatched_rows, all_rows = load_and_prepare_multinli()

    print("matched examples:", len(matched_rows))
    print("mismatched examples:", len(mismatched_rows))
    print("total examples:", len(all_rows))
    print()
    print("first matched example:")
    print(matched_rows[0])
    print()
    print("first mismatched example:")
    print(mismatched_rows[0])

    save_jsonl(all_rows, OUTPUT_JSONL)
    print(f"\nSaved to {OUTPUT_JSONL}")