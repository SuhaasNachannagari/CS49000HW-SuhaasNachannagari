from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
import stanza
import torch
from tqdm import tqdm

# ---------------------------
# Paths / config
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    BASE_DIR / "data.jsonl",
    BASE_DIR / "data.json",
]

# Final combined file for submission.
# This will contain original rows plus successful perturbations.
OUTPUT_JSONL = BASE_DIR / "data_with_perturbations.jsonl"

# Helpful debug file to inspect what changed.
PERTURBATION_DEBUG_JSONL = BASE_DIR / "perturbation_debug.jsonl"

# Set to None for the full dataset.
MAX_EXAMPLES = None

RANDOM_SEED = 51
random.seed(RANDOM_SEED)

IRRELEVANT_RELATIVE_CLAUSES = [
    "that was mentioned in an unrelated note",
    "that appeared in a separate discussion",
    "that was briefly referenced elsewhere",
    "that had no bearing on the main point",
    "that was included in an old report",
    "that was described in another context",
]

# ---------------------------
# I/O helpers
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


# ---------------------------
# Parsing helpers
# ---------------------------


def ensure_stanza_resources() -> None:
    stanza.download(
        "en",
        processors="tokenize,pos,lemma,depparse,ner",
        verbose=False,
    )


def build_pipeline():
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma,depparse,ner",
        use_gpu=torch.cuda.is_available(),
        verbose=False,
    )


def normalize_original_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "pair_id": row["pair_id"],
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row["label"],
                "source_split": row["source_split"],
                "perturbation_method": row.get("perturbation_method", "original"),
            }
        )
    return normalized


def first_sentence(doc, original_text: str):
    if doc.sentences:
        return doc.sentences[0]
    return None


def first_sentence_text(doc, original_text: str) -> str:
    sent = first_sentence(doc, original_text)
    return sent.text if sent is not None else original_text


def replace_first_whole_word(text: str, target: str, replacement: str) -> str:
    pattern = rf"\b{re.escape(target)}\b"
    return re.sub(pattern, replacement, text, count=1)


def splice_first_sentence(original_text: str, old_sent: str, new_sent: str) -> str:
    if old_sent in original_text:
        return original_text.replace(old_sent, new_sent, 1)
    return new_sent


def get_word_by_id(words, word_id):
    for w in words:
        if w.id == word_id:
            return w
    return None


# ---------------------------
# Wikidata helper
# ---------------------------


@lru_cache(maxsize=4096)
def wikidata_description(entity_text: str) -> str | None:
    try:
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": entity_text,
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        response = requests.get(search_url, params=params, timeout=4)
        response.raise_for_status()
        data = response.json()

        results = data.get("search", [])
        if not results:
            return None

        description = results[0].get("description")
        if not description:
            return None

        description = description.strip()
        if not description:
            return None

        article = "an" if description[0].lower() in "aeiou" else "a"
        return f"{article} {description}"
    except Exception:
        return None


# ---------------------------
# Perturbation helpers
# ---------------------------


def perturb_adjective_relative_clause(text: str, doc) -> str:
    sent = first_sentence(doc, text)
    sent_text = first_sentence_text(doc, text)

    if sent is None:
        return text.rstrip() + " This was notable."

    words = sent.words

    for word in words:
        if word.deprel != "amod":
            continue

        adj = word.text
        head = get_word_by_id(words, word.head)
        if head is None:
            continue

        if head.upos not in {"NOUN", "PROPN"}:
            continue

        old_phrase = f"{adj} {head.text}"
        if old_phrase in sent_text:
            new_phrase = f"{head.text}, that is {adj},"
            candidate = sent_text.replace(old_phrase, new_phrase, 1)
            candidate = re.sub(r",\s*,", ", ", candidate)
            if candidate != sent_text:
                return splice_first_sentence(text, sent_text, candidate)

    for word in words:
        if word.upos in {"NOUN", "PROPN"}:
            candidate = replace_first_whole_word(
                sent_text,
                word.text,
                f"{word.text}, that is notable,",
            )
            if candidate != sent_text:
                return splice_first_sentence(text, sent_text, candidate)

    return text.rstrip() + " This was notable."


def perturb_irrelevant_relative_clause(text: str, doc) -> str:
    sent = first_sentence(doc, text)
    sent_text = first_sentence_text(doc, text)

    if sent is None:
        return text.rstrip() + " This was mentioned in an unrelated note."

    words = sent.words

    for word in words:
        if word.upos in {"NOUN", "PROPN"}:
            clause = random.choice(IRRELEVANT_RELATIVE_CLAUSES)
            candidate = replace_first_whole_word(
                sent_text,
                word.text,
                f"{word.text}, {clause},",
            )
            if candidate != sent_text:
                return splice_first_sentence(text, sent_text, candidate)

    return text.rstrip() + " This was mentioned in an unrelated note."


def perturb_entity_linked_appositive(text: str, doc) -> str:
    sent_text = first_sentence_text(doc, text)

    for ent in getattr(doc, "ents", []):
        ent_text = ent.text.strip()
        if not ent_text or ent_text not in sent_text:
            continue

        desc = wikidata_description(ent_text)
        if desc is None:
            desc = "a named entity"

        candidate = sent_text.replace(ent_text, f"{ent_text}, {desc},", 1)
        if candidate != sent_text:
            return splice_first_sentence(text, sent_text, candidate)

    if doc.sentences:
        sent = doc.sentences[0]

        for word in sent.words:
            if word.upos == "PROPN":
                candidate = replace_first_whole_word(
                    sent_text,
                    word.text,
                    f"{word.text}, a named entity,",
                )
                if candidate != sent_text:
                    return splice_first_sentence(text, sent_text, candidate)

        for word in sent.words:
            if word.upos in {"NOUN", "PROPN"}:
                candidate = replace_first_whole_word(
                    sent_text,
                    word.text,
                    f"{word.text}, a named entity,",
                )
                if candidate != sent_text:
                    return splice_first_sentence(text, sent_text, candidate)

    return text.rstrip() + " This involved a named entity."


PERTURBATION_METHODS = {
    "adjective_relative_clause": perturb_adjective_relative_clause,
    "irrelevant_relative_clause": perturb_irrelevant_relative_clause,
    "entity_linked_appositive": perturb_entity_linked_appositive,
}


def generate_combined_rows(
    rows: list[dict[str, Any]], nlp
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="Generating perturbations"):
        original_row = dict(row)
        original_row["perturbation_method"] = "original"
        combined_rows.append(original_row)

        doc = nlp(row["premise"])

        for method_name, fn in PERTURBATION_METHODS.items():
            new_premise = fn(row["premise"], doc)

            perturbed_row = {
                "pair_id": row["pair_id"],
                "premise": new_premise,
                "hypothesis": row["hypothesis"],
                "label": row["label"],
                "source_split": row["source_split"],
                "perturbation_method": method_name,
            }
            combined_rows.append(perturbed_row)

            debug_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "source_split": row["source_split"],
                    "label": row["label"],
                    "perturbation_method": method_name,
                    "original_premise": row["premise"],
                    "perturbed_premise": new_premise,
                    "hypothesis": row["hypothesis"],
                }
            )

    return combined_rows, debug_rows


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    input_path = resolve_input_path()
    raw_rows = load_rows(input_path)
    rows = normalize_original_rows(raw_rows)

    source_rows = [
        row for row in rows
        if row.get("perturbation_method", "original") == "original"
    ]

    if MAX_EXAMPLES is not None:
        source_rows = source_rows[:MAX_EXAMPLES]

    print(f"Loaded {len(source_rows)} original rows from {input_path}")
    print(f"Output file: {OUTPUT_JSONL}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    ensure_stanza_resources()
    nlp = build_pipeline()

    combined_rows, debug_rows = generate_combined_rows(source_rows, nlp)

    save_jsonl(combined_rows, OUTPUT_JSONL)
    save_jsonl(debug_rows, PERTURBATION_DEBUG_JSONL)

    method_counts: dict[str, int] = {}
    for row in combined_rows:
        method = row["perturbation_method"]
        method_counts[method] = method_counts.get(method, 0) + 1

    print("\nSaved combined dataset.")
    print(f"Rows written: {len(combined_rows)}")
    for method_name, count in sorted(method_counts.items()):
        print(f"{method_name}: {count}")

    print(f"\nSaved combined data to: {OUTPUT_JSONL}")
    print(f"Saved perturbation debug data to: {PERTURBATION_DEBUG_JSONL}")


if __name__ == "__main__":
    main()