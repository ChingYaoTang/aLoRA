"""
Build the answerability training dataset from the human MTRAG sources.

The script combines the MTRAG Human and MTRAG-UN Human generation tasks,
maps answerable examples to Y and unanswerable-style examples to N, filters
examples that would be too long for the current training prompt, and writes
train/val JSONL files.
Raw source files are only read; they are never modified.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an answerability detection assistant. "
    "Given a conversation and supporting documents, output exactly Y if the final user "
    "question can be answered from the documents, or N if it cannot."
)

# The descriptive invocation is the longest variant in src/train.py. Filtering
# with it keeps every training variant under the same token limit.
DESCRIPTIVE_INVOC_SEQ = (
    "<|im_start|>determine if the question is answerable given the documents\n"
)

DEFAULT_SOURCES = [
    Path("raw_data/mtrag-human/generation_tasks/reference.jsonl"),
    Path("raw_data/mtragun-human/generation_tasks/reference.jsonl"),
]

NEGATIVE_LABELS = {"UNANSWERABLE", "PARTIAL", "UNDERSPECIFIED"}


# Formatting -----------------------------------------------------------------

def format_contexts(contexts: list) -> str:
    """Format supporting documents in the same readable layout as training."""
    if not contexts:
        return "(no documents provided)"

    parts = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title", "")
        text = ctx.get("text", "").strip()
        header = f"[Document {i}]" + (f" {title}" if title else "")
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def format_conversation(turns: list) -> str:
    """Format the conversation history from raw generation task turns."""
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "unknown").capitalize()
        text = turn.get("text", "").strip()
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_input_text(record: dict) -> str:
    """Build the user-side text consumed by src/train.py."""
    ctx_str = format_contexts(record.get("contexts", []))
    conv_str = format_conversation(record.get("input", []))
    return f"Documents:\n{ctx_str}\n\nConversation:\n{conv_str}"


def build_training_text(sample: dict, tokenizer) -> str:
    """Build the longest full training string used for token-length filtering."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["input_text"]},
    ]
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return f"{prefix}{DESCRIPTIVE_INVOC_SEQ}{sample['label']}<|im_end|>\n"


# Loading and filtering -------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into memory for this small preprocessing job."""
    records = []
    with path.open() as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_raw_answerability(record: dict) -> str:
    """Return the original answerability label from either MTRAG label field."""
    if "Answerability" in record:
        raw_value = record["Answerability"]
    else:
        raw_value = record.get("answerability")

    if isinstance(raw_value, list):
        if not raw_value:
            return "MISSING"
        return str(raw_value[0])
    if raw_value is None:
        return "MISSING"
    return str(raw_value)


def map_label(raw_label: str) -> str | None:
    """Map raw answerability labels to the binary Y/N training labels."""
    if raw_label == "ANSWERABLE":
        return "Y"
    if raw_label in NEGATIVE_LABELS:
        return "N"
    return None


def token_length(sample: dict, tokenizer) -> int:
    """Count tokens in the full training string used by the longest invocation."""
    train_text = build_training_text(sample, tokenizer)
    token_ids = tokenizer(train_text, add_special_tokens=False)["input_ids"]
    return len(token_ids)


def collect_samples(source_paths: list[Path], tokenizer, max_tokens: int) -> list[dict]:
    """Read all sources, apply label and length filters, and return train samples."""
    samples = []
    total_raw = 0
    excluded_by_label = Counter()
    excluded_by_length = Counter()

    for source_path in source_paths:
        records = load_jsonl(source_path)
        if "raw_data" in source_path.parts:
            source_name = source_path.parts[source_path.parts.index("raw_data") + 1]
        else:
            source_name = source_path.stem
        source_counts = Counter()
        total_raw += len(records)

        for record in records:
            raw_label = get_raw_answerability(record)
            label = map_label(raw_label)
            if label is None:
                excluded_by_label[raw_label] += 1
                continue

            sample = {
                "input_text": build_input_text(record),
                "label": label,
            }
            length = token_length(sample, tokenizer)
            if length > max_tokens:
                excluded_by_length[label] += 1
                continue

            samples.append(sample)
            source_counts[label] += 1

        print(
            f"{source_name}: kept {sum(source_counts.values())} samples "
            f"Y={source_counts['Y']} N={source_counts['N']}"
        )

    label_counts = Counter(sample["label"] for sample in samples)
    print(f"Raw records read: {total_raw}")
    print(
        f"Kept after label + length filtering: {len(samples)} "
        f"Y={label_counts['Y']} N={label_counts['N']}"
    )
    print(f"Excluded by label: {dict(sorted(excluded_by_label.items()))}")
    print(f"Excluded by length > {max_tokens}: {dict(sorted(excluded_by_length.items()))}")
    return samples


# Output ----------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as UTF-8 JSONL."""
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_split_summary(name: str, split: list[dict]) -> None:
    """Print a compact label summary for one output split."""
    labels = [sample["label"] for sample in split]
    print(f"{name}: {len(split)} samples  Y={labels.count('Y')}  N={labels.count('N')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCES,
        help="Raw JSONL generation task files to combine.",
    )
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Drop samples whose full descriptive training string is longer than this.",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    samples = collect_samples(args.sources, tokenizer, args.max_tokens)
    labels = [sample["label"] for sample in samples]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        stratify=labels,
        random_state=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "train.jsonl", train_samples)
    write_jsonl(output_dir / "val.jsonl", val_samples)

    print_split_summary("train", train_samples)
    print_split_summary("val", val_samples)
    print("Done. Wrote train.jsonl and val.jsonl only.")


if __name__ == "__main__":
    main()
