"""
Preprocess reference.jsonl into train/val/test JSONL files.

Filters out CONVERSATIONAL records, maps PARTIAL -> unanswerable,
builds input_text from contexts + conversation history, and splits
80/10/10 with stratification on label.
"""
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def format_contexts(contexts: list) -> str:
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
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "unknown").capitalize()
        text = turn.get("text", "").strip()
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_input_text(record: dict) -> str:
    ctx_str = format_contexts(record.get("contexts", []))
    conv_str = format_conversation(record.get("input", []))
    return f"Documents:\n{ctx_str}\n\nConversation:\n{conv_str}"


def get_label(record: dict) -> str:
    answerability = record.get("Answerability", [])
    if answerability == ["ANSWERABLE"]:
        return "Y"
    return "N"


def load_and_filter(jsonl_path: str) -> list:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("Answerability") == ["CONVERSATIONAL"]:
                continue
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reference.jsonl")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_and_filter(args.input)
    print(f"Records after filtering CONVERSATIONAL: {len(records)}")

    samples = [{"input_text": build_input_text(r), "label": get_label(r)} for r in records]
    labels = [s["label"] for s in samples]

    y_count = labels.count("Y")
    n_count = labels.count("N")
    print(f"Label distribution — Y: {y_count}, N: {n_count}")

    # 80/10/10 stratified split
    train_samples, tmp = train_test_split(
        samples, test_size=0.2, stratify=labels, random_state=args.seed
    )
    tmp_labels = [s["label"] for s in tmp]
    val_samples, test_samples = train_test_split(
        tmp, test_size=0.5, stratify=tmp_labels, random_state=args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in split:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        split_labels = [s["label"] for s in split]
        print(f"{name}: {len(split)} samples  Y={split_labels.count('Y')}  N={split_labels.count('N')}")

    print("Done.")


if __name__ == "__main__":
    main()
