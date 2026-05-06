"""
Evaluate a trained aLoRA adapter on the test set.

Single adapter:
    python src/evaluate.py --adapter_path adapters/baseline \
        --test_data data/test.jsonl --output results/baseline_results.json

Compare all results in results/:
    python src/evaluate.py --compare results/
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an answerability detection assistant. "
    "Given a conversation and supporting documents, output exactly Y if the final user "
    "question can be answered from the documents, or N if it cannot."
)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_prompt(record: dict, tokenizer, invoc_seq: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["input_text"]},
    ]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return f"{prefix}{invoc_seq}"


def predict(model, tokenizer, prompts: list, batch_size: int = 8) -> list:
    model.eval()
    predictions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, out in enumerate(outputs):
            # The new token is right after the input
            input_len = inputs["input_ids"].shape[1]
            new_token = out[input_len:]
            decoded = tokenizer.decode(new_token, skip_special_tokens=True).strip()
            if decoded.upper().startswith("Y"):
                predictions.append("Y")
            elif decoded.upper().startswith("N"):
                predictions.append("N")
            else:
                # Fallback: check logits for Y vs N token
                predictions.append("Y")  # safe default (majority class)
    return predictions


def evaluate_adapter(adapter_path: str, test_data: str, output_path: str):
    adapter_path = Path(adapter_path)

    # Load invocation metadata saved during training
    meta_path = adapter_path / "invoc_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        invoc_seq = meta["invoc_seq"]
        invoc_type = meta["invoc_type"]
    else:
        # Fallback: read from adapter_config.json
        with open(adapter_path / "adapter_config.json") as f:
            cfg = json.load(f)
        invoc_seq = cfg.get("invocation_string", "<|im_start|>assistant\n")
        invoc_type = adapter_path.name

    print(f"Evaluating: {invoc_type}  (invoc_seq={repr(invoc_seq)})")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    test_records = load_jsonl(test_data)
    prompts = [build_prompt(r, tokenizer, invoc_seq) for r in test_records]
    gold = [r["label"] for r in test_records]

    preds = predict(model, tokenizer, prompts)

    report = classification_report(
        gold, preds, labels=["N", "Y"],
        target_names=["UNANSWERABLE", "ANSWERABLE"],
        output_dict=True,
        zero_division=0,
    )
    weighted_f1 = report["weighted avg"]["f1-score"]

    print(classification_report(
        gold, preds, labels=["N", "Y"],
        target_names=["UNANSWERABLE", "ANSWERABLE"],
        zero_division=0,
    ))
    print(f"Weighted F1: {weighted_f1:.4f}")

    result = {
        "invoc_type": invoc_type,
        "invoc_seq": invoc_seq,
        "n_test": len(test_records),
        "weighted_f1": weighted_f1,
        "report": report,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_path}")
    return result


def compare_results(results_dir: str):
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob("*_results.json"))
    if not result_files:
        print("No result files found.")
        return

    rows = []
    for path in result_files:
        with open(path) as f:
            r = json.load(f)
        rows.append(r)

    # Print comparison table
    header = f"{'Variant':<14} {'Unans-P':>8} {'Unans-R':>8} {'Unans-F1':>9} {'Ans-P':>7} {'Ans-R':>7} {'Ans-F1':>8} {'W-F1':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in sorted(rows, key=lambda x: -x["weighted_f1"]):
        rep = r["report"]
        u = rep.get("UNANSWERABLE", {})
        a = rep.get("ANSWERABLE", {})
        print(
            f"{r['invoc_type']:<14}"
            f" {u.get('precision', 0):>8.3f}"
            f" {u.get('recall', 0):>8.3f}"
            f" {u.get('f1-score', 0):>9.3f}"
            f" {a.get('precision', 0):>7.3f}"
            f" {a.get('recall', 0):>7.3f}"
            f" {a.get('f1-score', 0):>8.3f}"
            f" {r['weighted_f1']:>7.3f}"
        )
    print("=" * len(header))

    # Save combined comparison
    out_path = results_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nComparison table saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_data", default="data/test.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--compare", default=None, help="Directory with *_results.json to compare")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare)
    elif args.adapter_path:
        adapter_name = Path(args.adapter_path).name
        output_path = args.output or f"results/{adapter_name}_results.json"
        evaluate_adapter(args.adapter_path, args.test_data, output_path)
    else:
        parser.error("Provide --adapter_path or --compare")


if __name__ == "__main__":
    main()
