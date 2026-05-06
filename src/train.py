"""
Train an aLoRA adapter on the answerability detection task.

Usage:
    python src/train.py --invoc_type baseline --output_dir adapters/baseline
    python src/train.py --invoc_type shorter  --output_dir adapters/shorter
    python src/train.py --invoc_type descriptive --output_dir adapters/descriptive
    python src/train.py --invoc_type generic   --output_dir adapters/generic

Smoke-test (2 epochs, 10 samples):
    python src/train.py --invoc_type baseline --output_dir adapters/smoke \
        --max_samples 10 --num_epochs 2
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Invocation sequences in Qwen chat-template role format
INVOC_SEQS = {
    "baseline":    "<|im_start|>answerability\n",
    "shorter":     "<|im_start|>check\n",
    "descriptive": "<|im_start|>determine if the question is answerable given the documents\n",
    "generic":     "<|im_start|>assistant\n",
}

SYSTEM_PROMPT = (
    "You are an answerability detection assistant. "
    "Given a conversation and supporting documents, output exactly Y if the final user "
    "question can be answered from the documents, or N if it cannot."
)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f]


def format_sample(record: dict, tokenizer, invoc_seq: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["input_text"]},
    ]
    # Apply chat template without the generation prompt so we can append our own invocation seq
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    label = record["label"]
    # Full training string: <chat_prefix><invoc_seq><Y/N><|im_end|>\n
    return f"{prefix}{invoc_seq}{label}<|im_end|>\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--invoc_type", required=True, choices=list(INVOC_SEQS.keys()))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_data", default="data/train.jsonl")
    parser.add_argument("--val_data", default="data/val.jsonl")
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for smoke test")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    invoc_seq = INVOC_SEQS[args.invoc_type]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Invocation type : {args.invoc_type}")
    print(f"Invocation seq  : {repr(invoc_seq)}")
    print(f"Output dir      : {output_dir}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"

    invoc_token_ids = tokenizer(invoc_seq, add_special_tokens=False)["input_ids"]
    print(f"Invocation token IDs: {invoc_token_ids}")

    # ── Model (QLoRA 4-bit) ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA / aLoRA config ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        alora_invocation_tokens=invoc_token_ids,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Save invocation sequence metadata alongside adapter
    with open(output_dir / "invoc_meta.json", "w") as f:
        json.dump({"invoc_type": args.invoc_type, "invoc_seq": invoc_seq, "invoc_token_ids": invoc_token_ids}, f, indent=2)

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_raw = load_jsonl(args.train_data)
    val_raw = load_jsonl(args.val_data)

    if args.max_samples:
        train_raw = train_raw[: args.max_samples]
        val_raw = val_raw[: max(2, args.max_samples // 4)]

    train_texts = [format_sample(r, tokenizer, invoc_seq) for r in train_raw]
    val_texts = [format_sample(r, tokenizer, invoc_seq) for r in val_raw]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    # ── Collator (loss only on label token after invocation seq) ───────────────
    # response_template is the invocation sequence — collator masks everything before it
    collator = DataCollatorForCompletionOnlyLM(
        response_template=invoc_seq,
        tokenizer=tokenizer,
    )

    # ── Training ───────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
