source env_cuda.sh

INVOC_TYPE=descriptive
python src/evaluate.py --adapter_path adapters/Qwen2.5-0.5B/"$INVOC_TYPE" --test_data data/val.jsonl --output results/Qwen2.5-0.5B/"$INVOC_TYPE"_results.json