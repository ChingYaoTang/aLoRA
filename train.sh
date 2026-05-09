source env_cuda.sh

INVOC_TYPE=shorter
python src/train.py --invoc_type "$INVOC_TYPE" --output_dir adapters/Qwen2.5-0.5B/"$INVOC_TYPE" 2>&1 | tee logs/Qwen2.5-0.5B/"$INVOC_TYPE".log