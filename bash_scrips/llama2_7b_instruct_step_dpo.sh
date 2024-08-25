export output_dir="/data/yuansui/dpo/outputs"
export hub_model_id="llama2_7b_instruct_sft_dpo"
export prompt="alpaca"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_cpu.yaml --mixed_precision bf16 \
    --num_processes 4 \
    train.py configs/config_full.yaml \
    --model_name_or_path=/data/yuansui/dpo/outputs/llama-2-7b-instruct-sft \
    --data_path="xinlai/Math-Step-DPO-10K" \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=16 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --beta=0.5 \
    --num_train_epochs=8 \
    --save_strategy='steps' \
    --save_steps=400 \
    --save_total_limit=1 \
    --output_dir=$output_dir/$hub_model_id \
    --hub_model_id=$hub_model_id \
    --prompt=$prompt \
    --push_to_hub=True

python eval_math.py --model $output_dir/$hub_model_id --data_file /data/yuansui/dpo/data/test/GSM8K_test_data.jsonl --save_path 'eval_results/gsm8k/'$hub_model_id'.json' --prompt $prompt --tensor_parallel_size 4

# python eval_math.py --model outputs/$output_dir --data_file ./data/test/MATH_test_data.jsonl --save_path 'eval_results/math/'$output_dir'.json' --prompt $prompt --tensor_parallel_size 4
