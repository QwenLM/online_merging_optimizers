#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 python ../../src/export_model.py \
    --model_name_or_path /cpfs2966/data/shared/public/dangkai.dk/hf-models/0124_sft/qwen2-1b.2100B--mixv22-base100w-cpt-chatml-seqlen8k-avg4k--v10.15.23_redbook-lr1e-5/ \
    --adapter_name_or_path /cpfs01/data/shared/public/lukeming.lkm/online_merging/trl/dpo_ultrafeedback_adam_0.01_ema_True_rescale_True_shrink_False_alpha_1e-5_beta_0.1_lr_8e-7_peft/checkpoint-8000 \
    --template default \
    --finetuning_type lora \
    --export_dir /cpfs01/data/shared/public/lukeming.lkm/online_merging/trl/dpo_ultrafeedback_adam_0.01_ema_True_rescale_True_shrink_False_alpha_1e-5_beta_0.1_lr_8e-7_peft_merge/checkpoint-8000 \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
