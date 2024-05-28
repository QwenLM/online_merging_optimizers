export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_DISABLED=true

MODEL_SIZE=7B
BETA=0.1
LR=1e-6
OPTIMIZER=ondare
USE_EMA=True
USE_RESCALE=False
RESERVE_P=0.01
ALPHA=1e-7
DATASET=ultrafeedback_binarized
POSTFIX=

MODEL_PATH="reference model path" # set your reference model
BASE_MODEL_PATH="base model path" # set your base model

OUTPUT_DIR=saves/Qwen-${MODEL_SIZE}/dpo/${DATASET}_${OPTIMIZER}_${DROP_RATE}_ema_${USE_EMA}_rescale_${USE_RESCALE}_shrink_${SHRINK_BASE}_alpha_${ALPHA}_beta_${BETA}_lr_${LR}$POSTFIX

accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --dataset_dir data \
    --template qwen \
    --dpo_beta $BETA \
    --dpo_loss sigmoid \
    --ref_model $MODEL_PATH \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy no \
    --learning_rate $LR \
    --num_train_epochs 2.0 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --report_to tensorboard \
    --bf16 \
    --use_online_merging \
    --om_mode $OPTIMIZER \
    --reserve_p ${RESERVE_P} \
    --alpha $ALPHA \
    --use_merge ${USE_EMA} \
    --base_model ${BASE_MODEL_PATH} \
    --rescale ${USE_RESCALE} 
