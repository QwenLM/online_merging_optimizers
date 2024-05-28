# Inspired by: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push, create_ref_model, create_base_model
from .trainer import CustomDPOTrainer
import accelerate

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    logger.warning('Loading model finished!')

    data_collator = PairwiseDataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Create reference model
    if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
        ref_model = model
    else:
        print("loading ref model begin")
        ref_model = create_ref_model(model_args, finetuning_args)
    
    base_model = create_base_model(model_args, finetuning_args)
    print("loading base model finished")

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        base_model=base_model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )

    logger.info('begin to wrap policy model...')
    print('begin to wrap policy model...')
    prepared_model = trainer._wrap_model(
        trainer.model, training=True, dataloader=None
    )
    if hasattr(trainer.lr_scheduler, "step"):
        prepared_model, trainer.optimizer = trainer.accelerator.prepare(
            prepared_model, trainer.optimizer
        )
    else:
        (
            prepared_model,
            trainer.optimizer,
            trainer.lr_scheduler,
        ) = trainer.accelerator.prepare(
            prepared_model, trainer.optimizer, trainer.lr_scheduler
        )
    trainer.model_wrapped = prepared_model
    if trainer.is_fsdp_enabled:
        trainer.model = prepared_model
    
    logger.info('begin to wrap ref model...')
    print('begin to wrap ref model...')
    trainer.ref_model = trainer.accelerator.prepare_model(trainer.ref_model)
    print("loading ref model finished")
    
    if trainer.base_model is not None:
        logger.info('begin to wrap base model...')
        print('begin to wrap base model...')
        trainer.base_model = trainer.accelerator.prepare_model(trainer.base_model)

    trainer.accelerator.prepare_model = lambda model, *args, **kwargs: model # Monkey-patch prepare_model a no-op , since we have manually prepared the models

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards without a reference model
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
