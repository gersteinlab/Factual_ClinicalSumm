import logging
import os, math
import sys, torch
import datasets
import nltk 
import transformers
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import numpy as np
from datasets import load_dataset, load_metric
from accelerate import Accelerator
import matplotlib.pyplot as plt
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    CONFIG_MAPPING,
    set_seed,
    AdamW,
    get_scheduler,
)


@dataclass
class ModelOptions:
    model_path: str = "Path to pre-trained model or model identifier from hugginface.co/models"
    configuration_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_directory: Optional[str] = None
    use_fast_tokenization: bool = True
    revision_version: str = "main"
    use_authentication_token: bool = False
    model_type: str = None

@dataclass
class TrainingDataOptions:
    data_collection: Optional[str] = None
    dataset_config: Optional[str] = None
    text_col: Optional[str] = None
    summary_col: Optional[str] = None
    train_data: Optional[str] = None
    eval_data: Optional[str] = None
    test_data: Optional[str] = None
    cache_overwrite: bool = False
    preprocessing_workers: Optional[int] = None
    max_input_len: Optional[int] = 1024
    max_output_len: Optional[int] = 128
    val_max_output_len: Optional[int] = None
    pad_to_max_len: bool = False
    max_training_samples: Optional[int] = None
    max_evaluation_samples: Optional[int] = None
    max_prediction_samples: Optional[int] = None
    beam_count: Optional[int] = None
    ignore_padding_in_loss: bool = True
    source_text_prefix: Optional[str] = None
    val_max_target_length: Optional[int] = None
    top_k: Optional[int] = None
    temperature: Optional[int] = None
    epetition_penalty: Optional[float] = 1.0
    learning_rate: Optional[float] = 5e-5
    model_type: Optional[str] = None
    num_train_epochs: Optional[int] = 3
    weight_decay: Optional[float] = 0.0
    reward_type: Optional[str] = None
    min_length: Optional[int] = None
    repetition_penalty: Optional[float] = 1.0
    ignore_pad_token_for_loss: Optional[bool] = True
    reward_rate: Optional[float] = 0
    gradient_accumulation_steps: Optional[int] = 1
    max_train_steps: Optional[int] = None
    seed: Optional[int] = None



def initialize_logger():
    log_name = __name__
    logger = logging.getLogger(log_name)

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"

    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    logger.addHandler(file_handler)

    return logger


def main():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)

    parser = HfArgumentParser((ModelOptions, TrainingDataOptions, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1][-5:-1](".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = initialize_logger()
    accelerator = Accelerator()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_text_prefix is None and model_args.model_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_text_prefix 'summarize: ' `"
        )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    if data_args.data_collection is not None:
        raw_datasets = load_dataset(
            data_args.data_collection, data_args.dataset_configuration_name, cache_directory=model_args.cache_directory
        )
    else:
        data_files = {}
        if data_args.train_data is not None:
            data_files["train"] = data_args.train_data
            extension = data_args.train_data.split(".")[-1]
        if data_args.eval_data is not None:
            data_files["validation"] = data_args.eval_data
            extension = data_args.eval_data.split(".")[-1]
        if data_args.test_data is not None:
            data_files["test"] = data_args.test_data
            extension = data_args.test_data.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_directory=model_args.cache_directory)

    if model_args.configuration_name:
        config = AutoConfig.from_pretrained(
            model_args.configuration_name,
            cache_dir=model_args.cache_directory,
            revision=model_args.revision_version,
            use_auth_token=True if model_args.use_authentication_token else None,
        )
    elif model_args.model_path:
        config = AutoConfig.from_pretrained(
            model_args.model_path,
            cache_dir=model_args.cache_directory,
            revision=model_args.revision_version,
            use_auth_token=True if model_args.use_authentication_token else None,
        )
    else:
        config = CONFIG_MAPPING[data_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_directory,
            use_fast=model_args.use_fast_tokenization,
            revision=model_args.revision_version,
            use_auth_token=True if model_args.use_authentication_token else None,
        )
    elif model_args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_path,
            cache_dir=model_args.cache_directory,
            use_fast=model_args.use_fast_tokenization,
            revision=model_args.revision_version,
            use_auth_token=True if model_args.use_authentication_token else None,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    if model_args.model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_path,
            from_tf=bool(".ckpt" in model_args.model_path),
            config=config,
            cache_dir=model_args.cache_directory,
            revision=model_args.revision_version,
            use_auth_token=True if model_args.use_authentication_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_text_prefix if data_args.source_text_prefix is not None else ""

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    dataset_columns = summarization_name_mapping.get(data_args.data_collection, None)
    if data_args.text_col is None:
        text_col = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_col = data_args.text_col
        if text_col not in column_names:
            raise ValueError(
                f"--text_col' value '{data_args.text_col}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_col is None:
        summary_col = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_col = data_args.summary_col
        if summary_col not in column_names:
            raise ValueError(
                f"--summary_col' value '{data_args.summary_col}' needs to be one of: {', '.join(column_names)}"
            )

    max_output_len = data_args.max_output_len
    padding = "max_length" if data_args.pad_to_max_len else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
 
    def preprocess_function(examples):
        inputs = examples[text_col]
        targets = examples[summary_col]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_input_len, padding=padding, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_output_len, padding=padding, truncation=True)

        if padding == "max_length" and data_args.ignore_padding_in_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_training_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_training_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.cache_overwrite,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_output_len = data_args.val_max_output_len
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_evaluation_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_evaluation_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.cache_overwrite,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_output_len = data_args.val_max_output_len
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_prediction_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_prediction_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.cache_overwrite,
                desc="Running tokenizer on prediction dataset",
            )

    label_pad_token_id = -100 if data_args.ignore_padding_in_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=data_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=data_args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": data_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=data_args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / data_args.gradient_accumulation_steps)
    if data_args.max_train_steps is None:
        data_args.max_train_steps = data_args.num_train_epochs * num_update_steps_per_epoch
    else:
        data_args.num_train_epochs = math.ceil(data_args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=data_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=data_args.num_warmup_steps,
        num_training_steps=data_args.max_train_steps,
    )

    # Train!
    total_batch_size = data_args.per_device_train_batch_size * accelerator.num_processes * data_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {data_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {data_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {data_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {data_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(data_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    # Metric
    if data_args.reward_type is not None:
        metric = load_metric(data_args.reward_type)

    loss_plot = []
        
    def process_batch_and_generate_tokens(network_model, data_accelerator, text_tokenizer, batch, generation_args, data_parameters):
        with torch.no_grad():
            generated_text = data_accelerator.unwrap_model(network_model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **generation_args,
            )

            generated_text = data_accelerator.pad_across_processes(
                generated_text, dim=1, pad_index=text_tokenizer.pad_token_id
            )
            ground_truth_labels = batch["labels"]
            if not data_parameters.pad_to_max_length:
                ground_truth_labels = data_accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=text_tokenizer.pad_token_id)

            generated_text = data_accelerator.gather(generated_text).cpu().numpy()
            ground_truth_labels = data_accelerator.gather(ground_truth_labels).cpu().numpy()

            if data_parameters.ignore_pad_token_for_loss:
                ground_truth_labels = np.where(ground_truth_labels != -100, ground_truth_labels, text_tokenizer.pad_token_id)
            if isinstance(generated_text, tuple):
                generated_text = generated_text[0]
                
            return generated_text, ground_truth_labels

    def calculate_reward_and_loss(data_parameters, decoded_predictions, decoded_labels, initial_loss, evaluation_metric):
        if data_parameters.reward_type == "bertscore":
            scoring = evaluation_metric.compute(predictions=decoded_predictions, references=decoded_labels, lang="en")
            reward = data_parameters.reward_rate*(1 - scoring['f1'][0])
        elif data_parameters.reward_type == "meteor":
            scoring = evaluation_metric.compute(predictions=decoded_predictions, references=decoded_labels)
            reward = data_parameters.reward_rate*(1 - scoring['meteor'])
            
        print("Initial loss = " + str(initial_loss) + ", Reward = "+ str(reward))
        updated_loss = initial_loss + reward
        updated_loss = updated_loss / data_parameters.gradient_accumulation_steps

        return updated_loss

    # Main loop starts here
    for epoch_count in range(data_args.num_train_epochs):
        model.train()
        for step_count, batch_data in enumerate(train_dataloader):
            model_outputs = model(**batch_data)
            loss_value = model_outputs.loss

            if data_args.reward_type is not None:
                model.eval()
                generation_params = {
                    "max_length": data_args.val_max_target_length if data_args.val_max_target_length is not None else config.max_length,
                    "min_length": data_args.min_length,
                    "top_k": data_args.top_k,
                    "temperature": data_args.temperature,
                    "repetition_penalty": data_args.repetition_penalty,
                    "early_stopping": True, 
                    "do_sample": True,
                }
                generated_tokens, label_tokens = process_batch_and_generate_tokens(model, accelerator, tokenizer, batch_data, generation_params, data_args)
                
                decoded_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_ground_truth = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
                
                loss_value = calculate_reward_and_loss(data_args, decoded_predictions, decoded_ground_truth, loss_value, metric)
                # Other training steps, e.g., backpropagation, optimizer, etc., are the same and have been omitted

    # Evaluation loop starts here
    model.eval()
    generation_params_eval = {
        "max_length": data_args.val_max_target_length if data_args.val_max_target_length is not None else config.max_length,
        "min_length": data_args.min_length,
        "top_k": data_args.top_k,
        "temperature": data_args.temperature,
        "repetition_penalty": data_args.repetition_penalty,
        "early_stopping": True, 
        "do_sample": True,
    }
    predictions_list, ground_truth_list = [], []
    for step_count, batch_data in enumerate(eval_dataloader):
        generated_tokens_eval, label_tokens_eval = process_batch_and_generate_tokens(model, accelerator, tokenizer, batch_data, generation_params_eval, data_args)
        
        decoded_predictions_eval = tokenizer.batch_decode(generated_tokens_eval, skip_special_tokens=True)
        decoded_ground_truth_eval = tokenizer.batch_decode(label_tokens_eval, skip_special_tokens=True)
        decoded_predictions_eval, decoded_ground_truth_eval = postprocess_text(decoded_predictions_eval, decoded_ground_truth_eval)
        
        # Other evaluation steps, e.g., metrics computation, etc., are the same and have been omitted

    if data_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(data_args.output_dir, save_function=accelerator.save)

    plot_y = list(range(len(loss_plot)))
    plt.plot(loss_plot, plot_y)
    plt.show()


if __name__ == "__main__":
    main()