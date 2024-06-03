from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,

)

from peft import LoraConfig

from trl import SFTTrainer


@dataclass
class ScriptArguments:
    """
    A class to hold script arguments for model training.

    Attributes:
    dataset_path (str): Path to the dataset.
    dataset_text_field (str): Dataset text field used for decoder-only training. Default is "text".
    model_id (str): Model ID to use for SFT training.
    max_seq_length (int): The maximum sequence length for SFT Trainer. Default is 512.
    training_mode (str): Training mode: lora, qlora, or fft. Default is "lora".
    attention_impl (str): Attention implementation: sdpa or flash_attention_2. Default is "sdpa".
    lora_r (int): LoRA r parameter. Default is 16.
    lora_alpha (int): LoRA alpha parameter. Default is 8.
    lora_dropout (float): LoRA dropout parameter. Default is 0.05.
    peft_target_modules (str): PEFT target modules. Default is "all-linear".
    """
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset text field used for decoder_only training"}
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    training_mode: str = field(
        default="lora", metadata={"help": "Training mode: lora, qlora or fft"}
    )
    attention_impl: str = field(
        default="sdpa", metadata={"help": "Attention implementation: sdpa or flash_attention_2"}
    )
    lora_r: int = field(
        default=16, metadata={"help": "LoRA r parameter"}
    )
    lora_alpha: int = field(
        default=8, metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout parameter"}
    )
    peft_target_modules: str = field(
        default="all-linear", metadata={"help": "PEFT target modules"}
    )


def training_function(script_args: ScriptArguments, training_args: TrainingArguments) -> None:
    """
    Train a model using the specified script arguments and training arguments.

    Parameters:
    script_args (ScriptArguments): The script arguments for model training.
    training_args (TrainingArguments): The training arguments for the Trainer.

    The function performs the following steps:
    1. Load the training and testing datasets from JSON files.
    2. Initialize the tokenizer using the specified model ID.
    3. Print a few random samples from the training set.
    4. Initialize the model with or without quantization based on the training mode.
    5. Configure PEFT settings if using LoRA or QLoRA training mode.
    6. Train the model using SFTTrainer and save the trained model.
    """

    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "test_dataset.json"),
        split="train",
    )

    ################
    # Model & Tokenizer
    ################

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token


    # print random sample
    with training_args.main_process_first(
            desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index][script_args.dataset_text_field])

    # Model
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    if script_args.training_mode == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation=script_args.attention_impl,
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=True if 'microsoft/Phi-3' in script_args.model_id else False,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    if script_args.training_mode in ["lora", "qlora"]:
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            target_modules=script_args.peft_target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"]
        )
    else:
        peft_config = None

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field=script_args.dataset_text_field,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )
    if trainer.accelerator.is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)
