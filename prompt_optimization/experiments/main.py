'''A main script to run attack for LLMs.'''
import argparse
import time
import importlib

from typing import Any
import os
import sys
import json
sys.path.append("../rpo/")  # Adds higher directory to python modules path.

from rpo.suffix_manager import get_goals_and_targets, get_workers

from huggingface_hub import login

def dynamic_import(module: str):
    """
    Dynamically import a module given its name as a string.

    Parameters:
    module (str): The name of the module to import.

    Returns:
    module: The imported module object.

    Example:
    >>> math_module = dynamic_import('math')
    >>> math_module.sqrt(16)
    4.0

    Raises:
    ImportError: If the module cannot be imported.
    """
    return importlib.import_module(module)

def main(params: Any) -> None:
    """
    Main function to run a Progressive Multi-Prompt Attack using specified parameters.

    Parameters:
    params (Any): The parameters required to configure and run the attack, typically 
                  provided through a configuration object or command-line arguments.

    The function performs the following steps:
    1. Retrieves the Hugging Face Hub token from environment variables and logs in if available.
    2. Dynamically imports the attack library.
    3. Initializes workers and data loaders for training and testing.
    4. Configures managers for the attack.
    5. Creates an instance of the ProgressiveMultiPromptAttack class with specified parameters.
    6. Runs the attack with specified parameters.
    7. Stops all workers after the attack is complete and prints "SUCCESS".
    """
    access_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if access_token:
        login(token=access_token)
    attack_lib = dynamic_import(f'rpo')

    workers, test_workers = get_workers(params)


    train_loader, test_loader = get_goals_and_targets(params)


    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")


    attack = attack_lib.ProgressiveMultiPromptAttack(
        train_loader,
        test_loader,
        workers,
        model_name=params.model,
        progressive_models=params.progressive_models,
        progressive_goals=params.progressive_goals,
        control_init=params.control_init,
        safe_init=params.safe_init,
        logfile=f"{params.result_prefix}/{params.model}_{timestamp}_cut_cand_i_len_to_20.json",
        managers=managers,
        test_workers=test_workers,
        mpa_deterministic=params.gbda_deterministic,
        mpa_lr=params.lr,
        mpa_batch_size=params.batch_size,
        mpa_n_steps=params.n_steps,
    )

    attack.run(
        n_epochs=params.n_epochs,
        batch_size=params.batch_size,
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=(params.allow_non_ascii == "True"),
        selection_interval=params.selection_interval
    )

    for worker in workers + test_workers:
        worker.stop()
    print("SUCCESS")

def set_config_default(config: Any) -> Any:
    """
    Set default configuration parameters for the attack.

    Parameters:
    config (Any): The configuration object to set default values for.

    Returns:
    Any: The configuration object with default values set.
    """
    config.target_weight = 1.0
    config.control_weight = 0.0
    config.progressive_goals = False
    config.progressive_models = False
    config.anneal = False
    config.incr_control = False
    config.stop_on_success = False
    config.verbose = True
    config.num_train_models = 1
    config.selection_interval = 100
    config.data_offset = 0

    # attack-related parameters
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True

    config.gbda_deterministic = True
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A main script to run attack for LLMs.')

    # Replace these with your actual command-line arguments
    parser.add_argument("--model", type=str, help='Model name.', default="llama-2")

    parser.add_argument("--attack", type=str, help='Attack type.',
                        default="gcg")
    parser.add_argument('--train_data', type=str, help='Path to train data.',
                        default="")
    parser.add_argument('--test_data', type=str, help='Path to test data.',
                        default="")
    parser.add_argument('--result_prefix', type=str, help='Prefix for result files.',
                        default=f"./experiments/sep_results")  # add model during saving!!!!
    parser.add_argument('--control_init', type=str, help='Initial control setting.',
                        default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
    parser.add_argument('--safe_init', type=str, help='Initial safe setting.',
                        default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
    parser.add_argument('--progressive_models',
                        help='Use progressive models. Defaults to False.', default=False)
    parser.add_argument('--progressive_goals',
                        help='Use progressive goals. Defaults to False.', default=False)
    parser.add_argument('--stop_on_success',
                        help='Stop on success. Defaults to False.', default=False)
    parser.add_argument('--num_train_models', type=int,
                        help='Number of training models.', default=1)
    parser.add_argument('--allow_non_ascii',
                        help='Allow non-ASCII characters. Defaults to False.', default=False)
    parser.add_argument('--n_test_data', type=int, help='Number of test data points.',
                        default=4)
    parser.add_argument('--n_steps', type=int, help='Number of steps.', default=2000)
    parser.add_argument('--n_epochs', type=int, help='Number of steps.', default=1)
    parser.add_argument('--test_steps', type=int, help='Number of test steps.', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size for tokens.', default=8)
    parser.add_argument('--data_batch_size', type=int, help='Batch size for data.', default=3)
    parser.add_argument('--steps_per_data_batch', type=int, help='Batch size for data.', default=20)

    parser.add_argument('--selection_interval', type=int, help='Selection interval.',
                        default=100)
    parser.add_argument('--transfer', type=str, help='Do transfer.',
                        default=True)
    parser.add_argument('--gbda_deterministic', type=str, help='Is GDBA deterministic.',
                        default=True)
    parser.add_argument('--tokenizer_paths', type=json.loads, help='Tokenizer paths.',
                        default=("meta-llama/Llama-2-7b-chat-hf",))
    parser.add_argument('--model_paths', type=json.loads, help='Tokenizer paths.',
                        default=("meta-llama/Llama-2-7b-chat-hf",))
    parser.add_argument('--tokenizer_kwargs', type=tuple, help='Tokenizer kwargs.',
                        default=(({"use_fast": False}, )))
    parser.add_argument('--model_kwargs', type=tuple, help='Model kwargs.',
                        default=(({"low_cpu_mem_usage": True, "use_cache": True}, )))
    parser.add_argument('--conversation_templates', type=json.loads, help='Conv templates.',
                        default=("llama-2",))


    parser.add_argument('--devices', type=tuple, help='Devices.',
                        default=("cuda:0", "cuda:1", "cuda:2", "cuda:3"))

    # This line parses the arguments
    args = parser.parse_args()
    args = set_config_default(args)
    main(args)
