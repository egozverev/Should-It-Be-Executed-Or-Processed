import os
import json
import sys
import time
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from huggingface_hub import login

import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai_utils import completions_with_backoff

from typing import Union, List, Dict, Tuple, Optional


class ModelAPIHandler:
    def __init__(self, model_name: str, model_type: str) -> None:
        """
        Initializes the model handler based on the model name. Loads the model for hugging face models.

        Parameters:
        - model_name (str): The name of the model to be used.
        - model_type (str): The type (i.e., short abbreviation) of the model

        """
        self.model_name = model_name
        self.model_type = model_type
        self.model_family = self._get_model_family()
        self.model, self.tokenizer, self.pipeline = None, None, None
        access_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if access_token:
            login(token=access_token)

        if self.model_family == "hf":
            self._setup_hf_model(model_name)  # Stores Hugging Face models and tokenizers
        elif self.model_family == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def call_model_api(self, system_instruction: str, user_instruction: str) -> str:
        """
        Calls the appropriate model API based on the model family and formats the input accordingly.

        Parameters:
        - system_instruction (str): The system instruction for the model.
        - user_instruction (str): The user instruction for the model.

        Returns:
        - str: The model's response.
        """
        model_input = self._format_model_input(system_instruction, user_instruction)
        if self.model_family == "openai":
            response = completions_with_backoff(
                model=self.model_name,
                messages=model_input,  # Adapted for OpenAI
                max_tokens=3072
            )
            return response['choices'][0]['message']['content']
        else:
            response = self.pipeline(model_input)[0]['generated_text']
            return response

    def _get_model_family(self) -> str:
        """Determines the model's family based on its name."""
        return "openai" if self.model_name.startswith("gpt") else "hf"

    def _setup_hf_model(self, model_name) -> None:
        """
        Sets up a Hugging Face model and tokenizer, caching it for future use.
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": 0})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=4096,
                                 return_full_text=False)

    def _format_model_input(self, system_instruction: str, user_instruction: str) -> Union[List[Dict[str, str]], str]:
        """
        Formats the input for the model based on its family.

        Parameters:
        - system_instruction (str): The system instruction for the model.
        - user_instruction (str): The user instruction for the model.

        Returns:
        - Union[List[Dict[str, str]], str]: The formatted model input.
        """
        if self.model_family == "openai":
            return [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction}
            ]
        elif self.model_type in ("llama", "llama13b"):
            return f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_instruction} [/INST]"
        else:
            chat = [{"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction}]
            return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def load_config(config_path: str = './model_eval/config.json') -> Dict:
    """
    Loads configuration settings from a JSON file.

    Parameters:
    - config_path (str): The path to the configuration JSON file.

    Returns:
    - Dict: The loaded configuration settings.
    """
    with open(config_path, 'r', ) as file:
        return json.load(file)


def load_data(data_path: str) -> List[Dict]:
    """
    Loads the dataset and prompt templates from specified paths.

    Parameters:
    - data_path (str): The path to the dataset JSON file.

    Returns:
    - List[Dict]: The loaded dataset.
    """
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def format_prompt(elem: Dict, mode: str = 'data_with_probe') -> Tuple[str, str]:
    """
    Formats the prompt based on the provided data point and the mode.

    Parameters:
    - elem (Dict): The data point containing information for prompt formatting.
    - mode (str): The mode of prompt formatting. 'data_with_probe' for probe with data,
                  'probe_with_task' for probe with task prompt.

    Returns:
    - Tuple[str, str]: The formatted system and user instructions.

    Raises:
    - ValueError: If an invalid mode is provided.
    """
    if mode == 'data_with_probe':
        system_instruction = elem["system_prompt_clean"]
        user_instruction = elem["prompt_instructed"]
    elif mode == 'probe_with_task':
        system_instruction = elem["system_prompt_instructed"]
        user_instruction = elem["prompt_clean"]
    else:
        raise ValueError(
            f"Invalid mode for prompt formatting: {mode}. Valid modes are 'data_with_probe' or 'probe_with_task'.")
    return system_instruction, user_instruction


def inference(dataset: List[Dict], output_path: str, handler: ModelAPIHandler,
              save_step: str = 20) -> None:
    """
    Runs the inference process on the dataset, generating responses based on two sets of prompts for each data point.
    Writes the inference results to a JSON file specified by the output_path.

    Parameters:
        dataset (List[Dict]): The dataset to process.
        output_path (str): The path where the inference results will be saved.
        handler (ModelAPIHandler): The API handler object for making model calls.
        save_step (str): saves inference result every save_step steps.
    """
    output = []
    for i, data_point in enumerate(tqdm(dataset, desc=f"Processing dataset")):
        # First prompt with probe in data
        sys_instr_1, user_instr_1 = format_prompt(data_point, mode='data_with_probe')
        # Second prompt with probe in task
        sys_instr_2, user_instr_2 = format_prompt(data_point, mode='probe_with_task')

        response1 = handler.call_model_api(sys_instr_1, user_instr_1)
        response2 = handler.call_model_api(sys_instr_2, user_instr_2)
        output.append({
            "output1_probe_in_data": response1,
            "output2_probe_in_task": response2,
            "model": handler.model_name,
            "instructions": {
                "sys_instr_1": sys_instr_1,
                "user_instr_1": user_instr_1,
                "sys_instr_2": sys_instr_2,
                "user_instr_2": user_instr_2
            },
            "data": data_point
        })
        if i % save_step == 0 or i == len(dataset) - 1:
            with open(output_path, "w") as f:
                json.dump(output, f)


def main(model_ix: int,  start_ix: Optional[int] = None, end_ix: Optional[int] = None) -> None:
    """
    Executes the model inference process based on specified command line arguments.

    Parameters:
        model_ix (int): Index to select the model configuration.
        start_ix (Optional[int]): Start index for slicing the dataset, or None to start from the beginning.
        end_ix (Optional[int]): End index for slicing the dataset, or None to go till the end of the dataset.
    """
    config = load_config()
    model_type = config["model_types"][model_ix]
    model_name = config["models"][model_ix]
    dataset = load_data(config["input_path"])

    if start_ix is None:
        start_ix = 0
    if end_ix is None:
        end_ix = len(dataset)
    output_dir_path = os.path.join(config["output_base_path"], model_type)
    os.makedirs(output_dir_path, exist_ok=True)
    output_file_path = os.path.join(output_dir_path, f"{start_ix}-{end_ix}.json")
    dataset = dataset[start_ix: end_ix]
    handler = ModelAPIHandler(model_name, model_type)
    print(f"Starting inference for model {model_name}. Dataset slice is dataset[{start_ix}:{end_ix}]")
    inference(dataset, output_file_path, handler)

    print(f"Inference complete. Results saved to {output_file_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        print(
            "Usage: get_model_outputs.py <model_ix> (optional) <start_ix> (optional) <end_ix>")
        sys.exit(1)

    main(*map(int, sys.argv[1:]))
