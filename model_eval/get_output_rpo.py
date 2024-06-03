from get_model_outputs import inference, load_config, load_data
from get_model_outputs import ModelAPIHandler
from typing import Optional

import os
import sys


def main(mode: str, model_ix: int, prompt_ix: int, prompt_ix_end: Optional[int], start_ix: Optional[int] = None, end_ix: Optional[int] = None) -> None:
    """
    Executes the model inference process based on specified command line arguments.

    Parameters:
        mode (str): Either "train" for saving results in train folder, or "eval" for saving in eval folder
        model_ix (int): Index to select the model configuration.
        prompt_ix (int): Index to select the prompt template.
        start_ix (Optional[int]): Start index for slicing the dataset, or None to start from the beginning.
        end_ix (Optional[int]): End index for slicing the dataset, or None to go till the end of the dataset.
    """
    assert mode in ("rpo", "rpoeval"), "Wrong mode"
    config = load_config()
    model_type = config["model_types"][model_ix]
    model_name = config["models"][model_ix]
    if prompt_ix == prompt_ix_end:
        raise Exception("Prompt index interval is empty")
    if prompt_ix_end is None:
        prompt_ix_end = prompt_ix + 1
    prompt_templates_path = f"./model_eval/rpo_suffixes/{model_type}.json"
    for p_ix in range(prompt_ix, prompt_ix_end):
        input_path = config["train_input_path"] if (mode == "rpo") else config["eval_input_path"]
        dataset, prompt_template = load_data(input_path, prompt_templates_path, p_ix)
        if start_ix is None:
            start_ix = 0
        if end_ix is None:
            end_ix = len(dataset)
        if mode == "rpo":
            output_dir_path = os.path.join(config["output_base_path"], mode, model_type, f"prompt_{prompt_template['step']}")
        else:
            assert  mode == "rpoeval"
            output_dir_path = os.path.join(config["output_base_path"], "eval", model_type, f"prompt_rpo_{prompt_template['step']}")

        os.makedirs(output_dir_path, exist_ok=True)
        output_file_path = os.path.join(output_dir_path, f"{start_ix}-{end_ix}.json")
        dataset = dataset[start_ix: end_ix]
        template_info = {"template_prompt_ix": p_ix, "template_prompt": prompt_template}
        handler = ModelAPIHandler(model_name, model_type, mode, prompt_ix)
        print(f"Starting inference for model {model_name} on prompt index {p_ix}. \
              Dataset slice is dataset[{start_ix}:{end_ix}]")
        inference(dataset, output_file_path, template_info, handler)

        print(f"Inference complete. Results saved to {output_file_path}")


if __name__ == "__main__":
    print("Arguments:", sys.argv)
    if len(sys.argv) not in (4, 5, 6, 7):
        print(
            "Usage: get_output_rpo.py mode <model_ix> <prompt_ix> <prompt_ix_end> <start_ix> <end_ix> or \
             get_model_outputs.py <model_ix> <prompt_ix>")
        raise Exception("Wrong number of arguments")
    assert not sys.argv[1].isdigit()
    main(sys.argv[1], *map(int, sys.argv[2:]))
