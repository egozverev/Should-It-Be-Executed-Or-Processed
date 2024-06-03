from get_model_outputs import inference, load_config, load_data
from get_model_outputs import ModelAPIHandler
from typing import Optional

import os
import sys
import json


def main(mode: str, model_ix: int, checkpoint_ix: int, checkpoint_ix_end: Optional[int], start_ix: Optional[int] = None,
         end_ix: Optional[int] = None) -> None:
    """
    Executes the model inference process based on specified command line arguments.

    Parameters:
        mode (str): Either "train" for saving results in train folder, or "eval" for saving in eval folder
        model_ix (int): Index to select the model configuration.
        checkpoint_ix (int): Index to select the model's checkpoint.
        start_ix (Optional[int]): Start index for slicing the dataset, or None to start from the beginning.
        end_ix (Optional[int]): End index for slicing the dataset, or None to go till the end of the dataset.
    """
    assert mode in ("ft", "fteval"), "Wrong mode"
    config = load_config()
    model_type = config["model_types"][model_ix]
    model_name = config["models"][model_ix]
    ckeckpoint_dir = config["checkpoints_path"]
    if checkpoint_ix == checkpoint_ix_end:
        raise Exception("Checkpoint index interval is empty")
    if checkpoint_ix_end is None:
        checkpoint_ix_end = checkpoint_ix + 1
    checkpoint_names_path = f"./model_eval/ft_checkpoints/{model_type}.json"
    for c_ix in range(checkpoint_ix, checkpoint_ix_end):
        input_path = config["train_input_path"] if (mode == "ft") else config["eval_input_path"]
        with open(input_path, 'r') as f:
            dataset = json.load(f)
        with open(checkpoint_names_path, "r") as f:
            checkpoint_name = json.load(f)[c_ix]
        if start_ix is None:
            start_ix = 0
        if end_ix is None:
            end_ix = len(dataset)
        if mode == "ft":
            output_dir_path = os.path.join(config["output_base_path"], mode, model_type, f"prompt_ft_{checkpoint_name}")
        else:
            assert mode == "fteval"
            output_dir_path = os.path.join(config["output_base_path"], "eval", model_type,
                                           f"prompt_ft_{checkpoint_name}")

        os.makedirs(output_dir_path, exist_ok=True)
        output_file_path = os.path.join(output_dir_path, f"{start_ix}-{end_ix}.json")
        dataset = dataset[start_ix: end_ix]
        checkpoint_info = {
            "ckeckpoint_ix": c_ix,
            "checkpoint_name": checkpoint_name,
            "template_prompt": {
                "system": "{}",
                "main": "{}",
            },
        }
        checkpoint_path = os.path.join(ckeckpoint_dir, checkpoint_name)
        handler = ModelAPIHandler(model_name, model_type, mode, 0, checkpoint_path)
        print(f"Starting inference for model {model_name} on checkpoint index {c_ix}. \
              Dataset slice is dataset[{start_ix}:{end_ix}]")
        inference(dataset, output_file_path, checkpoint_info, handler)

        print(f"Inference complete. Results saved to {output_file_path}")


if __name__ == "__main__":
    print("Arguments:", sys.argv)
    if len(sys.argv) not in (4, 5, 6, 7):
        print(
            "Usage: get_output_ft.py mode <model_ix> <checkpoint_ix> <checkpoint_ix_end> <start_ix> <end_ix>")
        raise Exception("Wrong number of arguments")
    assert not sys.argv[1].isdigit()
    main(sys.argv[1], *map(int, sys.argv[2:]))
