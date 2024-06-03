import json
import sys
import random
import numpy as np
from utils import load_config, load_json_data, read_file

from typing import Dict, Any, List, Tuple


def flatten_dataset(dataset: Dict[str, Any]) -> List[Dict]:
    """
    Flattens a structured dataset into a list of aggregated subtask data.

    This function traverses a nested dictionary structure, aggregating the data found in subtasks. Each aggregated
    subtask data entry is enhanced with its task type before being added to the resulting list.

    Parameters:
    - dataset (dict): The input dataset containing nested dictionaries of tasks and subtasks.

    Returns:
    - list: A list of dictionaries, each containing aggregated data from subtasks
            and their associated task type.
    """
    aggregated_data = []
    for task_type, tasks in dataset.items():
        if task_type == "descr":
            continue
        for task_name, task_ds in tasks.items():
            if task_name == "descr":
                continue
            subtasks = task_ds["subtasks"]
            for subtask_ds in subtasks:
                for base_data in subtask_ds["data"]:
                    aggregated_data.append({
                        "system_prompt": subtask_ds["system_prompt"],
                        "clean_prompt": base_data,
                        "info": {
                            "subtask_name": subtask_ds["name"],
                            "task_domain": task_type,
                            "general_task": task_name,
                            "task_descr": subtask_ds["description"]
                        }
                    })
    return aggregated_data


def assemble_probe_dataset(base_data_ds: List[Dict[str, Any]],
                           probes: List[Dict[str, str]],
                           appended_types: Tuple[str, str, str, str] = ("ll", "lr", "rl", "rr")) -> List[
    Dict[str, Any]]:
    """
    Assembles a dataset by appending probes to base data entries according to specified patterns.

    Parameters:
        base_data_ds (List[Dict[str, Any]]): The base dataset containing system and clean prompts.
        probes (List[Dict[str, str]]): A list of probes, each containing an instruction and an answer.
        appended_types (Tuple[str, str, str, str], optional): Tuple containing the patterns for appending probes to the base data.
            Each pattern is a two-character string where the first character ('l' or 'r') indicates the position (left or right)
            of the probe instruction relative to the system prompt, and the second character indicates its position relative to the clean prompt.

    Returns:
        List[Dict[str, Any]]: The new dataset with probes appended according to the specified patterns.
    """
    new_dataset = []

    for i, base_data in enumerate(base_data_ds):
        try:
            appended_id = np.random.randint(len(probes))  # i % 100
            appended_type = appended_types[np.random.randint(len(appended_types))]
            system_prompt_instruction = (probes[appended_id]["instruction"] + " " + base_data["system_prompt"]
                                         if appended_type[0] == "l" else
                                         base_data["system_prompt"] + " " + probes[appended_id]["instruction"])

            prompt_instruction = (probes[appended_id]["instruction"] + " " + base_data["clean_prompt"]
                                  if appended_type[1] == "l" else
                                  base_data["clean_prompt"] + " " + probes[appended_id]["instruction"])

            new_dataset.append({
                "system_prompt_clean": base_data["system_prompt"],
                "prompt_instructed": prompt_instruction,
                "system_prompt_instructed": system_prompt_instruction,
                "prompt_clean": base_data["clean_prompt"],
                "witness": probes[appended_id]["answer"],
                "info": dict(**base_data["info"], **{
                    "appended_task_id": appended_id,
                        "appended_type": appended_type,
                        "is_insistent": appended_id >= 50
                })
            })
        except Exception as e:
            print(f"Error assembling dataset entry: {e}")
    return new_dataset


def insert_probes(data_input_path: str, probes_input_path: str, output_path: str,
                  do_shuffle: bool = False) -> None:
    """
    Inserts probes into a dataset, optionally shuffles the dataset, and saves it to a file.

    Parameters:
        data_input_path (str): The file path to the input data JSON.
        probes_input_path (str): The file path to the probes JSON.
        output_path (str): The file path where the modified dataset with probes should be saved.
        do_shuffle (bool, optional): If True, shuffles the dataset before saving. Defaults to False.

    This function processes the input dataset by flattening it and then appending probe data
    to each entry based on the provided probes. The resultant dataset can optionally be shuffled
    to randomize the order of entries before being saved to the specified output file.
    """
    probes = load_json_data(probes_input_path)
    data = load_json_data(data_input_path)["output"]
    data = flatten_dataset(data)

    if do_shuffle:
        random.shuffle(data)
    full_dataset = assemble_probe_dataset(data, probes)
    with open(output_path, "w") as f:
        json.dump(full_dataset, f)


if __name__ == "__main__":
    config = load_config(sys.argv)
    input_path = config["raw_data_path"]
    output_path = config["assembled_data_path"]
    probes_path = config["probes_path"]
    insert_probes(input_path, probes_path, output_path)
