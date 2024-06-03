import os
import openai
import json
import sys
from tqdm import tqdm
sys.path.append("../..")
from iclr_workshop_version.openai_utils import get_messages_generic, call_openai_api, try_processing_json_str
from utils import load_config, load_json_data, read_file

from typing import Dict


def generate_data(input_path: str, output_path: str, prompt_path: str) -> None:
    """
    Generates data based on system prompts.

    Parameters:
        input_path (str): The path to the input JSON file containing tasks, subtasks and system prompts.
        output_path (str): The path to save the output JSON file with generated data.
        prompt_path (str): The path to the text file containing the generation prompt.
    """
    gen_prompt = read_file(prompt_path)
    data = load_json_data(input_path)["output"]

    exp_log = {
        "input_message": gen_prompt,
        "data": data,
        "output": {}
    }
    for task_type, tasks in data.items():
        if task_type == "descr":
            continue  # Skip description at root level
        print(f"Processing type {task_type}\n\n")
        exp_log["output"][task_type] = {"descr": tasks.get("descr", "")}
        for task, elem in tasks.items():
            print(f"Dealing with task: {task}")
            if not tasks.get("descr"):
                print(f"WARNING: Missing description for {task_type}, {task}")
            if task == "descr":
                continue
            subtasks = elem.get("subtasks", [])
            # Sometimes ChatGPT generates {subtasks: {subtasks: [...]}}
            if isinstance(subtasks, dict):
                subtasks = subtasks["subtasks"]
            outputs = generate_data_for_subtasks(gen_prompt, subtasks, task)
            exp_log["output"][task_type][task] = {"descr": tasks.get("descr", ""), "subtasks": outputs}

            with open(output_path, "w") as f:
                json.dump(exp_log, f)
    print(f"Output saved to {output_path}")


def generate_data_for_subtasks(gen_prompt: str, subtasks: list[Dict], task_descr: str,
                               n_attempts: int = 3) -> list:
    """
    Generates data for each subtask using OpenAI's API.
    API is called n_attempts times, call results are stacked.

    Parameters:
        gen_prompt (str): The general prompt to be appended before each subtask's specific info.
        subtasks (list[Dict]): A list of subtasks for which to generate data.
        task_descr (str): Description of the task, used for logging.
        n_attempts (int): Number of attempts to try generating data for a subtask.

    Returns:
        list: A list of generated responses for the subtasks.
    """
    outputs = []
    for subtask in tqdm(subtasks, desc=f"Processing subtasks for {task_descr}"):
        cur_prompt = f"{gen_prompt}\n {json.dumps(subtask)}"
        messages = get_messages_generic(cur_prompt)
        for _ in range(n_attempts):  # Try up to 3 times for a valid response
            response = call_openai_api(messages)
            processed_response = try_processing_json_str(response, 'dict')
            if processed_response:
                outputs.append(processed_response)
            else:
                print(f"Failed to get response for subtask: {subtask}")
    return outputs


# input_path = "./task_descr_step4_short_pt3.json"
# output_path = "./task_data_step5_shortsys_mid_pt3.json"
#
# promt_path = "./generate_data_prompt-mid.txt"

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    config = load_config(sys.argv)
    input_path = config["subtasks_sys_path"]
    output_path = config["raw_data_path"]
    prompt_path = config["sys_to_data_prompt_path"]
    generate_data(input_path, output_path, prompt_path)
