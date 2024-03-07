import json
import sys
import os

sys.path.append("..")
from openai_utils import get_messages_generic, call_openai_api, try_processing_json_str
from utils import load_config

import openai



def get_task_outputs(messages: list, max_subtasks: int = 30) -> list:
    """
    Generates subtsask for a given task by calling the OpenAI API and processing the response.
    The prompt should describe to the model how it is to convert a general task into a JSON list of subtasks.

    Parameters:
        messages (list): A message in ChatML format
        max_subtasks (int): The maximum number of subtasks to generate for the given task.

    Returns:
        list: A list of generated subtasks for the given task.
    """
    outputs = []
    while len(outputs) < max_subtasks:
        response_content = call_openai_api(messages)
        if not response_content:
            continue
        try:
            processed_output = try_processing_json_str(response_content, "list")
            outputs.extend(processed_output)
        except Exception as e:
            # Try again. Error is usually a failure to find correct JSON list in the output string.
            print(f"Caught exception while processing the API response: {e}")
    return outputs


def process_tasks(input_path: str, output_path: str, prompt_path: str) -> None:
    """
    Expands tasks based on the types defined in the input file using prompts,
    and saves the expanded tasks with descriptions to the output file.

    Note that the list of subtasks has to be reviewed (manually or automatically) to delete the repetitions.

    Parameters:
        input_path (str): Path to the input JSON file with task types.
        output_path (str): Path to save the output JSON file with expanded tasks.
        prompt_path (str): Path to the text file containing the expansion prompt.
    """
    with open(prompt_path, "r") as f:
        expand_prompt = f.read()

    with open(input_path, "r") as f:
        data = json.load(f)

    exp_log = {
        "input_message": expand_prompt,
        "data": data,
        "output": []
    }

    new_data = {}
    for task_type in data.keys():
        print(f"Dealing with type: {task_type}\n\n")
        if task_type == "descr":
            new_data[task_type] = data[task_type]
            continue
        new_data[task_type] = {}
        for task, text in data[task_type].items():
            print(f"Dealing with task: {task}")
            if task == "descr":
                new_data[task_type][task] = text
                continue

            cur_prompt = f"{expand_prompt} Primary Task: {task}\nDescription: {text}"
            messages = get_messages_generic(cur_prompt)
            outputs = get_task_outputs(messages)

            new_data[task_type][task] = {
                "descr": text,
                "subtasks": outputs
            }

            exp_log['output'] = new_data
            with open(output_path, "w") as f:
                json.dump(exp_log, f)


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    config = load_config(sys.argv)
    input_path = config["task_types_path"]
    output_path = config["subtasks_path"]
    prompt_path = config["task_to_subtasks_prompt_path"]
    process_tasks(input_path, output_path, prompt_path)
