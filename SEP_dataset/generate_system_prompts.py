import json
import sys
import os
import openai
sys.path.append("..")
from openai_utils import get_messages_generic, call_openai_api, try_processing_json_str
from utils import load_config, load_json_data, read_file, reduce_subtasks


def generate_system_prompts(input_path: str, output_path: str, prompt_path: str,
                            cut_subtasks: bool = True, subtask_limit: int = 10) -> None:
    """
    Generates system prompts from subtasks data, optionally limits the number of subtasks per task.

    Parameters:
    - input_path (str): Path to the input JSON file.
    - output_path (str): Path where the output JSON file will be saved.
    - prompt_path (str): Path to the text file containing the generation prompt for API calls.
    - cut_subtasks (bool): Flag to determine whether to cut down the number of subtasks before proceeding.
    - subtask_limit (int): The maximum number of subtasks to retain if cut_subtasks is True.

    The function processes each task type and task in the input data, generating system prompts for each subtasks.
    """
    gen_prompt = read_file(prompt_path)
    data = load_json_data(input_path)["output"]
    if cut_subtasks:
        data = reduce_subtasks(data, subtask_limit)

    exp_log = {
        "input_message": gen_prompt,
        "data": data,
        "output": {}
    }

    for task_type, tasks in data.items():
        if task_type == "descr":
            continue
        print(f"Processing type {task_type}\n\n")

        exp_log["output"][task_type] = {}
        descr = ""
        for task, subtasks in tasks.items():
            if task == "descr":
                exp_log["output"][task_type]["descr"] = tasks[task]  # not really subtasks
                descr = tasks[task]
                continue
            print(f"Dealing with task: {task}")

            if not descr:
                print(f"WARNING: len(descr)==0 for {task_type, task}")
            cur_input = {
                task: {
                    "descr": descr,
                    "subtasks": subtasks
                }
            }
            cur_prompt = gen_prompt + f"\n {json.dumps(cur_input)}"

            messages = get_messages_generic(cur_prompt)
            response = None

            while response is None:
                response = call_openai_api(messages)
                response = try_processing_json_str(response, "dict")
            exp_log["output"][task_type].update(response)
            with open(output_path, "w+") as f:
                json.dump(exp_log, f)


# input_path = "./task_descr_step3_v2.json"
# output_path = "./task_descr_step4_short_pt3.json"
# promt_path = "./create_system_prompts_short.txt"

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    config = load_config(sys.argv)
    input_path = config["subtasks_path"]
    output_path = config["subtasks_sys_path"]
    prompt_path = config["subtasks_to_sys_prompt_path"]
    generate_system_prompts(input_path, output_path, prompt_path)
