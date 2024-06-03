import json
import sys

from typing import Dict, Union, Any, Optional, List


def load_config(argv: List[str], default_config_path: str = './source/sep_config.json' ) -> Dict:
    """
    Loads configuration settings from a JSON file.
    Gets

    Parameters:
    - argv (List[str]): Script arguments
    - default_config_path (str): The path to the configuration JSON file.

    Returns:
    - Dict: The loaded configuration settings.
    """
    if len(argv) > 2:
        print(
            "Usage: get_model_outputs.py ... or get_model_outputs.py <config_path> ...")
        sys.exit(1)
    config_path = argv[1] if len(argv) == 2 else None
    if config_path:
        config = load_json_data(config_path)
    else:
        config = load_json_data(default_config_path)
    return config


def read_file(file_path: str) -> str:
    """
    Reads and returns the content of a text file.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - Str: Contents of the file
    """
    with open(file_path, "r") as file:
        return file.read()


def load_json_data(file_path: str) -> Union[Dict, List]:
    """
    Loads and returns data from a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - Union[Dict, List]: The loaded json.

    """
    with open(file_path, "r", encoding='utf-8') as file:
        return json.load(file)


def reduce_subtasks(ds: Union[dict, list, str], max_subtasks: Optional[int] = 10) -> Any:
    """
    Recursively reduces the number of subtasks in each leaf of a hierarchical tree of subtask to a specified maximum.

    Parameters:
    - ds (Union[dict, list, str]): The hierarchical structure containing subtasks.
    - max_subtasks (Optional[int]): The maximum number of subtasks to retain in each leaf. If None, no reduction is applied.

    Returns:
    - Any: The modified hierarchical structure with the number of subtasks limited at each leaf.
    """
    if max_subtasks is None:
        return ds

    if isinstance(ds, str):
        return ds

    if isinstance(ds, list):
        return ds[:max_subtasks]

    if isinstance(ds, dict):
        if isinstance(next(iter(ds.values()), []), list):
            return {key: value[:max_subtasks] for key, value in ds.items()}
        else:
            return {key: reduce_subtasks(value, max_subtasks) for key, value in ds.items()}

    raise TypeError(f"Input type should be Union[dict, list, str], received {type(ds)}")
