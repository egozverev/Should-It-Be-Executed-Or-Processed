import os.path

import fire
import pandas as pd


def main(data_path: str, model_type: str, out_dir: str) -> None:
    """
    Process the input data, generate text based on the specified model type, and save the 
    resulting datasets into training and testing sets.

    Parameters:
    data_path (str): The path to the input JSON file containing the data.
    model_type (str): The type of model to be used for generating text. This should be one 
                      of the keys in the `map_funcs` dictionary.
    out_dir (str): The directory where the output JSON files will be saved.

    The function performs the following steps:
    1. Reads the input JSON file into a pandas DataFrame.
    2. Retains only the 'inputs', 'goal_safe', and 'goal_unsafe' columns.
    3. Extracts 'prompt_1' and 'prompt_2' from the 'inputs' column and removes 'inputs'.
    4. Applies a text generation function based on the specified `model_type`.
    5. Splits the data into training (80%) and testing (20%) sets.
    6. Prints two random samples from the training set.
    7. Saves the training and testing sets as JSON files in the specified output directory.

    Raises:
    KeyError: If `model_type` is not one of the predefined keys in the `map_funcs` dictionary.
    """    df = pd.read_json(data_path)
    # remove columns except prompt_1 prompt_2 goal_safe goal_unsafe
    df = df[['inputs', 'goal_safe', 'goal_unsafe']]
    df['prompt_1'] = df['inputs'].apply(lambda x: x['prompt_1'])
    df['prompt_2'] = df['inputs'].apply(lambda x: x['prompt_2'])
    df = df.drop(columns=['inputs'])

    map_funcs = {
        "gemma-1.1": lambda x: x['prompt_1'] + x['goal_safe'] + "<end_of_turn>",
        "Starling-LM": lambda x: x['prompt_1'] + x['goal_safe'] + "<|end_of_turn|>",
        "Llama-3": lambda x: x['prompt_1'] + x['goal_safe'] + "<|eot_id|>",
        "Llama-2": lambda x: x['prompt_1'] + x['goal_safe'] + "</s>",
        "zephyr": lambda x: x['prompt_1'] + x['goal_safe'] + "</s>",
        "Phi-3": lambda x: x['prompt_1'] + x['goal_safe'] + "<|end|>",
    }

    df['text'] = df.apply(map_funcs[model_type], axis=1)

    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)

    #print a few random samples
    print(df_train['text'].sample(2).to_list())

    os.makedirs(out_dir, exist_ok=True)
    df_train.to_json(f'{out_dir}/train_dataset.json',
                     orient='records',
                     force_ascii=False)
    df_test.to_json(f'{out_dir}/test_dataset.json',
                    orient='records',
                    force_ascii=False)


if __name__ == '__main__':
    fire.Fire(main)
