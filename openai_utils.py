import openai
import random
import time
import json

def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        max_retries: int = 50
) -> callable:
    """
    Decorator to retry a function with exponential backoff and optional jitter.

    Parameters:
        func (callable): The function to apply the retry mechanism.
        initial_delay (float): Initial delay between retries in seconds.
        exponential_base (float): The base of the exponent for delay calculation.
        jitter (bool): If True, adds random jitter to the delay to avoid thundering herd problem.
        max_retries (int): Maximum number of retries before giving up.

    Returns:
        callable: A wrapper function that applies the retry mechanism.
    """

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        last_exception = None

        while num_retries < max_retries:
            try:
                return func(*args, **kwargs)
            except openai.error.OpenAIError as e:  # Adjust based on actual retry-worthy exceptions
                print(f"Retry {num_retries + 1} due to exception: {e}")
                last_exception = e
                num_retries += 1
                adjusted_delay = delay * (exponential_base ** num_retries)
                if jitter:
                    sleep_time = adjusted_delay + (random.random() * adjusted_delay)
                else:
                    sleep_time = adjusted_delay
                time.sleep(sleep_time)

        raise Exception(f"Maximum number of retries ({max_retries}) exceeded. Last exception: {last_exception}")

    return wrapper


# Example usage
@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    """Function wrapper to apply retry mechanism to OpenAI's ChatCompletion.create call."""
    return openai.ChatCompletion.create(**kwargs)


def process_gen_output(raw_output: str, expected_type: str) -> str:
    """
    Extracts the JSON-formatted string from the raw output of a language model.

    Parameters:
    - raw_output (str): The raw output string from a language model which may include
      JSON data surrounded by additional text.
    - expected_type (str): Whether the output should be a dict or list.

    Returns:
    - str: The extracted JSON-formatted string. If the expected characters are not found,
      an empty string is returned which may not be valid JSON.
    """
    assert expected_type in ("list", "dict"), "Expected type should be either 'list' or 'dict'"
    left_border = "[" if expected_type == "list" else "{"
    right_border = ["]"] if expected_type == "list" else "}"
    fst = raw_output.find(left_border)
    snd = raw_output.rfind(right_border)
    output = raw_output[fst:snd + 1] if fst != -1 and snd != -1 else ""
    return output


def try_processing_json_str(raw_str: str, expected_type: str) -> dict:
    """
    Attempts to process a JSON-formatted string and return the corresponding Python dictionary.

    This function tries to parse a string that is expected to be in JSON format after processing
    it to ensure it is valid JSON. If the processing or parsing fails, it catches the exception
    and prints an error message.

    Parameters:
    - raw_str (str): The raw string that needs to be processed and parsed.
    - expected_type (str): Whether the output should be a dict or list.

    Returns:
    - dict: A Python dictionary obtained from parsing the processed JSON string. If parsing fails,
            it returns an empty dictionary.

    Note:
    - This implementation assumes that `process_gen_output` returns a string that should be a valid
      JSON after processing. Adjustments might be needed based on the actual behavior of
      `process_gen_output`.
    """
    try:
        processed_str = process_gen_output(raw_str, expected_type)
        return json.loads(processed_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
    except Exception as e:
        print(f"Caught exception during processing: {e}")
    return {}



def get_messages_generic(cur_prompt: str) -> list:
    """
    Creates system and user messages for the API request based on the current prompt.
    System prompt is set to a generic message.

    Parameters:
        cur_prompt (str): The current prompt to append to the generic system message.

    Returns:
        list: A list of dictionaries representing the system and user messages.
    """
    return [
        {'role': "system",
         "content": "As a state-of-the-art AI, ChatGPT, your primary objective is to handle user requests with maximum efficiency and versatility. You are expected to quickly understand and accurately interpret a wide range of inquiries, ranging from simple factual questions to complex problem-solving tasks. Your responses should be concise yet comprehensive, prioritizing relevant information and omitting unnecessary details. You must adapt to the context and tone of each request, providing tailored and thoughtful solutions. Additionally, you should employ your advanced capabilities to offer creative and innovative insights where appropriate, while always adhering to ethical guidelines and maintaining user privacy. Your goal is to deliver high-quality, reliable, and user-friendly assistance, making each interaction a positive and informative experience."},
        {"role": "user", "content": cur_prompt}
    ]


def call_openai_api(messages: list, model: str = "gpt-4-1106-preview", max_tokens: int = 4096, temperature: float = 0.9) -> str:
    """
    Calls the OpenAI API with specified messages and returns the response content.

    Parameters:
        messages (list): The list of messages to send to the model in ChatML format.
        model (str): The model identifier to use for the completion (one of ChatGPT models).
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): Controls randomness in the generation process.

    Returns:
        str: The content of the response from the OpenAI API.
    """
    try:
        response = completions_with_backoff(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Caught exception while calling OpenAI API: {e}")
        return ""


def get_task_outputs(messages: list, max_subtasks: int = 30) -> list:
    """
    Generates a list of subtasks for a given task using the model's completions with backoff strategy

    Parameters:
        messages (list): The list of messages to send to the model.

    Returns:
        list: A list of generated subtasks for the given task.
    """
    outputs = []
    while len(outputs) < max_subtasks:
        try:
            response = completions_with_backoff(
                model="gpt-4-1106-preview",
                messages=messages,
                max_tokens=4096,
                temperature=0.9
            )
            response_content = response['choices'][0]['message']['content']
            outputs.extend(json.loads(process_gen_output(response_content)))
        except Exception as e:
            print(f"Caught exception: {e}")
            break  # Consider breaking or handling the error differently.
    return outputs
