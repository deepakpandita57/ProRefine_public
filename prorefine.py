# %%
import os
import json
import requests
import concurrent
from tqdm import tqdm
import numpy as np


import torch
import textgrad as tg
from textgrad.tasks import load_task


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


from dotenv import load_dotenv

load_dotenv(override=True)

# %%
# LLAMA 3.1 8B
# endpoint_url = "endpoint_url"
# endpoint_model_name = "meta/llama-3.1-8b-instruct"


# LLAMA 3.1 70B
endpoint_url = "endpoint_url"
endpoint_model_name = "meta/llama-3.1-70b-instruct"

headers = {"headers here"}


def get_response_from_endpoint(endpoint_url, endpoint_model_name, headers, messages):
    json_data = {
        "model": endpoint_model_name,
        "messages": messages,
        "max_tokens": 2048,
        "stream": False,
    }
    response = requests.post(endpoint_url, headers=headers, json=json_data)
    try:
        response_json = json.loads(response.text)
        response_text = (
            response_json.get("choices", [{}])[0].get("message", {}).get("content")
        )
        return response_text
    except:
        print(response)
        return ""


# %%
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from textgrad.engine.base import EngineLM, CachedEngine


class LLamaModel(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        endpoint_url,
        headers,
        endpoint_model_name="meta/llama-3.1-70b-instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ):
        """
        :param endpoint_model_name:
        :param system_prompt:
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_llama_{endpoint_model_name}.db")
        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        if os.getenv("API_KEY") is None:
            raise ValueError(
                "Please set the API_KEY environment variable if you'd like to use the LLama model."
            )

        self.endpoint_url = endpoint_url
        self.headers = headers
        self.endpoint_model_name = endpoint_model_name

    def generate(
        self,
        prompt,
        system_prompt=None,
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        json_data = {
            "model": self.endpoint_model_name,
            "messages": [
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2048,
            "stream": False,
        }
        response = requests.post(
            self.endpoint_url, headers=self.headers, json=json_data
        )
        try:
            response_json = json.loads(response.text)
            response = (
                response_json.get("choices", [{}])[0].get("message", {}).get("content")
            )
        except:
            print(response)
            response = ""
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


# %% [markdown]
# **Requirements:**
#
# * You need to have an API key to run this. This should be set as an environment variable as API_KEY.
#

# %%
eval_endpoint_url = "eval_endpoint_url"
eval_endpoint_model_name = "meta/llama-3.1-70b-instruct"
llm_api_eval = LLamaModel(
    endpoint_url=eval_endpoint_url,
    headers=headers,
    endpoint_model_name=eval_endpoint_model_name,
)

# llm_api_eval = tg.get_engine(engine_name="")

# task_name = "BBH_object_counting"
# task_name = "BBH_word_sorting"
task_name = "GSM8K_DSPy"

train_set, val_set, test_set, eval_fn = load_task(
    task_name, evaluation_api=llm_api_eval
)

print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()
prompt = STARTING_SYSTEM_PROMPT
print(STARTING_SYSTEM_PROMPT)

# %%
input_str = val_set[0][0]
label_str = val_set[0][1]
print(input_str, label_str)

# %% [markdown]
# ### Prepare model


# %%
def extract_prediction(text, model_name):
    if "llama" in model_name:
        splitted_text = text.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(splitted_text) > 1:
            label = splitted_text[1].split("<|eot_id|>")[0].strip()
        else:
            label = text
    else:
        label = text
    return label


# %%
def prepare_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        offload_buffers=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    return (tokenizer, model)


# %% [markdown]
# ### Call LLM


# %%
def call_LLM(
    messages,
    model,
    tokenizer,
    model_name,
    device="cuda",
    max_new_tokens=500,
):
    # messages = feedback_tokenizer.apply_chat_template(
    #     [
    #         {"role": "system", "content": prompt},
    #         {"role": "user", "content": f"Input: {input_str}"},
    #         {"role": "user", "content": f"Output: {output_str}"},
    #     ],
    #     tokenize=False,
    # )
    encodeds = tokenizer.encode(
        messages, add_special_tokens=False, return_tensors="pt", padding=True
    )
    attention_mask = (encodeds != tokenizer.pad_token_id).long()

    model_inputs = encodeds.to(device)
    attention_mask = attention_mask.to(device)
    model.to(device)

    generated_ids = model.generate(
        model_inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=False,
        # top_k=50,
        # top_p=0.95,
        top_p=None,
        temperature=None,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded = tokenizer.batch_decode(generated_ids)
    output = extract_prediction(decoded[0], model_name)
    return output


# %%
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# # model_name = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer_llm_call, model_llm_call = prepare_model(model_name)

# %% [markdown]
# ### Eval zero-shot


# %%
def prepare_input(input_str, system_prompt, tokenizer=None):
    if tokenizer is not None:
        text_input = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input: {input_str}"},
            ],
            tokenize=False,
        )
    else:
        text_input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {input_str}"},
        ]
    return text_input


# %%
# text_request = prepare_input(input_str, prompt, tokenizer_llm_call)
# output = call_LLM(text_request, model_llm_call, tokenizer_llm_call, model_name)
# print(output)

# %%
text_request = prepare_input(input_str, prompt)
output = get_response_from_endpoint(
    endpoint_url, endpoint_model_name, headers, text_request
)
print(output)

# %%
# llm_api_test = tg.get_engine(engine_name="")
# # llm_api_test = llm_api_eval

# system_prompt = tg.Variable(prompt,
#                             requires_grad=True,
#                             role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
# model = tg.BlackboxLLM(llm_api_test, system_prompt)

# %%
# type(model)==tg.model.BlackboxLLM, type(model_llm_call)==transformers.models.llama.modeling_llama.LlamaForCausalLM


# %%
def eval_sample(
    item,
    eval_fn,
    model,
    prompt=None,
    tokenizer=None,
    model_name=None,
    use_endpoint=False,
    fn_response=[],
):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(
        x, requires_grad=False, role_description="query to the language model"
    )
    y = tg.Variable(
        str(y), requires_grad=False, role_description="correct answer for the query"
    )
    if use_endpoint:
        text_request = prepare_input(x, prompt)
        response = get_response_from_endpoint(
            endpoint_url, endpoint_model_name, headers, text_request
        )
        response = tg.Variable(
            response,
            requires_grad=False,
            role_description="response from the language model",
        )
    else:
        if type(model) == tg.model.BlackboxLLM:
            response = model(x)
            # print(response)
        else:
            text_request = prepare_input(x, prompt, tokenizer)
            response = call_LLM(text_request, model, tokenizer, model_name)
            response = tg.Variable(
                response,
                requires_grad=False,
                role_description="response from the language model",
            )
            # print(response)
    fn_response.append(response)
    try:
        eval_output_variable = eval_fn(
            inputs=dict(prediction=response, ground_truth_answer=y)
        )
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


# %%
# eval_sample(val_set[0], eval_fn, model_llm_call, prompt, tokenizer_llm_call, model_name)
# eval_sample(val_set[5], eval_fn, model)
# eval_sample(val_set[0], eval_fn, model=None, prompt=prompt, use_endpoint=True)

# %%
# def eval_dataset(test_set, eval_fn, model, prompt=None, tokenizer=None, model_name=None, max_samples: int=None):
#     if max_samples is None:
#         max_samples = len(test_set)
#     accuracy_list = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
#         futures = []
#         for _, sample in enumerate(test_set):

#             future = executor.submit(eval_sample, sample, eval_fn, model, prompt, tokenizer, model_name)
#             futures.append(future)
#             if len(futures) >= max_samples:
#                 break
#         tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
#         for future in tqdm_loader:
#             # print(future)
#             acc_item = future.result()
#             accuracy_list.append(acc_item)
#             tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
#     return accuracy_list

# %%
# results = {"test_acc": [], "prompt": [], "validation_acc": []}
# # results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
# # results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))

# results["test_acc"].append(eval_dataset(test_set, eval_fn, model_llm_call, prompt, tokenizer_llm_call, model_name))
# results["validation_acc"].append(eval_dataset(val_set, eval_fn, model_llm_call, prompt, tokenizer_llm_call, model_name))


# %% [markdown]
# ### Get Feedback


# %%
def format_feedback_request(input_str, output_str, system_prompt, tokenizer=None):
    if tokenizer is not None:
        feedback_request = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input: {input_str}"},
                {"role": "user", "content": f"Output: {output_str}"},
            ],
            tokenize=False,
        )
    else:
        feedback_request = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {input_str}"},
            {"role": "user", "content": f"Output: {output_str}"},
        ]
    return feedback_request


# %%

FEEDBACK_SYSTEM_PROMPT = """"You are a smart language model that evaluates the output of a language model for a given input.\n
You do not propose new output, only evaluate the given output critically, think step-by-step, and give very concise feedback to improve the output. 
Ensure your feedback is correct and factual.\n
This is very important, if the output is correct do not provide any feedback, respond with 'the output is correct'.\n
Give your response by sending the feedback only. The text you send will be used directly.\n\n"""

# %% [markdown]
# ### Prompt Optimization


# %%
def format_prompt_optimization_request(
    feedback, task_prompt, system_prompt, tokenizer=None
):
    if tokenizer is not None:
        optimization_request = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task prompt: {task_prompt}"},
                {"role": "user", "content": f"Feedback: {feedback}"},
            ],
            tokenize=False,
        )
    else:
        optimization_request = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task prompt: {task_prompt}"},
            {"role": "user", "content": f"Feedback: {feedback}"},
        ]
    return optimization_request


# %%
OPTIMIZATION_SYSTEM_PROMPT = """"You are part of an optimization system that improves the input prompt given to a large language model.\n
You are given a feedback for the model output. Your goal is to use this feedback to improve the input prompt. 
The feedback may be noisy, identify what is important and what is correct. Pay attention to the constraints mentioned in the input prompt.\n
This is very important. You MUST make sure that the improved prompt does not deviate substantially from the input prompt and is generalizable for the task. 
If the input prompt cannot be improved further, your response should be the input prompt as is.\n
Think step-by-step and send the improved prompt between tags <IMPROVED_PROMPT> improved prompt </IMPROVED_PROMPT>. 
The text you send between the tags will be used directly to replace the system prompt for a large language model.\n\n
"""

# %%
# feedback_request = format_feedback_request(
#     input_str, label_str, FEEDBACK_SYSTEM_PROMPT, tokenizer_llm_call
# )
# feedback = call_LLM(feedback_request, model_llm_call, tokenizer_llm_call, model_name)
# print(feedback)

# %%
feedback_request = format_feedback_request(input_str, label_str, FEEDBACK_SYSTEM_PROMPT)
feedback = get_response_from_endpoint(
    endpoint_url, endpoint_model_name, headers, feedback_request
)
print(feedback)

# %%
# optimization_request = format_prompt_optimization_request(
#     feedback,
#     prompt,
#     OPTIMIZATION_SYSTEM_PROMPT,
#     tokenizer_llm_call,
# )
# optimized_prompt = call_LLM(optimization_request, model_llm_call, tokenizer_llm_call, model_name)
# print(optimized_prompt)

# %%
optimization_request = format_prompt_optimization_request(
    feedback,
    prompt,
    OPTIMIZATION_SYSTEM_PROMPT,
)
optimized_prompt = get_response_from_endpoint(
    endpoint_url, endpoint_model_name, headers, optimization_request
)
print(optimized_prompt)


# %%
def extract_text_between_tags(source_string, tag):
    if source_string:
        splitted = source_string.split(f"<{tag}>")
        if len(splitted) > 1:
            text = splitted[1].split(f"</{tag}>")[0].strip()
            return text
        else:
            return source_string
    return None


# %%
print(extract_text_between_tags(optimized_prompt, "IMPROVED_PROMPT"))

# %% [markdown]
# ### Our Method

# %%
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

device = "cuda"
total_tokens_to_generate = 250
tokens_per_step = 10


tokenizer, model = prepare_model(model_name)
model.to(device)


def prorefine(prompt, input_str, use_chat_history=False):
    # Iteratively generate tokens
    # ids = model_inputs.clone()
    cur_tokens = 0
    output_str = ""
    chat_history = []
    full_history = []
    for _ in range(total_tokens_to_generate // tokens_per_step):
        history = {}
        history["prompt"] = prompt

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Input: {input_str}"},
        ]

        if use_chat_history:
            chat_history.extend(messages)
            text_input = tokenizer.apply_chat_template(chat_history, tokenize=False)
        else:
            text_input = tokenizer.apply_chat_template(messages, tokenize=False)

        encodeds = tokenizer.encode(
            text_input, add_special_tokens=False, return_tensors="pt", padding=True
        )
        attention_mask = (encodeds != tokenizer.pad_token_id).long()

        model_inputs = encodeds.to(device)
        attention_mask = attention_mask.to(device)

        eos_token_id = tokenizer.eos_token_id

        generated_ids = model.generate(
            model_inputs,
            attention_mask=attention_mask,
            max_new_tokens=cur_tokens + tokens_per_step,
            num_return_sequences=1,
            do_sample=False,
            # top_k=50,
            # top_p=0.95,
            top_p=None,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
        )

        cur_tokens += tokens_per_step
        new_tokens = generated_ids[:, model_inputs.size(1) :]

        decoded = tokenizer.batch_decode(generated_ids)
        output_str = extract_prediction(decoded[0], model_name)
        # print(output_str)
        history["output"] = output_str

        # feedback_request = format_feedback_request(
        #     input_str, output_str, FEEDBACK_SYSTEM_PROMPT, tokenizer_llm_call
        # )
        # feedback = call_LLM(
        #     feedback_request, model_llm_call, tokenizer_llm_call, model_name
        # )
        feedback_request = format_feedback_request(
            input_str, output_str, FEEDBACK_SYSTEM_PROMPT
        )
        feedback = get_response_from_endpoint(
            endpoint_url, endpoint_model_name, headers, feedback_request
        )
        # print(feedback)
        history["feedback"] = feedback

        # optimization_request = format_prompt_optimization_request(
        #     feedback,
        #     prompt,
        #     OPTIMIZATION_SYSTEM_PROMPT,
        #     tokenizer_llm_call,
        # )
        # optimized_prompt = call_LLM(
        #     optimization_request, model_llm_call, tokenizer_llm_call, model_name
        # )
        optimization_request = format_prompt_optimization_request(
            feedback, prompt, OPTIMIZATION_SYSTEM_PROMPT
        )
        optimized_prompt = get_response_from_endpoint(
            endpoint_url, endpoint_model_name, headers, optimization_request
        )
        # print(optimized_prompt)

        prompt = extract_text_between_tags(optimized_prompt, "IMPROVED_PROMPT")
        history["optimized_prompt"] = prompt
        full_history.append(history)

        if use_chat_history:
            chat_history.append(
                {"role": "assistant", "content": f"Output: {output_str}"}
            )
            chat_history.append({"role": "user", "content": f"Feedback: {feedback}"})

        if eos_token_id in new_tokens:
            # print("EOS token encountered. Stopping generation.")
            break

        # break
    return prompt, output_str, full_history


# %%
# prompt, output_str = prorefine(prompt, input_str)
# print(prompt)
# print(output_str)

# %%
# eval_sample(val_set[0], eval_fn, model, prompt, tokenizer, model_name)


# %%
def eval_method(sample, eval_fn, model, prompt, tokenizer, model_name):
    x, y = sample
    improved_prompt, _ = prorefine(prompt, x)
    acc_item = eval_sample(
        sample, eval_fn, model, improved_prompt, tokenizer, model_name
    )
    return acc_item


# %%
# eval_method(test_set[35], eval_fn, model, prompt, tokenizer, model_name)

# %%
print(prompt)


# %%
def eval_dataset(
    test_set,
    eval_fn,
    model,
    prompt=None,
    tokenizer=None,
    model_name=None,
    max_samples: int = None,
):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for _, sample in enumerate(test_set):

            future = executor.submit(
                eval_method, sample, eval_fn, model, prompt, tokenizer, model_name
            )
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), position=0
        )
        for future in tqdm_loader:
            # print(future)
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


# %%
# results = {"test_acc": [], "prompt": [], "validation_acc": []}
# # # results["test_acc"].append(eval_dataset(test_set, eval_fn, model))

# results["test_acc"].append(eval_dataset(test_set, eval_fn, model, prompt, tokenizer, model_name))

# %% [markdown]
# ### LLM Eval


# %%
def format_llm_eval_request(question, prediction, system_prompt, tokenizer=None):
    if tokenizer is not None:
        eval_request = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "user", "content": f"Prediction: {prediction}"},
            ],
            tokenize=False,
        )
    else:
        eval_request = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "user", "content": f"Prediction: {prediction}"},
        ]
    return eval_request


EVALUATION_SYSTEM_PROMPT = "Below is a question from a question-answering task and reasoning with the final prediction. Is the final prediction correct? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"

# %%
eval_request = format_llm_eval_request(
    input_str,
    output,
    EVALUATION_SYSTEM_PROMPT,
)
llm_eval = get_response_from_endpoint(
    eval_endpoint_url, eval_endpoint_model_name, headers, eval_request
)
print(llm_eval)
print(extract_text_between_tags(llm_eval, "ACCURACY"))


# %%
use_chat_history = False
# use_chat_history = True
output_data = []
accuracy_list = []
test_tqdm = tqdm(range(len(test_set)))
# test_tqdm = tqdm(range(5))
for i in test_tqdm:
    x, y = test_set[i]

    json_output = {}
    json_output["input"] = x
    json_output["label"] = str(y)
    json_output["initial_prompt"] = prompt

    initial_response = []
    init_acc_item = eval_sample(
        test_set[i],
        eval_fn,
        model,
        prompt,
        tokenizer,
        model_name,
        fn_response=initial_response,
    )
    if initial_response:
        json_output["initial_response"] = initial_response[0].value
    json_output["initial_output_is_correct"] = init_acc_item

    eval_request = format_llm_eval_request(
        x,
        initial_response[0].value,
        EVALUATION_SYSTEM_PROMPT,
    )
    llm_eval = get_response_from_endpoint(
        eval_endpoint_url, eval_endpoint_model_name, headers, eval_request
    )
    llm_eval_text = extract_text_between_tags(llm_eval, "ACCURACY")
    llm_acc_item = 0
    if llm_eval_text and llm_eval_text == "1":
        llm_acc_item = 1
    json_output["initial_output_is_correct_llm"] = llm_acc_item

    # if (init_acc_item == llm_acc_item):
    #     accuracy_list.append(1)
    # else:
    #     accuracy_list.append(0)

    if llm_acc_item == 0:
        improved_prompt, output_str, history = prorefine(prompt, x, use_chat_history)
        json_output["final_prompt"] = improved_prompt
        json_output["history"] = history
        final_response = []
        acc_item = eval_sample(
            test_set[i],
            eval_fn,
            model,
            improved_prompt,
            tokenizer,
            model_name,
            fn_response=final_response,
        )
        if final_response:
            json_output["final_response"] = final_response[0].value
        json_output["final_output_is_correct"] = acc_item
        accuracy_list.append(acc_item)
    else:
        accuracy_list.append(init_acc_item)

    test_tqdm.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    output_data.append(json_output)
    # break

# %%
print(np.mean(accuracy_list))

# %%
len(output_data)

# %%
base_path = "output"
out_path = os.path.join(base_path, f"{endpoint_model_name}/{model_name}")
isExist = os.path.exists(out_path)
if not isExist:
    os.makedirs(out_path)
if use_chat_history:
    output_path_json = os.path.join(
        out_path, f"prorefine_llm_eval_with_chat_history_{task_name}.json"
    )
else:
    output_path_json = os.path.join(out_path, f"prorefine_llm_eval_{task_name}.json")

# %%
json_string = json.dumps(output_data, indent=4)

with open(output_path_json, "w") as outfile:
    outfile.write(json_string)

# %%
