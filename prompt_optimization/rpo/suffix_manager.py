import gc
import os
import json
import math
import random
import time
from copy import deepcopy
from typing import Optional, Any
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM, MistralForCausalLM,
                          LlamaForCausalLM, GemmaForCausalLM)
from queue import Queue
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_embedding_layer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM) or \
            isinstance(model, MistralForCausalLM) or \
            isinstance(model, GemmaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    else:
        return model.model.embed_tokens


def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM) or isinstance(model, MistralForCausalLM) or isinstance(model,
                                                                                                    GemmaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        return model.model.embed_tokens.weight



def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM) or isinstance(model, MistralForCausalLM) or isinstance(model,
                                                                                                    GemmaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        return model.model.embed_tokens(input_ids)


def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)


"""
The AttackPrompt class is a comprehensive component designed to facilitate the generation, 
testing, and evaluation of attack prompts against language models (LMs), aiming to assess 
and improve their robustness to adversarial inputs. Here's an overview of its functionality
and key elements:
"""


class AttackPrompt(object):
    """
    A class used to generate an attack prompt. 
    """

    def __init__(self,
                 task_prompt,
                 data_prompt,
                 safe_target,
                 adv_target,
                 tokenizer,
                 conv_template,
                 model_name,
                 control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 *args, **kwargs
                 ):

        """
        Initializes the AttackPrompt object with the provided parameters.

        Parameters
        ----------
        goal : str
            The intended goal of the attack
        target : str
            The target of the attack
        tokenizer : Transformer Tokenizer
            The tokenizer used to convert text into tokens
        conv_template : Template
            The conversation template used for the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        """
        self.model_name = model_name

        self.task_prompt = task_prompt
        self.data_prompt = data_prompt
        self.safe_target = safe_target
        self.adv_target = adv_target
        self.control = control_init
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.conv_template.messages = []
        self.adv_target = adv_target

        self.test_new_toks = len(self.tokenizer(self.safe_target).input_ids) + 2  # buffer
        self._update_ids()

    def _update_ids(self):
        use_system_prompt = True
        suffix_size = None
        if "llama-2" in self.model_name:
            suffix_size = len(" [/INST]")
            assistant_chat_name = "assistant"
        elif "llama-3" in self.model_name:
            suffix_size = len("<|eot_id|>")
            assistant_chat_name = "assistant"
        elif "gemma" in self.model_name:
            suffix_size = len("<end_of_turn>\n")
            use_system_prompt = False
            assistant_chat_name = "model"
        elif "starling" in self.model_name:
            suffix_size = len("<|end_of_turn|>")
            use_system_prompt = False
            assistant_chat_name = "assistant"
        elif "zephyr" in self.model_name:
            suffix_size = len("</s>\n")
            assistant_chat_name = "assistant"
        elif "phi-3" in self.model_name:
            suffix_size = len("<|end|>\n<|assistant|>\n")
            assistant_chat_name = "assistant"
        else:
            raise Exception("Suffix size not implemented")
        assert suffix_size != 0  # if 0, then array[:-0] is empty array

        if use_system_prompt:
            messages = [{"role": "system", "content": self.task_prompt},
                        {"role": "user", "content": ""}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[
                  :-suffix_size]
            toks = self.tokenizer(msg).input_ids
            self._user_role_slice = slice(0, len(toks))

            messages = [{"role": "system", "content": self.task_prompt},
                        {"role": "user", "content": self.data_prompt}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[
                  :-suffix_size]
            toks = self.tokenizer(msg).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, len(toks))

            messages = [{"role": "system", "content": self.task_prompt},
                        {"role": "user", "content": f"{self.data_prompt} {self.control}"}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[
                  :-suffix_size]
            toks = self.tokenizer(msg).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            messages = [{"role": "system", "content": self.task_prompt},
                        {"role": "user", "content": f"{self.data_prompt} {self.control}"}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            toks = self.tokenizer(msg).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
            messages = [{"role": "system", "content": self.task_prompt},
                        {"role": "user", "content": f"{self.data_prompt} {self.control}"},
                        {"role": assistant_chat_name, "content": self.safe_target}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=False)# for zephyr [:-suffix_size]
            toks = self.tokenizer(msg).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            messages = [{"role": "user", "content": ""}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[
                  :-suffix_size]
            toks = self.tokenizer(msg).input_ids
            self._user_role_slice = slice(0, len(toks))

            messages = [{"role": "user", "content": f"{self.task_prompt} {self.data_prompt}"}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[
                  :-suffix_size]

            toks = self.tokenizer(msg).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, len(toks))

            messages = [{"role": "user", "content": f"{self.task_prompt} {self.data_prompt} {self.control}"}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[
                  :-suffix_size]

            toks = self.tokenizer(msg).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            messages = [{"role": "user", "content": f"{self.task_prompt} {self.data_prompt} {self.control}"}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            toks = self.tokenizer(msg).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
            messages = [{"role": "user", "content": f"{self.task_prompt} {self.data_prompt} {self.control}"},
                        {"role": assistant_chat_name, "content": self.safe_target}]
            msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            toks = self.tokenizer(msg).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)


        self.input_ids = torch.tensor(toks[:self._target_slice.stop], device="cuda")

    @torch.no_grad()
    def generate(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 16

        if gen_config.max_new_tokens > 32:
            print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        input_ids = self.input_ids[:self._assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id)[0]

        return output_ids[self._assistant_role_slice.stop:]

    def generate_str(self, model, gen_config=None):
        return self.tokenizer.decode(self.generate(model, gen_config))


    @torch.no_grad()
    def test_loss(self, model):
        logits, ids = self.logits(model, return_ids=True)
        return self.target_loss(logits, ids).mean().item()

    @torch.no_grad()
    def test_adv_loss(self, model):
        logits, ids = self.logits(model, return_ids=True)
        return self.adv_target_loss(logits, ids).mean().item()

    def grad(self, model):

        raise NotImplementedError("Gradient function not yet implemented")

    @torch.no_grad()
    def logits(self, model, test_controls=None, return_ids=False):

        pad_tok = -1
        if test_controls is None:
            test_controls = self.control_toks
        if isinstance(test_controls, torch.Tensor):
            if len(test_controls.shape) == 1:
                test_controls = test_controls.unsqueeze(0)
            test_ids = test_controls.to(model.device)
        elif not isinstance(test_controls, list):
            test_controls = [test_controls]
        elif isinstance(test_controls[0], str):
            max_len = self._control_slice.stop - self._control_slice.start
            test_ids = [
                torch.tensor(self.tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
                for control in test_controls
            ]
            pad_tok = 0
            while pad_tok in self.input_ids or any([pad_tok in ids for ids in test_ids]):
                pad_tok += 1
            nested_ids = torch.nested.nested_tensor(test_ids)
            test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
        else:
            raise ValueError(
                f"test_controls must be a list of strings or a tensor of token ids, got {type(test_controls)}")

        if not (test_ids[0].shape[0] == self._control_slice.stop - self._control_slice.start):
            raise ValueError((
                f"test_controls must have shape "
                f"(n, {self._control_slice.stop - self._control_slice.start}), "
                f"got {test_ids.shape}"
            ))

        locs = torch.arange(self._control_slice.start, self._control_slice.stop).repeat(test_ids.shape[0], 1).to(
            model.device)
        ids = torch.scatter(
            self.input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids
        )
        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None
        if return_ids:
            del locs, test_ids;
            gc.collect()
            return model(input_ids=ids, attention_mask=attn_mask).logits, ids
        else:
            del locs, test_ids
            logits = model(input_ids=ids, attention_mask=attn_mask).logits
            del ids;
            gc.collect()
            return logits

    def target_loss(self, logits, ids):
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(self._target_slice.start - 1, self._target_slice.stop - 1)
        loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, self._target_slice])
        return loss

    def control_loss(self, logits, ids):
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(self._control_slice.start - 1, self._control_slice.stop - 1)
        loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, self._control_slice])
        return loss

    def adv_target_loss(self, logits, ids):
        crit = nn.CrossEntropyLoss(reduction='none')
        original_target = self.target_str
        self.target_str = self.adv_target
        loss_slice = slice(self._target_slice.start - 1, self._target_slice.stop - 1)

        loss = crit(logits[:, loss_slice, :].transpose(1, 2),
                    ids[:, self._target_slice.start - 1:self._target_slice.stop - 1])

        self.target_str = original_target
        return loss

    @property
    def assistant_str(self):
        return self.tokenizer.decode(self.input_ids[self._assistant_role_slice]).strip()

    @property
    def assistant_toks(self):
        return self.input_ids[self._assistant_role_slice]

    @property
    def jailbreak_str(self):
        return self.tokenizer.decode(self.input_ids[self._jailbreak_slice]).strip()

    @jailbreak_str.setter
    def jailbreak_str(self, jailbreak):
        self.jailbreak = jailbreak
        self._update_ids()

    @property
    def adv_target_str(self):
        return self.adv_target

    @adv_target_str.setter
    def adv_target_str(self, adv_target):
        self.adv_target = adv_target

    @property
    def goal_str(self):
        return self.tokenizer.decode(self.input_ids[self._goal_slice]).strip()

    @goal_str.setter
    def goal_str(self, goal):
        self.goal = goal
        self._update_ids()

    @property
    def goal_toks(self):
        return self.input_ids[self._goal_slice]

    @property
    def target_str(self):
        return self.tokenizer.decode(self.input_ids[self._target_slice]).strip()

    @target_str.setter
    def target_str(self, target):
        self.target = target
        self._update_ids()

    @property
    def target_toks(self):
        return self.input_ids[self._target_slice]

    @property
    def control_str(self):
        return self.tokenizer.decode(self.input_ids[self._control_slice]).strip()

    @control_str.setter
    def control_str(self, control):
        self.control = control
        self._update_ids()

    @property
    def control_toks(self):
        return self.input_ids[self._control_slice]

    @control_toks.setter
    def control_toks(self, control_toks):
        self.control = self.tokenizer.decode(control_toks)
        self._update_ids()

    @property
    def prompt(self):
        return self.tokenizer.decode(self.input_ids[self._goal_slice.start:self._control_slice.stop])

    @property
    def input_toks(self):
        return self.input_ids

    @property
    def input_str(self):
        return self.tokenizer.decode(self.input_ids)

    @property
    def eval_str(self):
        return self.tokenizer.decode(self.input_ids[:self._assistant_role_slice.stop]).replace('<s>', '').replace(
            '</s>', '')


"""
The PromptManager class is responsible for overseeing a collection of attack prompts during the optimization process, facilitating operations like generation, testing, evaluation, and updates across multiple prompts simultaneously. Let's dissect its functionalities and structure:

PromptManager acts as a central control unit for managing and optimizing a suite of attack prompts, crucial for efficiently navigating the space of possible attacks and defenses in RPO. By aggregating operations across multiple prompts, it enables systematic and scalable optimization processes.
"""


class PromptManager(object):
    """A class used to manage the prompt during optimization."""

    def __init__(self,
                 train_tasks,
                 train_data_prompts,
                 train_target_outputs,
                 adv_train_targets,
                 tokenizer,
                 conv_template,
                 model_name,
                 control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 managers=None,
                 *args, **kwargs
                 ):
        """
        Initializes the PromptManager object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        tokenizer : Transformer Tokenizer
            The tokenizer used to convert text into tokens
        conv_template : Template
            The conversation template used for the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        """

        if len(train_tasks) != len(train_data_prompts) or len(train_data_prompts) != len(train_target_outputs):
            raise ValueError("Length of goals and targets must match")
        if len(train_tasks) == 0:
            raise ValueError("Must provide at least one goal, target pair")

        self.tokenizer = tokenizer
        self.adv_train_targets = adv_train_targets
        self._prompts = []

        for task_prompt, data_prompt, safe_target, adv_target in zip(train_tasks, train_data_prompts,
                                                                     train_target_outputs, adv_train_targets):
            prompt = managers['AP'](
                task_prompt=task_prompt,
                data_prompt=data_prompt,
                safe_target=safe_target,
                adv_target=adv_target,
                tokenizer=tokenizer,
                conv_template=conv_template,
                model_name=model_name,
                control_init=control_init,
            )
            self._prompts.append(prompt)
        self._nonascii_toks = get_nonascii_toks(tokenizer, device='cpu')

    def generate(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 64

        return [prompt.generate(model, gen_config) for prompt in self._prompts]

    def generate_str(self, model, gen_config=None):
        return [
            self.tokenizer.decode(output_toks)
            for output_toks in self.generate(model, gen_config)
        ]

    def test_loss(self, model):
        return [prompt.test_loss(model) for prompt in self._prompts]

    def grad(self, model):
        grad = [prompt.grad(model) for prompt in self._prompts]
        grad_len = [len(g) for g in grad]
        counts = np.bincount(grad_len)
        common_len = np.argmax(counts)
        edited_grads = [g for g in grad if len(g) == common_len]
        if len(edited_grads) < len(grad):
            print("Warning: inconsistent gradients length", grad_len)
        return sum(edited_grads)

    def logits(self, model, test_controls=None, return_ids=False):
        vals = [prompt.logits(model, test_controls, return_ids) for prompt in self._prompts]
        if return_ids:
            return [val[0] for val in vals], [val[1] for val in vals]
        else:
            return vals

    def target_loss(self, logits, ids):
        return torch.cat(
            [
                prompt.target_loss(logit, id).mean(dim=1).unsqueeze(1)
                for prompt, logit, id in zip(self._prompts, logits, ids)
            ],
            dim=1
        ).mean(dim=1)

    def control_loss(self, logits, ids):
        return torch.cat(
            [
                prompt.control_loss(logit, id).mean(dim=1).unsqueeze(1)
                for prompt, logit, id in zip(self._prompts, logits, ids)
            ],
            dim=1
        ).mean(dim=1)



    def sample_control(self, *args, **kwargs):

        raise NotImplementedError("Sampling control tokens not yet implemented")

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, i):
        return self._prompts[i]

    def __iter__(self):
        return iter(self._prompts)

    @property
    def control_str(self):
        return self._prompts[0].control_str

    @property
    def control_toks(self):
        return self._prompts[0].control_toks

    @control_str.setter
    def control_str(self, control):
        for prompt in self._prompts:
            prompt.control_str = control

    @control_toks.setter
    def control_toks(self, control_toks):
        for prompt in self._prompts:
            prompt.control_toks = control_toks

    @property
    def disallowed_toks(self):
        return self._nonascii_toks


"""
The MultiPromptAttack class is a sophisticated framework designed to orchestrate multiple prompt-based attacks, optimizing and evaluating their effectiveness in inducing or preventing specific behaviors in language models (LMs). This class integrates closely with the previously discussed components, like PromptManager and individual prompt functionalities, to conduct comprehensive attack simulations and optimizations. Here's an outline of its core functionalities and mechanisms:

MultiPromptAttack stands as a central orchestrator for conducting sophisticated, multi-faceted attacks on LMs, combining elements of prompt engineering, strategic exploration, and parallel processing to challenge and improve the robustness of models. By systematically managing and evolving a diverse set of attack prompts, it embodies the iterative, adversarial approach central to robust prompt optimization (RPO).
"""


class MultiPromptAttack(object):
    """A class used to manage multiple prompt-based attacks."""

    def __init__(self,
                 train_tasks,
                 train_data_prompts,
                 train_target_outputs,
                 adv_train_targets,
                 workers,
                 control_init,
                 logfile,
                 managers,
                 test_tasks,
                 test_data_prompts,
                 test_target_outputs,
                 adv_test_targets,
                 test_workers,
                 model_name,
                 *args, **kwargs
                 ):
        """
        Initializes the MultiPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        workers : list of Worker objects
            The list of workers used in the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list of str, optional
            The list of test goals of the attack
        test_targets : list of str, optional
            The list of test targets of the attack
        test_workers : list of Worker objects, optional
            The list of test workers used in the attack
        """
        self.model_name = model_name
        self.train_tasks = train_tasks
        self.train_data_prompts = train_data_prompts
        self.train_target_outputs = train_target_outputs
        self.adv_train_targets = adv_train_targets
        self.test_tasks = test_tasks
        self.test_data_prompts = test_data_prompts
        self.test_target_outputs = test_target_outputs
        self.adv_test_targets = adv_test_targets
        self.control_init = control_init
        self.workers = workers
        self.test_workers = test_workers
        self.models = [worker.model for worker in workers]
        self.logfile = logfile
        self.prompts = [
            managers['PM'](
                train_tasks,
                train_data_prompts,
                train_target_outputs,
                adv_train_targets,
                worker.tokenizer,
                worker.conv_template,
                self.model_name,
                control_init,
                managers
            )
            for worker in workers
        ]
        self.managers = managers

    @property
    def control_str(self):
        return self.prompts[0].control_str

    @control_str.setter
    def control_str(self, control):
        for prompts in self.prompts:
            prompts.control_str = control

    @property
    def control_toks(self):
        return [prompts.control_toks for prompts in self.prompts]

    @control_toks.setter
    def control_toks(self, control):
        if len(control) != len(self.prompts):
            raise ValueError("Must provide control tokens for each tokenizer")
        for i in range(len(control)):
            self.prompts[i].control_toks = control[i]

    def get_filtered_cands(self, worker_index, control_cand, filter_cand=True, curr_control=None):
        cands, count = [], 0
        worker = self.workers[worker_index]
        for i in range(control_cand.shape[0]):

            max_token_id = self.workers[0].tokenizer.vocab_size - 1
            if any(id > max_token_id or id < 0 for id in control_cand[i]):
                count += 1
                continue

            decoded_str = worker.tokenizer.decode(control_cand[i],
                                                  skip_special_tokens=True)
            if filter_cand:
                if decoded_str != curr_control and len(
                        worker.tokenizer(decoded_str, add_special_tokens=False).input_ids) <= len(control_cand[i]):
                    cands.append(decoded_str)
                else:
                    count += 1
            else:
                cands.append(decoded_str)
        if filter_cand:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        return cands

    def step(self, *args, **kwargs):

        raise NotImplementedError("Attack step function not yet implemented")

    def run(self,
            n_steps=100,
            batch_size=1024,
            topk=256,
            temp=1,
            allow_non_ascii=True,
            target_weight=None,
            control_weight=None,
            anneal=True,
            anneal_from=0,
            prev_loss=np.infty,
            # stop_on_success=True,
            test_steps=50,
            log_first=True,
            filter_cand=True,
            verbose=True,
            selection_interval=100
            ):

        def P(e, e_prime, k):
            T = max(1 - float(k + 1) / (n_steps + anneal_from), 1.e-7)
            return True if e_prime < e else math.exp(-(e_prime - e) / T) >= random.random()

        if target_weight is None:
            target_weight_fn = lambda _: 1
        elif isinstance(target_weight, (int, float)):
            target_weight_fn = lambda i: target_weight
        if control_weight is None:
            control_weight_fn = lambda _: 0.1
        elif isinstance(control_weight, (int, float)):
            control_weight_fn = lambda i: control_weight

        steps = 0
        loss = best_loss = 1e6
        best_control = self.control_str
        runtime = 0.

        for i in range(n_steps):

            steps += 1
            start = time.time()
            torch.cuda.empty_cache()
            control, loss = self.step(
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight_fn(i),
                control_weight=control_weight_fn(i),
                filter_cand=filter_cand,
                verbose=verbose
            )
            runtime = time.time() - start
            keep_control = True if not anneal else P(prev_loss, loss, i + anneal_from)
            if keep_control:
                self.control_str = control

            prev_loss = loss
            if loss < best_loss:
                best_loss = loss
                best_control = control
            print('Current Loss in MPA:', loss, 'Best Loss in MPA:', best_loss, "\n\n")

            for i in range(len(self.prompts)):
                print("curr output", self.prompts[i].generate_str(self.models[i]), "\n")

            if self.logfile is not None and (i + 1 + anneal_from) % test_steps == 0:


                model_tests = "Not relevant"
                self.log(i + 1 + anneal_from, n_steps + anneal_from, self.control_str, best_loss, runtime, model_tests,
                         verbose=verbose)


        return self.control_str, loss, steps

    def test(self, workers, prompts, include_loss=False):
        for j, worker in enumerate(workers):
            worker(prompts[j], "test", worker.model)
        model_tests = np.array([worker.results.get() for worker in workers])
        model_tests_jb = model_tests[..., 0].tolist()
        model_tests_mb = model_tests[..., 1].tolist()
        model_tests_loss = []
        if include_loss:
            for j, worker in enumerate(workers):
                worker(prompts[j], "test_loss", worker.model)
            model_tests_loss = [worker.results.get() for worker in workers]

        return model_tests_jb, model_tests_mb, model_tests_loss

    def test_all(self):
        all_workers = self.workers + self.test_workers
        all_prompts = [
            self.managers['PM'](
                self.train_tasks + self.test_tasks,
                self.train_data_prompts + self.test_data_prompts,
                self.train_target_outputs + self.test_target_outputs,
                self.adv_train_targets + self.adv_test_targets,
                worker.tokenizer,
                worker.conv_template,
                self.model_name,
                self.control_str,
                self.managers
            )
            for worker in all_workers
        ]
        return self.test(all_workers, all_prompts, include_loss=True)

    def parse_results(self, results):
        x = len(self.workers)
        i = len(self.train_data_prompts)
        id_id = results[:x, :i].sum()
        id_od = results[:x, i:].sum()
        od_id = results[x:, :i].sum()
        od_od = results[x:, i:].sum()
        return id_id, id_od, od_id, od_od

    def log(self, step_num, n_steps, control, loss, runtime, model_tests, verbose=True):

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        log['controls'].append(control)
        log['losses'].append(loss)
        log['runtimes'].append(runtime)

        with open(self.logfile, 'w') as f:
            json.dump(log, f, indent=4, cls=NpEncoder)
        if verbose:
            print((
                f"\n====================================================\n"
                f"Step {step_num:>4}/{n_steps:>4} ({runtime:.4} s)\n"
                # f"{output_str}"
                f"control='{control}'\n"
                f"====================================================\n"
            ))


"""
The ProgressiveMultiPromptAttack class extends the framework of MultiPromptAttack by introducing progressive strategies for both the goals and the models involved in the prompt-based attacks. This approach allows for a more dynamic and adaptable optimization process, adjusting the complexity and focus of the attack prompts as the attack progresses. Let's break down its key components and functionalities:

"""


class ProgressiveMultiPromptAttack(object):
    """A class used to manage multiple progressive prompt-based attacks."""

    def __init__(self,
                 train_loader,
                 test_loader,
                 workers,
                 model_name,
                 progressive_goals=True,
                 progressive_models=True,
                 control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 logfile=None,
                 managers=None,
                 test_workers=[],
                 *args, **kwargs
                 ):

        """
        Initializes the ProgressiveMultiPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        workers : list of Worker objects
            The list of workers used in the attack
        progressive_goals : bool, optional
            If true, goals progress over time (default is True)
        progressive_models : bool, optional
            If true, models progress over time (default is True)
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list of str, optional
            The list of test goals of the attack
        test_targets : list of str, optional
            The list of test targets of the attack
        test_workers : list of Worker objects, optional
            The list of test workers used in the attack
        """
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.workers = workers
        self.test_workers = test_workers
        self.progressive_goals = progressive_goals
        self.progressive_models = progressive_models
        self.control = control_init
        self.logfile = logfile
        self.managers = managers
        self.mpa_kwargs = ProgressiveMultiPromptAttack.filter_mpa_kwargs(**kwargs)

        if logfile is not None:
            with open(logfile, 'w+') as f:
                json.dump({
                    'params': {
                        'progressive_goals': progressive_goals,
                        'progressive_models': progressive_models,
                        'control_init': control_init,
                        'models': [
                            {
                                'model_path': self.model_name,
                                'tokenizer_path': self.model_name,
                                'conv_template': worker.conv_template.name
                            }
                            for worker in self.workers
                        ],
                        'test_models': [
                            {
                                'model_path': self.model_name,
                                'tokenizer_path': self.model_name,
                                'conv_template': worker.conv_template.name
                            }
                            for worker in self.test_workers
                        ]
                    },
                    'controls': [],
                    'losses': [],
                    'runtimes': [],
                    'tests': []
                }, f, indent=4
                )

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    def run(self,
            n_epochs: int = 1,
            batch_size: int = 1024,
            topk: int = 256,
            temp: float = 1.,
            allow_non_ascii: bool = False,
            target_weight=None,
            control_weight=None,
            anneal: bool = True,
            test_steps: int = 50,
            incr_control: bool = True,
            stop_on_success: bool = True,
            verbose: bool = True,
            filter_cand: bool = True,
            selection_interval: int = 100
            ):
        """
        Executes the progressive multi prompt attack.

        Parameters
        ----------
        n_steps : int, optional
            The number of steps to run the attack (default is 1000)
        batch_size : int, optional
            The size of batches to process at a time (default is 1024)
        topk : int, optional
            The number of top candidates to consider (default is 256)
        temp : float, optional
            The temperature for sampling (default is 1)
        allow_non_ascii : bool, optional
            Whether to allow non-ASCII characters (default is False)
        target_weight
            The weight assigned to the target
        control_weight
            The weight assigned to the control
        anneal : bool, optional
            Whether to anneal the temperature (default is True)
        test_steps : int, optional
            The number of steps between tests (default is 50)
        incr_control : bool, optional
            Whether to increase the control over time (default is True)
        stop_on_success : bool, optional
            Whether to stop the attack upon success (default is True)
        verbose : bool, optional
            Whether to print verbose output (default is True)
        filter_cand : bool, optional
            Whether to filter candidates whose lengths changed after re-tokenization (default is True)
        """

        if self.logfile is not None:
            print("logfile", self.logfile)
            with open(self.logfile, 'r') as f:
                log = json.load(f)

            log['params']['n_epochs'] = n_epochs
            log['params']['test_steps'] = test_steps
            log['params']['batch_size'] = batch_size
            log['params']['topk'] = topk
            log['params']['temp'] = temp
            log['params']['allow_non_ascii'] = allow_non_ascii
            log['params']['target_weight'] = target_weight
            log['params']['control_weight'] = control_weight
            log['params']['anneal'] = anneal
            log['params']['incr_control'] = incr_control
            log['params']['stop_on_success'] = stop_on_success
            log['params']['selection_interval'] = selection_interval

            with open(self.logfile, 'w') as f:
                json.dump(log, f, indent=4, cls=NpEncoder)
        num_workers = 1 if self.progressive_models else len(self.workers)
        step = 0
        stop_inner_on_success = self.progressive_goals
        loss = np.infty

        test_batch = next(iter(self.test_loader))

        print(f"Len(train_loader) = {len(self.train_loader)}")
        for epoch in range(n_epochs):
            for i, train_batch in enumerate(tqdm(self.train_loader, desc="Train batch")):
                print(f"Starting step:{i}, CUDA:{torch.cuda.current_device()}, control={self.control}")

                attack = self.managers['MPA'](

                    train_batch["task_prompt"],
                    train_batch["data_prompt"],
                    train_batch["targets"],
                    train_batch["adv_targets"],
                    self.workers[:num_workers],
                    self.control,
                    self.logfile,
                    self.managers,
                    test_batch["task_prompt"],
                    test_batch["data_prompt"],
                    test_batch["targets"],
                    test_batch["adv_targets"],
                    self.test_workers,
                    self.model_name,
                    **self.mpa_kwargs
                )

                control, loss, inner_steps = attack.run(
                    n_steps=1,  # self.steps_per_data_batch,#n_steps - step,
                    batch_size=batch_size,
                    topk=topk,
                    temp=temp,
                    allow_non_ascii=allow_non_ascii,
                    target_weight=target_weight,
                    control_weight=control_weight,
                    anneal=anneal,
                    anneal_from=step,
                    prev_loss=loss,
                    # stop_on_success=stop_inner_on_success,
                    test_steps=test_steps,
                    filter_cand=filter_cand,
                    verbose=verbose,
                    selection_interval=selection_interval
                )

                step += inner_steps

                self.control = control
                print(f"Step:{i}, CUDA:{torch.cuda.current_device()}, control={self.control}")

                if num_workers < len(self.workers):
                    num_workers += 1
                    loss = np.infty
                elif num_workers == len(self.workers) and stop_on_success:
                    model_tests = attack.test_all()
                    attack.log(step, n_epochs, self.control, loss, 0., model_tests, verbose=verbose)
                    break
                else:
                    if isinstance(control_weight, (int, float)) and incr_control:
                        if control_weight <= 0.09:
                            control_weight += 0.01
                            loss = np.infty
                            if verbose:
                                print(f"Control weight increased to {control_weight:.5}")
                        else:
                            stop_inner_on_success = False

        return self.control, step


class ModelWorker(object):

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs
        ).to("cuda")  #
        self.model.eval()

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.results = Queue()

    @staticmethod
    def execute_task(ob, fn, *args, **kwargs):

        if fn == "grad":
            with torch.enable_grad():
                return ob.grad(*args, **kwargs)
        else:
            with torch.no_grad():
                if fn == "logits":
                    return ob.logits(*args, **kwargs)
                elif fn == "contrast_logits":
                    return ob.contrast_logits(*args, **kwargs)
                elif fn == "test":
                    return "Not relevant"  # or ob.test(*args, **kwargs)
                elif fn == "test_loss":
                    return ob.test_loss(*args, **kwargs)
                else:
                    return fn(*args, **kwargs)

    def __call__(self, ob, fn, *args, **kwargs):
        res = ModelWorker.execute_task(deepcopy(ob), fn, *args, **kwargs)
        self.results.put(res)
        return self


def get_workers(params):
    tokenizers = []
    for i in range(len(params.tokenizer_paths)):
        tokenizer = AutoTokenizer.from_pretrained(
            params.tokenizer_paths[i],
            trust_remote_code=True,
            **params.tokenizer_kwargs[i]
        )

        if 'llama' in params.tokenizer_paths[i]:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        tokenizers.append(tokenizer)


    raw_conv_templates = [
        get_conversation_template(template)
        for template in params.conversation_templates
    ]
    conv_templates = []
    for conv in raw_conv_templates:
        if conv.name == 'zero_shot':
            conv.roles = tuple(['### ' + r for r in conv.roles])
            conv.sep = '\n'
        elif conv.name == 'llama-2':
            conv.sep2 = conv.sep2.strip()
        conv_templates.append(conv)


    workers = [
        ModelWorker(
            params.model_paths[i],
            params.model_kwargs[i],
            tokenizers[i],
            conv_templates[i]
        )
        for i in range(len(params.model_paths))
    ]


    num_train_models = getattr(params, 'num_train_models', len(workers))
    print('Loaded {} train models'.format(num_train_models))
    print('Loaded {} test models'.format(len(workers) - num_train_models))

    return workers[:num_train_models], workers[num_train_models:]


def process_train_data(data):
    outputs = {}
    for k in data[0].keys():
        outputs[k] = []
    for elem in data:
        for k, v in elem.items():
            outputs[k].append(v)
    return outputs


class CustomDataset(Dataset):
    def __init__(self, tasks, data_prompts, targets, adv_targets):
        self.tasks = tasks
        self.data_prompts = data_prompts
        self.targets = targets
        self.adv_targets = adv_targets

    def __len__(self):
        return len(self.tasks)  # Assuming all lists are of the same length

    def __getitem__(self, idx):
        return {
            'task_prompt': self.tasks[idx],
            'data_prompt': self.data_prompts[idx],
            'targets': self.targets[idx],
            'adv_targets': self.adv_targets[idx]
        }


def restrict_len(array, lim=200):
    assert isinstance(array, list)
    return [elem[:lim] for elem in array]


def get_goals_and_targets(params):
    train_tasks = []
    train_data_prompts = []
    train_targets = []
    adv_train_targets = []
    test_tasks = []
    test_data_prompts = []
    test_targets = []
    adv_test_targets = []

    if params.train_data:
        with open(params.train_data, "r") as f:
            train_data = process_train_data(json.load(f))
        train_targets = restrict_len(train_data['goal_safe'][params.n_test_data:])
        train_tasks = train_data['system_prompt'][params.n_test_data:]
        train_data_prompts = train_data['data_prompt_instructed'][params.n_test_data:]
        adv_train_targets = restrict_len(train_data['goal_unsafe'][params.n_test_data:])
        if params.n_test_data > 0:
            test_targets = restrict_len(train_data['goal_safe'][:params.n_test_data])
            test_tasks = restrict_len(train_data['system_prompt'][:params.n_test_data])
            test_data_prompts = train_data['data_prompt_instructed'][:params.n_test_data]
            adv_test_targets = train_data['goal_unsafe'][:params.n_test_data]
    assert len(train_tasks) == len(train_data_prompts)
    assert len(train_targets) == len(train_data_prompts)
    assert len(adv_train_targets) == len(train_data_prompts)

    assert len(test_tasks) == len(test_targets)
    assert len(test_data_prompts) == len(test_targets)
    assert len(adv_test_targets) == len(test_targets)
    train_dataset = CustomDataset(train_tasks, train_data_prompts, train_targets, adv_train_targets)
    test_dataset = CustomDataset(test_tasks, test_data_prompts, test_targets, adv_test_targets)
    train_loader = DataLoader(train_dataset, batch_size=params.data_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params.data_batch_size, shuffle=False)
    print('Loaded {} train goals'.format(len(train_targets)))
    print('Loaded {} test goals'.format(len(test_targets)))

    return train_loader, test_loader

def print_gpu_memory(label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for all operations to complete
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - allocated_memory
        print(f"Memory log for cuda:{torch.cuda.current_device()}. Label::: {label}:")
        print(f"Total GPU Memory: {total_memory / 1e9:.2f} GB")
        print(f"Allocated Memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Cached Memory: {cached_memory / 1e9:.2f} GB")
        print(f"Free Memory: {free_memory / 1e9:.2f} GB")
    else:
        print("No CUDA device available")


