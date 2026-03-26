import time
import os
import json
import re
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from .prompts import init_system_prompt, plan_prompt, question_prefix, subquestion_prefix, overall_question_prefix, answer_prefix, reward_prompt, new_subquestion_prefix, reward_prefix, reward_prompt_for_answer, reward_prefix_for_answer

from transformers import AutoTokenizer
try:
    _vrest_tokenizer = AutoTokenizer.from_pretrained(
        "/data00/mohan/ckpt/Qwen3_VL_4B_Instruct",
        trust_remote_code=True
    )
except Exception as e:
    print(f"[vrest] tokenizer load failed, counting disabled: {e}")
    _vrest_tokenizer = None


# def _count_messages_tokens(messages):
#     if _vrest_tokenizer is None:
#         return 0
#     parts = []
#     for m in messages:
#         content = m.get("content", "")
#         if isinstance(content, list):
#             for c in content:
#                 if isinstance(c, dict) and c.get("type") == "text":
#                     parts.append(c.get("text", ""))
#         else:
#             parts.append(str(content))
#     text = "\n".join(parts)
#     return len(_vrest_tokenizer(text, return_tensors="pt").input_ids[0])

# This should be at module scope, near your tokenizer cache
def _count_messages_tokens(messages, model=None):
    """
    Count text tokens via tokenizer, and image tokens via model.count_image_tokens (if provided).
    Falls back to text-only if model lacks the helper.
    """
    # 1) text tokens
    if _vrest_tokenizer is None:
        text_tokens = 0
    else:
        parts = []
        for m in messages or []:
            content = m.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
            else:
                parts.append(str(content))
        text = "\n".join(parts)
        try:
            text_tokens = len(_vrest_tokenizer(text, return_tensors="pt").input_ids[0])
        except Exception:
            text_tokens = 0

    # 2) image tokens (dynamic, from the model if available)
    image_tokens = 0
    if model is not None and hasattr(model, "count_image_tokens"):
        try:
            image_tokens = int(model.count_image_tokens(messages))
        except Exception:
            image_tokens = 0
    
    # print("Image tokens", image_tokens)

    return int(text_tokens) + int(image_tokens)


def _count_text_tokens(text):
    if _vrest_tokenizer is None or text is None:
        return 0
    return len(_vrest_tokenizer(text, return_tensors="pt").input_ids[0])

def generate_message(query, image_url, text_only=False, assistant_prompt=None):
    if text_only:
        messages=[
            {"role": "system", "content": init_system_prompt},
            {
                "role": "user",
                "content": query,
            },
        ]
    else:
        messages=[
            {"role": "system", "content": init_system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
                {"type": "text", "text": query},
            ],
            },
        ]
    if assistant_prompt is not None and assistant_prompt != "":
        messages.append({"role": "assistant", "content": assistant_prompt})
    return messages


# def default_call(cfg, messages, model, default_response, task_name="get response", n=1, continue_final_message=False, stop_token=None):
#     for attempt in range(cfg.prompt_method.attempt_num):
#         try:
#             response = model.get_response(messages=messages, json_format=None, n=n, continue_final_message=continue_final_message, stop_token=stop_token)
#             if "response" in response:
#                 response = response["response"]
#             return response
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             print(f"The model response was:\n {response}")
#             if attempt == cfg.prompt_method.attempt_num-1:
#                 print(f"Failed to {task_name} after {cfg.prompt_method.attempt_num} attempts. Error: {str(e)}")
#                 return default_response

def default_call(
    cfg,
    messages,
    model,
    default_response,
    task_name="get response",
    n=1,
    continue_final_message=False,
    stop_token=None,
    tracker=None,   # NEW
):
    # we can pre-count prompt tokens
    prompt_tokens = _count_messages_tokens(messages, model=model)

    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_response(
                messages=messages,
                json_format=None,
                n=n,
                continue_final_message=continue_final_message,
                stop_token=stop_token
            )
            # ORIGINAL BEHAVIOR:
            # if model returns {"response": ...}, unwrap to response
            if "response" in response:
                raw = response["response"]
            else:
                raw = response

            # counting (does NOT change what we return)
            if tracker is not None:
                # normalize to list for counting
                outs = raw if isinstance(raw, list) else [raw]
                completion_tokens = sum(_count_text_tokens(o) for o in outs)
                tracker["student_calls"] = tracker.get("student_calls", 0) + 1
                tracker["input_tokens"] = tracker.get("input_tokens", 0) + prompt_tokens
                tracker["output_tokens"] = tracker.get("output_tokens", 0) + completion_tokens
                tracker["total_tokens"] = tracker.get("total_tokens", 0) + prompt_tokens + completion_tokens
                tracker.setdefault("calls", []).append(
                    {
                        "type": task_name,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "n": len(outs),
                    }
                )

            # return exactly what original code returned
            return raw
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.attempt_num - 1:
                print(
                    f"Failed to {task_name} after {cfg.prompt_method.attempt_num} attempts. Error: {str(e)}"
                )
                return default_response


# def reward_call(cfg, messages, model):
#     for attempt in range(cfg.prompt_method.attempt_num):
#         try:
#             response = model.get_reward(messages=messages)
#             if "yes_prob" in response:
#                 response = response["yes_prob"]
#             return response
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             print(f"The model response was:\n {response}")
#             if attempt == cfg.prompt_method.attempt_num-1:
#                 print(f"Failed to get reward after {cfg.prompt_method.attempt_num} attempts. Error: {str(e)}")
#                 return 0.0

def reward_call(cfg, messages, model, tracker=None):
    prompt_tokens = _count_messages_tokens(messages, model=model)

    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_reward(messages=messages)
            # ORIGINAL: if it has yes_prob, return that value
            if "yes_prob" in response:
                value = response["yes_prob"]
            else:
                value = response

            if tracker is not None:
                tracker["student_calls"] = tracker.get("student_calls", 0) + 1
                tracker["input_tokens"] = tracker.get("input_tokens", 0) + prompt_tokens
                tracker["total_tokens"] = tracker.get("total_tokens", 0) + 1 + prompt_tokens
                tracker.setdefault("calls", []).append(
                    {
                        "type": "reward",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 1,
                        "n": 1,
                    }
                )

            return value
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.attempt_num - 1:
                print(
                    f"Failed to get reward after {cfg.prompt_method.attempt_num} attempts. Error: {str(e)}"
                )
                return 0.0


def decompose_question_call(cfg, image_url, model, question, sub_questions, sub_answers, last_sub_question=False, tracker=None):
    user_prompt = plan_prompt + question_prefix.format(question=question)
    assistant_prompt = ""
    for i, sub_question in enumerate(sub_questions):
        assistant_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
        assistant_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}\n\n"
    if last_sub_question:
        assistant_prompt += overall_question_prefix.format(step=len(sub_questions)+1, overall_question='')
    else:
        assistant_prompt += subquestion_prefix.format(step=len(sub_questions)+1, subquestion='').strip()
    stop_token = subquestion_prefix.format(step=len(sub_questions)+2, subquestion='').strip()
    messages = generate_message(user_prompt, image_url, assistant_prompt=assistant_prompt)
    responses = default_call(cfg, messages, model, "", "decompose question", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=stop_token, tracker=tracker) # list
    responses = [assistant_prompt + response for response in responses]
    return responses

def simple_cot_call(cfg, image_url, model, question, n=1):
    user_prompt = plan_prompt + question_prefix.format(question=question)
    assistant_prompt = ""
    assistant_prompt += subquestion_prefix.format(step=1, subquestion='').strip()
    messages = generate_message(user_prompt, image_url, assistant_prompt=assistant_prompt)
    return default_call(cfg, messages, model, "", "simple cot", n=n, continue_final_message=True)

def extract_final_answer(response):
    try:
        final_answer = re.findall(r"Answer \d+: (.*)", response)[-1]
    except:
        print(f"Error: Failed to extract final answer!")
        final_answer = response
    return final_answer

def extract_final_answer_word(response):
    if "answer is" in response:
        final_answer = response.split("answer is")[-1].strip(". \n")
    else:
        final_answer = response
    return final_answer

def majority_vote(answers):
    answer = Counter(answers).most_common(1)[0][0]
    index = answers.index(answer)
    return answer, index

def extract_sub_questions_and_answers(response, step):
    sub_question_pattern = subquestion_prefix.format(step=step, subquestion="").strip()
    sub_answer_pattern = answer_prefix.format(step=step, answer="").strip()
    # 找到pattern后的那个subquestion
    response = response.strip()
    try:
        sub_question = response.split(sub_question_pattern)[1].split("\n")[0].strip()
        sub_answer = response.split(sub_question_pattern)[1].split("\n")[-1].strip()
        # 将"Answer %d:"替换为空
        sub_answer = re.sub(r"Answer \d+:", "", sub_answer).strip()
    except:
        print(f"Error: {response}")
        sub_question, sub_answer = "", ""
    return sub_question, sub_answer

def get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer, tracker=None):
    user_prompt = reward_prompt_for_answer + question_prefix.format(question=question)
    for i, sub_question in enumerate(sub_questions):
        user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
        user_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}"
    user_prompt += f"{subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}\n"
    user_prompt += f"{answer_prefix.format(step=len(sub_questions)+1, answer=curr_sub_answer)}"
    user_prompt += reward_prefix_for_answer
    return reward_call(cfg, generate_message(user_prompt, image_url), model, tracker=tracker)

def get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question, tracker=None):
    user_prompt = reward_prompt + question_prefix.format(question=question) + "\n"
    for i, sub_question in enumerate(sub_questions):
        user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
    user_prompt += f"{new_subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}"
    user_prompt += f"{reward_prefix}"
    return reward_call(cfg, generate_message(user_prompt, image_url), model, tracker=tracker)

def get_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer, tracker=None):
    reward_subquestion = get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question, tracker=tracker)
    if is_terminal_question(question, curr_sub_question):
        reward_answer = get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer, tracker=tracker)
    else:
        reward_answer = 0.5
    return reward_subquestion, reward_answer

def get_reward_parallel(cfg, image_url, model, question, sub_questions, sub_answers, candidate_sub_questions, candidate_sub_answers, tracker=None):
    
    # 串行版本
    candidate_sub_questions_rewards = []
    for i, sub_question in enumerate(candidate_sub_questions):
        reward_subquestion, reward_answer = get_reward(cfg, image_url, model, question, sub_questions, sub_answers, sub_question, candidate_sub_answers[i], tracker=tracker)

        candidate_sub_questions_rewards.append({
            "sub_question_reward": reward_subquestion,
            "sub_answer_reward": reward_answer
        })
    return candidate_sub_questions_rewards

def is_terminal_question(question, curr_sub_question):
    if 'Now we can answer' in curr_sub_question:
        return True
    # if curr_sub_question.lower() in question.lower():
    #     return True
    return False
        
