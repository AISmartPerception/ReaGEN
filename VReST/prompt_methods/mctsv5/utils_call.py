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


def default_call(cfg, messages, model, default_response, task_name="get response", n=1, continue_final_message=False, stop_token=None):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None, n=n, continue_final_message=continue_final_message, stop_token=stop_token)
            if "response" in response:
                response = response["response"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.attempt_num-1:
                print(f"Failed to {task_name} after {cfg.prompt_method.attempt_num} attempts. Error: {str(e)}")
                return default_response
            
def reward_call(cfg, messages, model):
    for attempt in range(cfg.prompt_method.attempt_num):
        try:
            response = model.get_reward(messages=messages)
            if "yes_prob" in response:
                response = response["yes_prob"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.attempt_num-1:
                print(f"Failed to get reward after {cfg.prompt_method.attempt_num} attempts. Error: {str(e)}")
                return 0.0


def decompose_question_call(cfg, image_url, model, question, sub_questions, sub_answers, last_sub_question=False):
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
    return default_call(cfg, messages, model, "", "decompose question", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=stop_token) # list

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

def get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer):
    user_prompt = reward_prompt_for_answer + question_prefix.format(question=question)
    for i, sub_question in enumerate(sub_questions):
        user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
        user_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}"
    user_prompt += f"{subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}\n"
    user_prompt += f"{answer_prefix.format(step=len(sub_questions)+1, answer=curr_sub_answer)}"
    user_prompt += reward_prefix_for_answer
    return reward_call(cfg, generate_message(user_prompt, image_url), model)

def get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question):
    user_prompt = reward_prompt + question_prefix.format(question=question) + "\n"
    for i, sub_question in enumerate(sub_questions):
        user_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
    user_prompt += f"{new_subquestion_prefix.format(step=len(sub_questions)+1, subquestion=curr_sub_question)}"
    user_prompt += f"{reward_prefix}"
    return reward_call(cfg, generate_message(user_prompt, image_url), model)

def get_reward(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer):
    # reward_subquestion = get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question)
    reward_subquestion = 1.0
    reward_answer = get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer)
    return reward_subquestion, reward_answer

def get_reward_parallel(cfg, image_url, model, question, sub_questions, sub_answers, candidate_sub_questions, candidate_sub_answers):
    
    # 串行版本
    candidate_sub_questions_rewards = []
    for i, sub_question in enumerate(candidate_sub_questions):
        if is_terminal_question(question, sub_question):
            reward_subquestion, reward_answer = get_reward(cfg, image_url, model, question, sub_questions, sub_answers, sub_question, candidate_sub_answers[i])
        else:
            reward_subquestion, reward_answer = 1.0, 0.5
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
        
