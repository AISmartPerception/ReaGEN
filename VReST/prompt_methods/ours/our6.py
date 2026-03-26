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
from .prompts6 import init_system_prompt, plan_prompt, question_prefix, subquestion_prefix, overall_question_prefix, answer_prefix, reward_prompt, new_subquestion_prefix, reward_prefix, reward_prompt_for_answer, reward_prefix_for_answer


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
                return response["yes_prob"]
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == cfg.prompt_method.attempt_num-1:
                print(f"Failed to get reward after {cfg.prompt_method.attempt_num} attempts. Error: {str(e)}")
                return 0.0


def decompose_question_call(cfg, image_url, model, question, sub_questions, sub_answers):
    user_prompt = plan_prompt + question_prefix.format(question=question)
    assistant_prompt = ""
    for i, sub_question in enumerate(sub_questions):
        assistant_prompt += f"{subquestion_prefix.format(step=i+1, subquestion=sub_question)}\n"
        assistant_prompt += f"{answer_prefix.format(step=i+1, answer=sub_answers[i])}\n\n"
    assistant_prompt += subquestion_prefix.format(step=len(sub_questions)+1, subquestion='').strip()
    stop_token = subquestion_prefix.format(step=len(sub_questions)+2, subquestion='').strip()
    messages = generate_message(user_prompt, image_url, assistant_prompt=assistant_prompt)

    responses = default_call(cfg, messages, model, "", "decompose question", n=cfg.prompt_method.tree_width, continue_final_message=True, stop_token=stop_token) # list
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
    if cfg.prompt_method.reward_method == "subquestion_answer":
        reward_subquestion = get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question)
        reward_answer = get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer)
        return reward_subquestion * reward_answer
    elif cfg.prompt_method.reward_method == "subquestion":
        return get_reward_subquestion(cfg, image_url, model, question, sub_questions, curr_sub_question)
    else:
        return get_reward_answer(cfg, image_url, model, question, sub_questions, sub_answers, curr_sub_question, curr_sub_answer)

def get_reward_parallel(cfg, image_url, model, question, sub_questions, sub_answers, candidate_sub_questions, candidate_sub_answers):
    
    # 串行版本
    candidate_sub_questions_rewards = []
    for i, sub_question in enumerate(candidate_sub_questions):
        reward = get_reward(cfg, image_url, model, question, sub_questions, sub_answers, sub_question, candidate_sub_answers[i])
        candidate_sub_questions_rewards.append(reward)
    
    # 并行版本
    # candidate_sub_questions_rewards = [0] * len(candidate_sub_questions)
    
    # def get_reward_with_index(index, sub_question):
    #     reward = get_reward(cfg, image_url, model, question, sub_questions, sub_answers, sub_question, candidate_sub_answers[index])
    #     return index, reward
    
    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(get_reward_with_index, i, sub_question) 
    #                 for i, sub_question in enumerate(candidate_sub_questions)]
        
    #     for future in as_completed(futures):
    #         index, reward = future.result()
    #         candidate_sub_questions_rewards[index] = reward
    
    return candidate_sub_questions_rewards

def is_terminal_question(question, curr_sub_question):
    if 'Now we can answer' in curr_sub_question:
        return True
    if curr_sub_question.lower() in question.lower():
        return True
    return False
        
def our6_forward(query, img_path, model, cfg):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_base64}"
    
    if cfg.prompt_method.simple_cot:
        path = simple_cot_call(cfg, image_url, model, query, n=cfg.prompt_method.n_vote)
        if isinstance(path, list):
            answers = [extract_final_answer(p) for p in path]
            answers_word = [extract_final_answer_word(a) for a in answers]
            final_answer, index = majority_vote(answers_word)
            final_answer = answers[index]
        else:
            final_answer = extract_final_answer(path)
        res = {
            "memory": path,
            "response": final_answer
        }
        return res
    
    memory = {}  # agent memory

    """
    1、根据当前状态，继续生成子问题，采样多条。
    2、对每个子问题进行评估，得到一个分数。
    3、根据分数，选择一个子问题进行回答。
    4、判断是不是最后一个问题，如果是，则输出最终答案，否则，返回第1步。
    """
    
    sub_questions = []
    sub_answers = []
    
    final_answer = ""

    # Decompose the question
    for i in range(cfg.prompt_method.decompose_question_num):
        
        candidate_paths = decompose_question_call(cfg, image_url, model, query, sub_questions, sub_answers)
        candidate_sub_questions = []
        candidate_sub_answers = []
        for j, path in enumerate(candidate_paths):
            sub_question, sub_answer = extract_sub_questions_and_answers(path, i+1)
            if sub_question == "" or sub_answer == "":
                continue
            candidate_sub_questions.append(sub_question)
            candidate_sub_answers.append(sub_answer)
        
        if len(candidate_sub_questions) == 0 or len(candidate_sub_answers) == 0:
            print(f"Error: Failed to decompose question!")
            final_answer = candidate_paths[0]
            break
        
        # 对每个子问题进行评估，得到一个分数。
        candidate_sub_questions_rewards = get_reward_parallel(cfg, image_url, model, query, sub_questions, sub_answers, candidate_sub_questions, candidate_sub_answers)
            
        # 选择分数最高的子问题
        best_sub_question_index = candidate_sub_questions_rewards.index(max(candidate_sub_questions_rewards))
        best_sub_question = candidate_sub_questions[best_sub_question_index]
        best_sub_answer = candidate_sub_answers[best_sub_question_index]
        sub_questions.append(best_sub_question)
        sub_answers.append(best_sub_answer)
        
        # 判断是不是最后一个问题，如果是，则输出最终答案，否则，返回第1步。
        if is_terminal_question(query, best_sub_question):
            final_answer = best_sub_answer
            break
    memory["sub_questions"] = sub_questions
    memory["sub_answers"] = sub_answers
    
    if final_answer == "":
        if len(sub_answers) > 0:
            final_answer = sub_answers[-1]
        else:
            print(f"Error: Failed to get final answer!")
            final_answer = candidate_paths[0]

    res = {
        "memory": memory,
        "response": final_answer
    }

    return res