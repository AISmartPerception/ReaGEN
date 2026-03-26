import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io
import re
import copy

from .prompts4 import init_system_prompt, plan_prompt, solve_plan_prompt, plan_answer_template, validate_answers_prompt, judge_plans_prompt, solve_plan_with_feedback_prompt, summarize_final_answer_prompt, summarize_final_answer_without_plans_prompt

attempt_num = 3
validate_answer_num = 3

def generate_message(query, image_url, text_only=False):
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
    return messages

def default_call(messages, model, default_response, task_name="get response") -> str:
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to {task_name} after {attempt_num} attempts. Error: {str(e)}")
                return default_response

def plan_call(question, image_url, model) -> str:
    text_only = model.llm_cfg.text_only
    messages = generate_message(plan_prompt.format(question=question), image_url, text_only)
    plans_str = default_call(messages, model, "", "planning")
    return plans_str

def parse_plans(plans_str) -> list[str]:
    plans_str = re.sub(r'\d. \*\*Plan \d:\*\*', '', plans_str)
    plans = plans_str.split("\n\n")
    plans = [result.strip() for result in plans]
    # 过滤掉空的和重复的字符串
    plans = list(filter(None, plans))
    plans = list(dict.fromkeys(plans))
    return plans

def solve_plan_call(plan, image_url, model, feedback="", answer="") -> str:
    text_only = model.llm_cfg.text_only
    if feedback == "":
        messages = generate_message(solve_plan_prompt.format(plan=plan), image_url, text_only)
    else:
        messages = generate_message(solve_plan_with_feedback_prompt.format(plan=plan, feedback=feedback, answer=answer), image_url, text_only)
    answer = default_call(messages, model, "", "solving the plan")
    return answer

def merge_plans_and_answers(plans, answers) -> str:
    plans_and_answers = ""
    for i, (plan, answer) in enumerate(zip(plans, answers)):
        plans_and_answers += plan_answer_template.format(step=i+1, plan=plan, answer=answer)
    return plans_and_answers

def validate_answers_call(plans, answers, image_url, model) -> str:
    text_only = model.llm_cfg.text_only
    plans_and_answers = merge_plans_and_answers(plans, answers)
    messages = generate_message(validate_answers_prompt.format(plans_and_answers=plans_and_answers), image_url, text_only)
    validation_results = default_call(messages, model, "", "validating the answers")
    return validation_results

def parse_validation_results(plans, answers, validation_results_str) -> list[str]:
    validation_results = re.sub(r'\d. \*\*Plan \d:\*\*', '', validation_results_str)
    validation_results = validation_results.split("\n\n")
    validation_results = [result.strip() for result in validation_results]
    plan_num = len(plans)
    if len(validation_results) >= plan_num:
        validation_results = validation_results[len(validation_results)-plan_num:]
    else:
        validation_results = validation_results + [""]*(plan_num-len(validation_results))
    return validation_results

def judge_plans_call(plans, answers, validation_results_str, image_url, model) -> str:
    text_only = model.llm_cfg.text_only
    plans_and_answers = merge_plans_and_answers(plans, answers)
    messages = generate_message(judge_plans_prompt.format(plans_and_answers=plans_and_answers, validation_results=validation_results_str), image_url, text_only)
    judge_str = default_call(messages, model, "", "judging the plans")
    return judge_str

def parse_judge_results(plans, answers, validation_results, judge_results_str) -> dict:
    judge_results = re.sub(r'\d. \*\*Plan \d:\*\*', '', judge_results_str)
    judge_results = judge_results_str.split("\n\n")
    judge_results = [result.strip() for result in judge_results]
    plan_num = len(plans)
    if len(judge_results) >= plan_num:
        judge_results = judge_results[len(judge_results)-plan_num:]
    else:
        judge_results = judge_results + [""]*(plan_num-len(judge_results))
    actions = []
    for judge_result in judge_results:
        if "Regenerate Answer" in judge_result:
            actions.append("Regenerate Answer")
        elif "Ignore" in judge_result:
            actions.append("Ignore")
        else:
            actions.append("Save to Memory")
    return actions

def summarize_final_answer_call(question, plans, answers, judge_results, validation_results, image_url, model) -> str:
    text_only = model.llm_cfg.text_only
    correct_plans = [plan for plan, action in zip(plans, judge_results) if action=="Save to Memory"]
    correct_answers = [answer for answer, action in zip(answers, judge_results) if action=="Save to Memory"]
    if len(correct_plans) == 0:
        messages = generate_message(summarize_final_answer_without_plans_prompt.format(question=question), image_url, text_only)
    else:
        plans_and_answers = merge_plans_and_answers(correct_plans, correct_answers)
        messages = generate_message(summarize_final_answer_prompt.format(plans_and_answers=plans_and_answers, question=question), image_url, text_only)
    final_answer = default_call(messages, model, "", "summarizing the final answer")
    return final_answer
    

def our4_forward(query, img_path, model, cfg):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_base64}"

    memory = {}  # agent memory
    final_answer = ""
    memory["plans"] = [] # list of plans
    memory["answers"] = [] # list of answers
    memory["validation_results"] = [] # list of validation results
    memory["judge_results"] = [] # list of judge results
    memory["history"] = {
        "answers_list": [],
        "validation_results_list": [],
        "judge_results_list": []
    }

    # planning
    plans_str = plan_call(query, image_url, model.planner)
    plans = parse_plans(plans_str)
    memory["plans"] = plans
    memory["judge_results"] = ["Regenerate Answer"]*len(plans)
    memory["answers"] = [""]*len(plans)
    memory["validation_results"] = [""]*len(plans)

    # solving the plans
    for i in range(validate_answer_num):
        for j, plan in enumerate(plans):
            if memory["judge_results"][j] == "Regenerate Answer":
                answer = solve_plan_call(plan, image_url, model.reasoner, feedback=memory["validation_results"][j])
                memory["answers"][j] = answer
        
        # validating the answers
        validation_results_str = validate_answers_call(memory["plans"], memory["answers"], image_url, model.verifier)
        validation_results = parse_validation_results(memory["plans"], memory["answers"], validation_results_str)
        for j, validation_result in enumerate(validation_results):
            if memory["judge_results"][j] == "Regenerate Answer":
                memory["validation_results"][j] = validation_result
        
        # judging the plans
        judge_results_str = judge_plans_call(memory["plans"], memory["answers"], validation_results_str, image_url, model.judger)
        judge_results = parse_judge_results(memory["plans"], memory["answers"], validation_results, judge_results_str)
        
        for j, action in enumerate(judge_results):
            if memory["judge_results"][j] == "Regenerate Answer":
                memory["judge_results"][j] = action
        
        # save to history
        memory["history"]["answers_list"].append(copy.deepcopy(memory["answers"]))
        memory["history"]["validation_results_list"].append(copy.deepcopy(memory["validation_results"]))
        memory["history"]["judge_results_list"].append(copy.deepcopy(memory["judge_results"]))

        if sum([1 for action in memory["judge_results"] if action=="Regenerate Answer"]) == 0:
            break
    
    # summarizing the final answer
    final_answer = summarize_final_answer_call(query, memory["plans"], memory["answers"], memory["judge_results"], memory["validation_results"], image_url, model.summarizer)


    res = {
        "memory": memory,
        "response": final_answer
    }
    return res