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

from .prompts5 import init_system_prompt, plan_prompt, solve_first_plan_prompt, solve_first_plan_with_feedback_prompt, solve_last_plan_prompt, solve_last_plan_with_feedback_prompt, plan_answer_template, validate_last_answers_prompt, judge_last_plans_prompt, summarize_final_answer_prompt, summarize_final_answer_without_plans_prompt

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
    plans_str = re.sub(r'\*\*Plan \d:\*\*', '', plans_str) # remove the plan number
    plans_str = re.sub(r'^\d+\.\s*', '', plans_str, flags=re.MULTILINE) # remove the step number
    plans = plans_str.split("\n")
    plans = [result.strip('-').strip() for result in plans]
    # 过滤掉空的和重复的字符串
    plans = list(filter(None, plans))
    plans = list(dict.fromkeys(plans))
    return plans

def solve_plan_call(plan, image_url, model, feedback="", answer="", step=1, plans_and_answers="") -> str:
    text_only = model.llm_cfg.text_only
    if feedback != "" and answer != "":
        if step == 1:
            messages = generate_message(solve_first_plan_with_feedback_prompt.format(plan=plan, feedback=feedback, answer=answer), image_url, text_only)
        else:
            messages = generate_message(solve_last_plan_with_feedback_prompt.format(plans_and_answers=plans_and_answers, step=step, plan=plan, feedback=feedback, answer=answer), image_url, text_only)
    else:
        if step == 1:
            messages = generate_message(solve_first_plan_prompt.format(plan=plan), image_url, text_only)
        else:
            messages = generate_message(solve_last_plan_prompt.format(plans_and_answers=plans_and_answers, step=step, plan=plan), image_url, text_only)

    answer = default_call(messages, model, "", "solving the plan")
    return answer

def parse_answer(answer_str) -> str:
    answer = re.sub(r'\*\*Answer \d:\*\*', '', answer_str)
    answer = answer.strip().strip('-').strip()
    return answer

def merge_plans_and_answers(plans, answers) -> str:
    plans_and_answers = ""
    for i, (plan, answer) in enumerate(zip(plans, answers)):
        plans_and_answers += plan_answer_template.format(step=i+1, plan=plan, answer=answer)
    return plans_and_answers

def validate_answers_call(plans_and_answers, image_url, model, step=1) -> str:
    text_only = model.llm_cfg.text_only
    message = generate_message(validate_last_answers_prompt.format(plans_and_answers=plans_and_answers, step=step), image_url, text_only)
    validation_results_str = default_call(message, model, "", "validating the answers")
    return validation_results_str

def judge_plans_call(plans_and_answers, validation_results_str, image_url, model, step=1) -> str:
    text_only = model.llm_cfg.text_only
    message = generate_message(judge_last_plans_prompt.format(plans_and_answers=plans_and_answers, validation_results=validation_results_str, step=step), image_url, text_only)
    judge_results_str = default_call(message, model, "", "judging the plans")
    return judge_results_str

def summarize_final_answer_call(question, plans, answers, image_url, model) -> str:
    text_only = model.llm_cfg.text_only
    if len(plans) == 0:
        messages = generate_message(summarize_final_answer_without_plans_prompt.format(question=question), image_url, text_only)
    else:
        assert len(plans) == len(answers)
        plans_and_answers = merge_plans_and_answers(plans, answers)
        messages = generate_message(summarize_final_answer_prompt.format(plans_and_answers=plans_and_answers, question=question), image_url, text_only)
    final_answer = default_call(messages, model, "", "summarizing the final answer")
    return final_answer
    

def our5_forward(query, img_path, model, cfg):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_base64}"

    memory = {}  # agent memory
    final_answer = ""
    memory["plans"] = [] # list of final saved plans
    memory["answers"] = [] # list of final saved answers
    
    # planning
    plans_str = plan_call(query, image_url, model.planner)
    plans = parse_plans(plans_str)
    memory["all_plans"] = plans

    memory["history"] = {}

    # solving the plans sequentially
    step = 1
    for i, plan in enumerate(plans):
        feedback, answer = "", ""
        memory["history"][f"plan {i+1}"] = {
            "plan": plan,
            "answers": [],
            "validation_results": [],
            "judge_results": []
        }
        for j in range(validate_answer_num):
            if len(memory["answers"]) > 0:
                assert len(memory["plans"]) == len(memory["answers"])
                plans_and_answers = merge_plans_and_answers(memory["plans"], memory["answers"])
            else:
                plans_and_answers = ""
            answer = solve_plan_call(plan, image_url, model.reasoner, feedback=feedback, answer=answer, step=step, plans_and_answers=plans_and_answers)
            answer = parse_answer(answer)

            memory["history"][f"plan {i+1}"]["answers"].append(answer)

            memory["plans"].append(plan)
            memory["answers"].append(answer)
            plans_and_answers = merge_plans_and_answers(memory["plans"], memory["answers"])
            
            # validating the answers
            validation_results_str = validate_answers_call(plans_and_answers, image_url, model.verifier, step=step)

            memory["history"][f"plan {i+1}"]["validation_results"].append(validation_results_str)
            
            # judging the plans
            judge_results_str = judge_plans_call(plans_and_answers, validation_results_str, image_url, model.judger, step=step)

            memory["history"][f"plan {i+1}"]["judge_results"].append(judge_results_str)

            if "Regenerate Answer" in judge_results_str:
                feedback = validation_results_str
                memory["plans"].pop()
                memory["answers"].pop()
                continue
            elif "Ignore" in judge_results_str:
                break
            else:
                step += 1
                break
    
    # summarizing the final answer
    final_answer = summarize_final_answer_call(query, memory["plans"], memory["answers"], image_url, model.summarizer)

    res = {
        "memory": memory,
        "response": final_answer
    }
    return res