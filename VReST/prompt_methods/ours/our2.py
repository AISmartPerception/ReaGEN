import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io

from .prompts import init_system_prompt, refine_question, decompose_question, answer_sub_question, validate_answer, check_final_answer, generate_final_answer

attempt_num = 3
decompose_question_num = 5
validate_answer_num = 3

def generate_message(query, image_url):
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

def refine_question_call(messages, model, ori_question):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to refine question after {attempt_num} attempts. Error: {str(e)}")
                return ori_question

def decompose_question_call(messages, model):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to decompose question after {attempt_num} attempts. Error: {str(e)}")
                return None

def answer_sub_question_call(messages, model):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to answer sub-question after {attempt_num} attempts. Error: {str(e)}")
                return ""

def validate_answer_call(messages, model):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            assert 'next_action:' in response
            next_action = response.split('next_action:')[1].strip()
            return next_action
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to validate answer after {attempt_num} attempts. Error: {str(e)}")
                return "check_final_answer"

def check_final_answer_call(messages, model):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            assert 'next_action:' in response
            next_action = response.split('next_action:')[1].strip()
            return next_action
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to check final answer after {attempt_num} attempts. Error: {str(e)}")
                return "generate_final_answer"

def generate_final_answer_call(messages, model):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to generate final answer after {attempt_num} attempts. Error: {str(e)}")
                return ""

def our2_forward(query, img_path, model):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_base64}"

    steps = {}  # agent memory
    steps["sub_questions"] = []
    steps["reasoning"] = "\n"
    steps["actions"] = []

    """
    1、首先对问题进行精细化，澄清其中不清楚的实体、定义和概念。
    2、从这个复杂的问题中动态分解出一个子问题。
    3、回答这个子问题。
    4、验证子问题是否回答正确，如果不正确，重新执行第3步。
    5、判断以上推理是否能够得出最终答案，如果不可以则返回第2步，继续动态分解子问题。
    6、综合上述推理，总结成一个思维链，得出最终答案。
    """

    # Refine the question
    steps["actions"].append("refine_question")
    messages=generate_message(refine_question.format(question=query), image_url)
    refined_question = refine_question_call(messages, model, query)
    steps["refined_question"] = refined_question

    # Decompose the question
    for i in range(decompose_question_num):

        steps["actions"].append("decompose_question")
        messages=generate_message(decompose_question.format(question=refined_question, sub_questions=steps["reasoning"]), image_url)
        sub_question = decompose_question_call(messages, model)
        if sub_question is None or sub_question == "":
            break

        for j in range(validate_answer_num):
            # Answer the sub-question
            steps["actions"].append("answer_sub_question")
            messages=generate_message(answer_sub_question.format(question=sub_question), image_url)
            sub_question_answer = answer_sub_question_call(messages, model)
            if sub_question_answer == "":
                break

            # Validate the answer
            steps["actions"].append("validate_answer")
            messages=generate_message(validate_answer.format(question=sub_question, answer=sub_question_answer), image_url)
            next_action = validate_answer_call(messages, model)

            if next_action == "check_final_answer":
                sub_qa = {
                    "sub_question": sub_question,
                    "sub_answer": sub_question_answer
                }
                steps["sub_questions"].append(sub_qa)
                steps["reasoning"] += f"Sub-question {i+1}: {sub_question}\nAnswer: {sub_question_answer}\n"
                break
        
        # Check the final answer
        steps["actions"].append("check_final_answer")
        messages=generate_message(check_final_answer.format(question=query, reasoning=steps["reasoning"]), image_url)
        next_action = check_final_answer_call(messages, model)

        if next_action == "generate_final_answer":
            break
    
    # Generate the final answer
    steps["actions"].append("generate_final_answer")
    messages=generate_message(generate_final_answer.format(question=query, reasoning=steps["reasoning"]), image_url)
    final_answer = generate_final_answer_call(messages, model)

    res = {
        "memory": steps,
        "response": final_answer
    }

    return res