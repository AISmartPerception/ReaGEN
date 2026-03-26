import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io

from .prompts3 import init_system_prompt, refine_question, decompose_question, answer_sub_question, validate_answer, check_final_answer, generate_final_answer, decompose_question_first, decompose_question_with_feedback, reanswer_sub_question, reanswer_sub_question_with_feedback

attempt_num = 3
decompose_question_num = 5
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
        return messages
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


def default_call(messages, model, default_response, task_name="get response", text_only=False):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None, text_only=text_only)
            response = response["response"]
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to {task_name} after {attempt_num} attempts. Error: {str(e)}")
                return default_response

def feedback_and_next_action_call(messages, model, default_action):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            assert 'next_action:' in response
            next_action = response.split('next_action:')[1].strip()
            if "feedback:" in response:
                feedback = response.split('feedback:')[1].split('next_action:')[0].strip().strip('-').strip()
            else:
                feedback = ""
            response = {
                "feedback": feedback,
                "next_action": next_action
            }
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to extract feedback and next action after {attempt_num} attempts. Error: {str(e)}")
                response = {
                    "feedback": "",
                    "next_action": default_action
                }
                return response

def refine_question_call(messages, model, ori_question):
    return default_call(messages, model, "", "refine question")

def decompose_question_call(completed_question, sub_questions, feedback, model, image_url, text_only=False):
    if sub_questions != "":
        if feedback != "":
            messages=generate_message(decompose_question_with_feedback.format(question=completed_question, sub_questions=sub_questions, feedback=feedback), image_url, text_only)
        else:
            messages=generate_message(decompose_question.format(question=completed_question, sub_questions=sub_questions), image_url, text_only)
    else:
        messages=generate_message(decompose_question_first.format(question=completed_question), image_url, text_only)
    return default_call(messages, model, "", "decompose question", text_only)

def answer_sub_question_call(sub_question, sub_question_answer, feedback, model, image_url):
    if sub_question_answer != "":
        if feedback != "":
            messages=generate_message(reanswer_sub_question_with_feedback.format(question=sub_question, answer=sub_question_answer, feedback=feedback), image_url)
        else:
            messages=generate_message(reanswer_sub_question.format(question=sub_question, answer=sub_question_answer), image_url)
    else:
        messages=generate_message(answer_sub_question.format(question=sub_question), image_url)
    return default_call(messages, model, "", "answer sub-question")

def validate_answer_call(messages, model):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            if "is correct" in response:
                response = {
                    "feedback": "",
                    "next_action": "check_final_answer"
                }
                return response
            if "is incorrect" in response:
                response = {
                    "feedback": response.split("next_action:")[0].strip().strip('-').strip(),
                    "next_action": "answer_sub_question"
                }
                return response
            assert 'next_action:' in response
            next_action = response.split('next_action:')[1].strip()
            if "feedback:" in response:
                feedback = response.split('feedback:')[1].split('next_action:')[0].strip().strip('-').strip()
            else:
                feedback = ""
            response = {
                "feedback": feedback,
                "next_action": next_action
            }
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to extract feedback and next action after {attempt_num} attempts. Error: {str(e)}")
                response = {
                    "feedback": "",
                    "next_action": "check_final_answer"
                }
                return response

def check_final_answer_call(messages, model):
    for attempt in range(attempt_num):
        try:
            response = model.get_response(messages=messages, json_format=None)
            response = response["response"]
            if "is sufficient" in response:
                response = {
                    "feedback": "",
                    "next_action": "generate_final_answer"
                }
                return response
            if "is insufficient" in response:
                response = {
                    "feedback": response.split("next_action:")[0].strip().strip('-').strip(),
                    "next_action": "decompose_question"
                }
                return response
            assert 'next_action:' in response
            next_action = response.split('next_action:')[1].strip()
            if "feedback:" in response:
                feedback = response.split('feedback:')[1].split('next_action:')[0].strip().strip('-').strip()
            else:
                feedback = ""
            response = {
                "feedback": feedback,
                "next_action": next_action
            }
            return response
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == attempt_num-1:
                print(f"Failed to extract feedback and next action after {attempt_num} attempts. Error: {str(e)}")
                response = {
                    "feedback": "",
                    "next_action": "generate_final_answer"
                }
                return response

def generate_final_answer_call(messages, model):
    return default_call(messages, model, "", "generate final answer")

def our3_forward(query, img_path, model, cfg):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{image_base64}"

    memory = {}  # agent memory
    memory["refined_question"] = ""
    memory["sub_questions"] = []
    memory["sub_questions_content"] = ""
    memory["reasoning"] = ""
    memory["actions"] = []

    """
    1、首先对问题进行精细化，澄清其中不清楚的实体、定义和概念。
    2、从这个复杂的问题中动态分解出一个子问题。
    3、回答这个子问题。
    4、验证子问题是否回答正确，如果不正确，重新执行第3步。
    5、判断以上推理是否能够得出最终答案，如果不可以则返回第2步，继续动态分解子问题。
    6、综合上述推理，总结成一个思维链，得出最终答案。
    """

    reasoning = ""

    # Refine the question
    if cfg.prompt_method.refine_question:
        memory["actions"].append("refine_question")
        messages=generate_message(refine_question.format(question=query), image_url)
        refined_question = refine_question_call(messages, model, query)
        reasoning += f"First, we refine the question.\n {refined_question}\n"
    else:
        refined_question = ""
    completed_question = query + "\n" + refined_question if refined_question != "" else query
    memory["refined_question"] = refined_question    

    check_final_feedback = ""

    # Decompose the question
    for i in range(decompose_question_num):

        memory["actions"].append("decompose_question")
        if i == 0:
            reasoning += f"Then, we decompose the original question and get a sub-plan.\n"
        else:
            reasoning += f"Then, we continue decompose the original question and get another sub-plan.\n"
            
        sub_question = decompose_question_call(completed_question, memory["sub_questions_content"], check_final_feedback, model, image_url, cfg.prompt_method.decompose_text_only)
        if sub_question is None or sub_question == "":
            reasoning + f"We have no more sub-plans to decompose.\n"
            break
        sub_qa = {
            "sub_question": sub_question,
            "sub_question_answer": [],
        }
        memory["sub_questions_content"] += f"  - Step {i+1}: {sub_question}\n"
        reasoning += f"Step {i+1}: {sub_question}\n"

        sub_question_answer, feedback = "", ""
        for j in range(validate_answer_num):
            # Answer the sub-question
            memory["actions"].append("answer_sub_question")
            if j == 0:
                reasoning += f"Next, we answer the sub-plan.\n"
            else:
                reasoning += f"Next, we re-answer the sub-plan.\n"
            
            sub_question_answer = answer_sub_question_call(sub_question, sub_question_answer, feedback, model, image_url)
            
            if sub_question_answer == "":
                reasoning += f"We can not answer the sub-question.\n"
                break
            sub_qa["sub_question_answer"].append(sub_question_answer)

            reasoning += f"Step {i+1} answer: {sub_question_answer}\n"

            # Validate the answer
            if cfg.prompt_method.validate:
                memory["actions"].append("validate_answer")
                reasoning += f"Next, we validate the answer to the sub-question.\n"
                messages=generate_message(validate_answer.format(question=sub_question, answer=sub_question_answer), image_url)
                validate = validate_answer_call(messages, model)
                next_action, feedback = validate["next_action"], validate["feedback"]

                if next_action == "check_final_answer":
                    reasoning += f"Sub-question {i+1} answer is correct.\n"                
                    break

                if feedback == "":
                    reasoning += f"Sub-question {i+1} answer is incorrect.\n"
                else:
                    reasoning += f"Sub-question {i+1} answer is incorrect because: {feedback}\n"
            else:
                break
        
        memory["sub_questions"].append(sub_qa)
        
        # Check the final answer
        memory["actions"].append("check_final_answer")
        reasoning += f"Next, we checked whether the current reasoning and answers to sub-questions are sufficient to derive the final answer.\n"
        messages=generate_message(check_final_answer.format(question=query, reasoning=reasoning), image_url)
        check = check_final_answer_call(messages, model)
        next_action, check_final_feedback = check["next_action"], check["feedback"]

        if next_action == "generate_final_answer":
            reasoning += f"The reasoning is sufficient to derive the final answer.\n"
            break

        if check_final_feedback == "":
            reasoning += f"The reasoning is insufficient to derive the final answer.\n"
        else:
            reasoning += f"The reasoning is insufficient to derive the final answer because: {check_final_feedback}\n"
    
    # Generate the final answer
    memory["actions"].append("generate_final_answer")
    reasoning += f"Finally, we summarize the above reasoning process and derive the final answer.\n"
    messages=generate_message(generate_final_answer.format(question=query, reasoning=reasoning), image_url)
    final_answer = generate_final_answer_call(messages, model)

    memory["reasoning"] = reasoning

    res = {
        "memory": memory,
        "response": final_answer
    }

    return res