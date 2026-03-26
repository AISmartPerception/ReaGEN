import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io

from .prompts import system_prompt, first_assistant_prompt, generate_final_answer_prompt, system_prompt_final

class Step(BaseModel):
    title: str
    content: str
    next_action: str

class Answer(BaseModel):
    final_answer: str

def make_api_call(messages, model=None, is_final_answer=False):

    for attempt in range(3):
        try:
            if is_final_answer:
                response = model.get_response(messages=messages, json_format=Answer)
                response = response["response"]
                return json.loads(response)
            else:
                response = model.get_response(messages=messages, json_format=Step)
                response = response["response"]
                return json.loads(response)
            
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    print(f"Failed to generate final answer after 3 attempts. Error: {str(e)}")
                    return {"final_answer": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    print(f"Failed to generate step after 3 attempts. Error: {str(e)}")
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}

def generate_response(query, img_path, model):
    # 打开本地图片
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
    # 将图片数据转换为 base64 编码
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    # 构建 data URL
    image_url = f"data:image/jpeg;base64,{image_base64}"

    messages=[
        {"role": "system", "content": system_prompt},
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

    final_messages = [
        {"role": "system", "content": system_prompt_final},
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
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, model)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time, step_data["next_action"]))

        step_content = f"Step {step_count}: {step_data['title']}" + "\n" + step_data['content']

        messages.append({"role": "assistant", "content": [{"type": "text", "text" : json.dumps(step_data)}]})
        
        final_messages.append({"role": "assistant", "content": [{"type": "text", "text" : step_content}]})
        
        if step_data['next_action'] == 'final_answer' or step_count > 10: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
            break
        
        step_count += 1

        # Yield after each step for Streamlit to update
        # yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    final_messages.append({"role": "user", "content": [{"type": "text", "text": generate_final_answer_prompt}]})
    
    start_time = time.time()
    final_data = make_api_call(final_messages, model=model, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    final_answer = final_data['final_answer']
    
    steps.append(("Final Answer", final_answer, thinking_time))

    res = {
        "steps": steps,
        "response": final_answer
    }

    return res

def our_forward(query, img_path, model):
    return generate_response(query, img_path, model)