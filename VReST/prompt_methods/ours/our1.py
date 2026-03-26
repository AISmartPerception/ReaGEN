import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io

from .prompts import system_prompt, generate_final_answer_prompt, system_prompt_final

def make_api_call(messages, model=None, is_final_answer=False):

    for attempt in range(3):
        try:
            if is_final_answer:
                response = model.get_response(messages=messages, json_format=None)
                response = response["response"]
                return response
    
            else:
                response = model.get_response(messages=messages, json_format=None)
                response = response["response"]
                assert '"title":' in response
                assert '"content":' in response
                assert '"next_action":' in response
                title = response.split('"title":')[1].split('"content":')[0].strip().strip(',"')
                content = response.split('"content":')[1].split('"next_action":')[0].strip().strip(',"')
                next_action = response.split('"next_action":')[1].strip().strip('}').strip().strip('"')
                # response = json.loads(response)
                response = {"title": title, "content": content, "next_action": next_action}
                
                return response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"The model response was:\n {response}")
            if attempt == 2:
                if is_final_answer:
                    print(f"Failed to generate final answer after 3 attempts. Error: {str(e)}")
                    return f"Failed to generate final answer after 3 attempts. Error: {str(e)}"
                else:
                    print(f"Failed to generate step after 3 attempts. Error: {str(e)}")
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}

def our1_forward(query, img_path, model):
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

        step_content = f"Step {step_count}: {step_data['title']}" + "\n" + step_data['content'] + "\n" + "Next Action: " + step_data['next_action']

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

    final_answer = final_data
    
    steps.append(("Final Answer", final_answer, thinking_time))

    res = {
        "steps": steps,
        "response": final_answer
    }

    return res