import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
import base64
from PIL import Image
import io

from .prompts import system_prompt

class CoT(BaseModel):
    chain_of_thought: str
    final_answer: str

def cot_forward(query, img_path, model):
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

    for attempt in range(3):
        try:
            response = model.get_response(messages=messages, json_format=CoT)
            response = response["response"]
            response = json.loads(response)
            cot = response["chain_of_thought"]
            answer = response["final_answer"]
            response_content = f"Chain of thought: {cot}\nFinal answer: {answer}"
            res = {
                "response": response_content,
            }
            return res
        except Exception as e:
            if attempt == 2:
                print(f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}")
                answer = f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}"
                res = {
                    "response": answer
                }
                return res
        
        # try:
        #     response = model.get_response(messages=messages, json_format=None)
        #     response = response["response"]
        #     res = {
        #         "response": response,
        #     }
        #     return res
        # except Exception as e:
        #     if attempt == 2:
        #         print(f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}")
        #         answer = f"Failed to generate CoT and final answer after 3 attempts. Error: {str(e)}"
        #         res = {
        #             "response": answer
        #         }
        #         return res
