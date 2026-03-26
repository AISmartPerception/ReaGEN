import math
import itertools
import threading
import collections
import omegaconf
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import AsyncOpenAI

from prompt_methods.cantor.cantor import cantor_forward
from prompt_methods.ours.our import our_forward
from prompt_methods.ours.our1 import our1_forward
from prompt_methods.ours.our2 import our2_forward
from prompt_methods.ours.our3 import our3_forward
from prompt_methods.ours.our4 import our4_forward
from prompt_methods.ours.our5 import our5_forward
from prompt_methods.ours.our6 import our6_forward
from prompt_methods.cot.cot import cot_forward
from prompt_methods.qa.qa import qa_forward

from prompt_methods.mcts.chartqa_mcts import mcts_forward
from prompt_methods.mctsv2.chartqa_mcts import mcts_forward_v2
from prompt_methods.mctsv3.chartqa_mcts import mcts_forward_v3
from prompt_methods.mctsv4.chartqa_mcts import mcts_forward_v4
from prompt_methods.mctsv5.chartqa_mcts import mcts_forward_v5
from prompt_methods.mctsv6.chartqa_mcts import mcts_forward_v6
from prompt_methods.mctsv7.chartqa_mcts import mcts_forward_v7
from prompt_methods.mctsv8.chartqa_mcts import mcts_forward_v8

# text only

class Llama_Model():
    def __init__(self, llm_cfg):

        self.llm_cfg = llm_cfg
        
        if llm_cfg.call_method == "api":
            self.client = OpenAI(
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
            )

    def get_response(self, messages=None, json_format=None):

        if self.llm_cfg.call_method == "api":

            if json_format is not None:
            
                chat_response = self.client.beta.chat.completions.parse(
                    model=self.llm_cfg.name,
                    messages=messages,
                    response_format=json_format if json_format else None,
                    temperature=self.llm_cfg.temperature,
                    max_tokens=self.llm_cfg.max_output_len,
                    extra_body={
                        "repetition_penalty": 1.05,
                    },
                )

            else:
                    
                chat_response = self.client.chat.completions.create(
                    model=self.llm_cfg.name,
                    messages=messages,
                    temperature=self.llm_cfg.temperature,
                    max_tokens=self.llm_cfg.max_output_len,
                    extra_body={
                        "repetition_penalty": 1.05,
                    },
                )

            response = {
                "response": chat_response.choices[0].message.content
            }

            return response
        else:
            raise ValueError("Invalid call method")


class Qwen25_Model():
    def __init__(self, llm_cfg):

        self.llm_cfg = llm_cfg
        
        if llm_cfg.call_method == "api":
            if isinstance(llm_cfg.base_url, omegaconf.listconfig.ListConfig):
                clients = []
                for base_url in llm_cfg.base_url:
                    clients.append(OpenAI(
                        api_key=llm_cfg.api_key,
                        base_url=base_url,
                    ))
                self.all_clients = clients
                # self.client = itertools.cycle(self.all_clients)
            else:
                self.client = OpenAI(
                    api_key=llm_cfg.api_key,
                    base_url=llm_cfg.base_url,
                )
                self.all_clients = [self.client]
                # self.client = itertools.cycle(self.all_clients)
            # 定义一个锁
            self.lock = threading.Lock()
            # 定义一个字典记录每个client的请求次数
            self.request_count = collections.Counter()
            
            self.attempt_num = 2*len(self.all_clients)
            
    def get_least_request_client(self):
        for client in self.all_clients:
            if self.request_count[client] == 0:
                return client
        return min(self.request_count, key=self.request_count.get)
    
    def get_yes_or_no(self, messages=None):
        
        for attempt in range(self.attempt_num):
            try:
                with self.lock:
                    client = self.get_least_request_client()
                    self.request_count[client] += 1
                
                chat_response = client.chat.completions.create(
                    n=self.llm_cfg.n,
                    model=self.llm_cfg.name,
                    messages=messages,
                    temperature=self.llm_cfg.temperature,
                    top_p=self.llm_cfg.top_p,
                    max_tokens=self.llm_cfg.max_output_len,
                    extra_body={
                        "guided_choice": ["Yes", "No"],
                    },
                )
            
                response = {
                    "response": [choice.message.content for choice in chat_response.choices] if self.llm_cfg.n > 1 else chat_response.choices[0].message.content
                }
                
                with self.lock:
                    self.request_count[client] -= 1

                return response
            
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt == self.attempt_num-1:
                    print(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
                    raise ValueError(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")

    def get_response(self, messages=None, json_format=None):

        if self.llm_cfg.call_method == "api":
            
            for attempt in range(self.attempt_num):
                try:
                    with self.lock:
                        client = self.get_least_request_client()
                        self.request_count[client] += 1

                    if json_format is not None:
                        
                        json_schema = json_format.model_json_schema()
                        
                        chat_response = client.chat.completions.create(
                            n=self.llm_cfg.n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": 1.05,
                                # "guided_decoding_backend": "outlines",
                                "guided_json": json_schema,
                            },
                        )
                        
                        # chat_response = client.beta.chat.completions.parse(
                        #     n=self.llm_cfg.n,
                        #     model=self.llm_cfg.name,
                        #     messages=messages,
                        #     response_format=json_format if json_format else None,
                        #     temperature=self.llm_cfg.temperature,
                        #     top_p=self.llm_cfg.top_p,
                        #     max_tokens=self.llm_cfg.max_output_len,
                        #     extra_body={
                        #         "repetition_penalty": 1.05,
                        #     },
                        # )

                    else:
                    
                        chat_response = client.chat.completions.create(
                            n=self.llm_cfg.n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": 1.05,
                            },
                        )
            
                    response = {
                        "response": [choice.message.content for choice in chat_response.choices] if self.llm_cfg.n > 1 else chat_response.choices[0].message.content
                    }

                    with self.lock:
                        self.request_count[client] -= 1

                    return response
                
                except Exception as e:
                    print(f"Error: {str(e)}")
                    if attempt == self.attempt_num-1:
                        print(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
                        raise ValueError(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
        else:
            raise ValueError("Invalid call method")

# vision and text

class Phi_Model():
    def __init__(self, cfg):
        # Note: The default setting of max_num_seqs (256) and
        # max_model_len (128k) for this model may cause OOM.
        # You may lower either to run this example on lower-end GPUs.

        # In this example, we override max_num_seqs to 5 while
        # keeping the original context length of 128k.
        llm_cfg = cfg.llm
        self.llm = LLM(
            model=llm_cfg.path,
            trust_remote_code=True,
            max_model_len=llm_cfg.max_input_len,
            max_num_seqs=llm_cfg.max_output_len,
        )
        stop_token_ids = None
        self.sampling_params = SamplingParams(temperature=llm_cfg.temperature,
                                     max_tokens=1024,
                                     stop_token_ids=stop_token_ids)

    def get_response(self, input_text, image_path):
        
        prompt = f"<|user|>\n<|image_1|>\n{input_text}<|end|>\n<|assistant|>\n"  # noqa: E501
        image = Image.open(image_path)
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params, use_tqdm=False)
        response = {
            "response": outputs[0].outputs[0].text
        }
        return response

from io import BytesIO
import base64
import requests
from PIL import Image
import hashlib

class Qwen2_VL_Model():
    def __init__(self, llm_cfg):

        self.llm_cfg = llm_cfg
        self._img_token_cache = {} 

        try:
            self.processor = AutoProcessor.from_pretrained(self.llm_cfg.path)
            self._vision_patch = int(getattr(getattr(self.processor, "image_processor", None), "patch_size", 14))
        except Exception:
            self.processor = None
            self._vision_patch = 14

        if llm_cfg.call_method == "local":
            self.llm = LLM(
                model=llm_cfg.path,
                limit_mm_per_prompt={"image": 10, "video": 10},
                max_model_len=llm_cfg.max_input_len,
                max_num_seqs=llm_cfg.max_output_len,
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=llm_cfg.max_output_len,
                stop_token_ids=[],
            )
        else:
            self.llm = None
            self.sampling_params = None

        if llm_cfg.call_method == "api":
            if isinstance(llm_cfg.base_url, omegaconf.listconfig.ListConfig):
                clients = []
                for base_url in llm_cfg.base_url:
                    clients.append(OpenAI(
                        api_key=llm_cfg.api_key,
                        base_url=base_url,
                    ))
                self.all_clients = clients
                # self.client = itertools.cycle(self.all_clients)
            else:
                self.client = OpenAI(
                    api_key=llm_cfg.api_key,
                    base_url=llm_cfg.base_url,
                )
                self.all_clients = [self.client]
                # self.client = itertools.cycle(self.all_clients)
            # 定义一个锁
            self.lock = threading.Lock()
            # 定义一个字典记录每个client的请求次数
            self.request_count = collections.Counter()
            
            self.attempt_num = 2*len(self.all_clients)
    
    def get_least_request_client(self):
        for client in self.all_clients:
            if self.request_count[client] == 0:
                return client
        return min(self.request_count, key=self.request_count.get)

    def _image_key(self, content_item) -> str | None:
        """
        Make a stable cache key for an image content item.
        - For data URIs: SHA1 of decoded bytes
        - For http(s) or local paths: the path/URL string itself
        - For PIL/bytes inputs: SHA1 of bytes
        """
        src = self._extract_url_or_obj(content_item)
        if src is None:
            return None

        # Already a PIL.Image
        if isinstance(src, Image.Image):
            try:
                buf = BytesIO()
                src.save(buf, format="PNG")
                digest = hashlib.sha1(buf.getvalue()).hexdigest()
                return f"pil:{digest}"
            except Exception:
                return None

        # Raw bytes
        if isinstance(src, (bytes, bytearray)):
            try:
                digest = hashlib.sha1(src).hexdigest()
                return f"bytes:{digest}"
            except Exception:
                return None

        # Strings (data URL, http(s), or path)
        if isinstance(src, str):
            if src.startswith("data:image"):
                try:
                    _, b64 = src.split(",", 1)
                    raw = base64.b64decode(b64)
                    digest = hashlib.sha1(raw).hexdigest()
                    return f"data:{digest}"
                except Exception:
                    return None
            # http(s) or local path: use string directly (cheap, stable)
            return f"url:{src}"

        return None


    def _extract_url_or_obj(self, content_item):
        """Return a string URL/data-URI/local path, a PIL.Image, or bytes, from an image content item."""
        if not isinstance(content_item, dict):
            return None
        t = content_item.get("type")
        if t == "image_url":
            iu = content_item.get("image_url")
            # Handle nested dict {"url": "..."} or direct string
            if isinstance(iu, dict):
                return iu.get("url") or iu.get("data") or iu.get("path")
            return iu
        if t == "image":
            # Some toolkits put image content under "image", similar nesting
            img = content_item.get("image")
            if isinstance(img, dict):
                return img.get("url") or img.get("data") or img.get("path")
            return img
        return None

    def _load_pil_from_content(self, content_item):
        """
        Supports:
        - {"type":"image_url","image_url":{"url":"data:image/...;base64,..."}} or "...http(s)..." or local path
        - {"type":"image","image": <same forms or PIL.Image or bytes>}
        Returns PIL.Image.Image or None.
        """
        src = self._extract_url_or_obj(content_item)
        if src is None:
            return None

        # 1) Already a PIL image
        if isinstance(src, Image.Image):
            return src.convert("RGB")

        # 2) Raw bytes
        if isinstance(src, (bytes, bytearray)):
            try:
                return Image.open(BytesIO(src)).convert("RGB")
            except Exception:
                return None

        # 3) String cases
        if isinstance(src, str):
            # data URL
            if src.startswith("data:image"):
                try:
                    _, b64 = src.split(",", 1)
                    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                except Exception:
                    return None
            # http(s)
            if src.startswith("http://") or src.startswith("https://"):
                try:
                    r = requests.get(src, timeout=10)
                    r.raise_for_status()
                    return Image.open(BytesIO(r.content)).convert("RGB")
                except Exception:
                    return None
            # local path
            try:
                return Image.open(src).convert("RGB")
            except Exception:
                return None

        return None
    
    def _resize_to_max(self, img, max_side=1664):
        """Keep aspect ratio. If the longer side > max_side, downscale; else return as is.
        Prints original and new sizes once per call."""
        if not hasattr(img, "size"):
            return img
        w, h = img.size
        print(f"[VReST-img] original size: {w}x{h}")
        long_edge = max(w, h)
        if long_edge <= max_side:
            print(f"[VReST-img] capped size: {w}x{h} (no resize)")
            return img
        scale = max_side / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img2 = img.resize((new_w, new_h))
        print(f"[VReST-img] resized to:  {new_w}x{new_h} (max_side={max_side})")
        return img2


    def count_image_tokens(self, messages):
        """
        Count image tokens with caching:
        - Build keys for all image content
        - For cached keys: reuse token counts
        - For misses: load+resize, preprocess, compute tokens (pixels or embeddings), then cache
        """
        # Ensure we have a processor
        proc = getattr(self, "processor", None)
        if proc is None:
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.llm_cfg.path)
                self._vision_patch = int(getattr(getattr(self.processor, "image_processor", None), "patch_size", 14))
                proc = self.processor
            except Exception:
                return 0

        # Collect (key, PIL) pairs — only load PIL when we need to compute
        image_items = []  # list of (key, content_item)
        for m in messages or []:
            content = m.get("content", "")
            if isinstance(content, list):
                for c in content:
                    k = self._image_key(c)
                    if k is not None:
                        image_items.append((k, c))

        if not image_items:
            return 0

        import torch

        def tokens_from_pixels_tensor(pv):
            if pv.ndim == 4:
                _, _, H, W = pv.shape
            elif pv.ndim == 3:
                _, H, W = pv.shape
            else:
                return 0
            patch = max(int(getattr(self, "_vision_patch", 14) or 14), 1)
            tokens = (H // patch) * (W // patch)
            return int(tokens)

        def tokens_from_emb_tensor(pv):
            if pv.ndim == 2:      # (T, D)
                T = pv.shape[0]
            elif pv.ndim == 3:    # (1, T, D) or (B, T, D)
                T = pv.shape[1]
            else:
                return 0
            return int(T)

        def compute_one_image_tokens_from_pil(img):
            # resize-cap
            # img = self._resize_to_max(img, max_side=1664)
            iproc = getattr(proc, "image_processor", None)
            # Path A: image_processor
            try:
                if iproc is not None:
                    out = iproc(images=img, return_tensors="pt")
                    pv = out.get("pixel_values", None)
                    if isinstance(pv, torch.Tensor):
                        if pv.ndim in (3, 4):
                            return tokens_from_pixels_tensor(pv)
                        if pv.ndim in (2, 3):
                            return tokens_from_emb_tensor(pv)
            except Exception:
                pass
            # Path B: unified processor
            try:
                out = proc(images=img, text=[""], return_tensors="pt")
                pv = (
                    out.get("pixel_values", None)
                    or out.get("images", None)
                    or out.get("pixel_values_vit", None)
                )
                if isinstance(pv, torch.Tensor):
                    if pv.ndim in (3, 4):
                        return tokens_from_pixels_tensor(pv)
                    if pv.ndim in (2, 3):
                        return tokens_from_emb_tensor(pv)
                elif isinstance(pv, (list, tuple)) and len(pv) > 0 and isinstance(pv[0], torch.Tensor):
                    t0 = pv[0]
                    if t0.ndim in (3, 4):
                        return tokens_from_pixels_tensor(t0)
                    if t0.ndim in (2, 3):
                        return tokens_from_emb_tensor(t0)
            except Exception:
                pass
            return 0

        total_tokens = 0
        misses = []

        # 1) Sum cached hits; collect misses
        for k, c in image_items:
            if k in self._img_token_cache:
                total_tokens += self._img_token_cache[k]
            else:
                misses.append((k, c))

        # 2) Compute for misses individually, then cache
        for k, c in misses:
            pil_img = self._load_pil_from_content(c)
            if pil_img is None:
                self._img_token_cache[k] = 0
                continue
            tok = compute_one_image_tokens_from_pil(pil_img)
            self._img_token_cache[k] = int(tok)
            total_tokens += int(tok)

        return int(total_tokens)



    def get_reward(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=5):
        response = None
        for attempt in range(self.attempt_num):
            try:
                with self.lock:
                    client = self.get_least_request_client()
                    self.request_count[client] += 1
                    
                if json_format is not None:
                
                    chat_response = client.beta.chat.completions.parse(
                        n=n,
                        model=self.llm_cfg.name,
                        messages=messages,
                        response_format=json_format if json_format else None,
                        top_p=self.llm_cfg.top_p,
                        temperature=self.llm_cfg.temperature,
                        max_tokens=self.llm_cfg.max_output_len,
                        extra_body={
                            "repetition_penalty": self.llm_cfg.repetition_penalty,
                            "guided_choice": ["Yes", "No"],
                            "add_generation_prompt": False if continue_final_message else True,
                            "continue_final_message": continue_final_message,
                        },
                        logprobs=True if top_logprobs > 0 else False,
                        top_logprobs=top_logprobs,
                    )

                else:

                    chat_response = client.chat.completions.create(
                        n=n,
                        model=self.llm_cfg.name,
                        messages=messages,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=self.llm_cfg.max_output_len,
                        extra_body={
                            "repetition_penalty": self.llm_cfg.repetition_penalty,
                            "guided_choice": ["Yes", "No"],
                            "add_generation_prompt": False if continue_final_message else True,
                            "continue_final_message": continue_final_message,
                        },
                        logprobs=True if top_logprobs > 0 else False,
                        top_logprobs=top_logprobs,
                    )
                # 提取Yes和No的logprobs
                logprobs = chat_response.choices[0].logprobs.content[0].top_logprobs if chat_response.choices[0].logprobs else None
                logits = {}
                for token in logprobs:
                    logits[token.token] = token.logprob
                        
                yes_logprob = logits['Yes'] if 'Yes' in logits else 0.0
                no_logprob = logits['No'] if 'No' in logits else 0.0
                
                # 转换为0-1的概率
                yes_prob = math.exp(yes_logprob) / (math.exp(yes_logprob) + math.exp(no_logprob))
                no_prob = math.exp(no_logprob) / (math.exp(yes_logprob) + math.exp(no_logprob))
                
                response = {
                    "yes_prob": yes_prob,
                    "no_prob": no_prob,
                }
                
                with self.lock:
                    self.request_count[client] -= 1
                    
                return response
            
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt == self.attempt_num-1:
                    print(f"Failed to get reward after {self.attempt_num} attempts. Error: {str(e)}")
                    raise ValueError(f"Failed to get reward after {self.attempt_num} attempts. Error: {str(e)}")

    def get_response(self, messages=None, json_format=None, n=1, continue_final_message=False, top_logprobs=0, stop_token=None):

        if self.llm_cfg.call_method == "api":
            response = None
            for attempt in range(self.attempt_num):
                try:
                    with self.lock:
                        client = self.get_least_request_client()
                        self.request_count[client] += 1

                    if json_format is not None:
                    
                        chat_response = client.beta.chat.completions.parse(
                            n=n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            response_format=json_format if json_format else None,
                            top_p=self.llm_cfg.top_p,
                            temperature=self.llm_cfg.temperature,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": self.llm_cfg.repetition_penalty,
                                "add_generation_prompt": False if continue_final_message else True,
                                "continue_final_message": continue_final_message,
                            },
                            stop=stop_token,
                        )

                    else:

                        chat_response = client.chat.completions.create(
                            n=n,
                            model=self.llm_cfg.name,
                            messages=messages,
                            temperature=self.llm_cfg.temperature,
                            top_p=self.llm_cfg.top_p,
                            max_tokens=self.llm_cfg.max_output_len,
                            extra_body={
                                "repetition_penalty": self.llm_cfg.repetition_penalty,
                                "add_generation_prompt": False if continue_final_message else True,
                                "continue_final_message": continue_final_message,
                                # "echo": True,
                            },
                            stop=stop_token,
                            # seed=self.llm_cfg.seed,
                        )

                    response = {
                        "response": [choice.message.content for choice in chat_response.choices] if n > 1 else chat_response.choices[0].message.content
                    }
                
                    with self.lock:
                        self.request_count[client] -= 1
                    
                    return response
                
                except Exception as e:
                    print(f"Error: {str(e)}")
                    if attempt == self.attempt_num-1:
                        print(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
                        raise ValueError(f"Failed to get response after {self.attempt_num} attempts. Error: {str(e)}")
        
        elif self.llm_cfg.call_method == "local":

            # processor = AutoProcessor.from_pretrained(self.llm_cfg.path)
            processor = self.processor  # already built in __init__
            if processor is None:
                processor = AutoProcessor.from_pretrained(self.llm_cfg.path)
                self.processor = processor
                self._vision_patch = int(getattr(getattr(processor, "image_processor", None), "patch_size", 14))
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }

            json_shape = '''{
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string"
                            },
                            "content": {
                                "type": "string"
                            },
                            "next_action": {
                                "type": "string"
                            }
                        },
                        "required": ["title", "content", "next_action"]
                    }'''

            outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params, guided_options_request={
                    "guided_json": json_shape,
                    # "guided_json_object": json_format
                } if json_format else None, 
                use_tqdm=False)
            generated_text = outputs[0].outputs[0].text

            response = {
                "response": generated_text
            }
            return response
        
        else:
            raise ValueError("Invalid call method")


# multi agents

class Multi_Agents_Model():
    def __init__(self, cfg):
        self.cfg = cfg

        self.planner = self.init_agent("planner")
        self.reasoner = self.init_agent("reasoner")
        self.verifier = self.init_agent("verifier")
        self.judger = self.init_agent("judger")
        self.summarizer = self.init_agent("summarizer")
    
    def init_agent(self, agent_name):
        prompt_cfg = self.cfg.prompt_method
        agents_cfg = self.cfg.llm.agents
        
        llm_cfg = agents_cfg[prompt_cfg[agent_name]]
        base_model_name = llm_cfg.name
        return model_name_map[base_model_name](llm_cfg)

model_name_map = {
    "Qwen2-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen2-VL-2B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-3B-Instruct": Qwen2_VL_Model,
    "Qwen2.5-VL-7B-Instruct": Qwen2_VL_Model,
    "Qwen3_VL_4B_Instruct": Qwen2_VL_Model,
    "Qwen3_VL_8B_Instruct": Qwen2_VL_Model,
    "Qwen2.5-7B-Instruct": Qwen25_Model,
    "Qwen2.5-14B-Instruct": Qwen25_Model,
    "multi_agents": Multi_Agents_Model,
}

class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        self.prompt_cfg = cfg.prompt_method
        if cfg.llm.name == "multi_agents":
            self.model = model_name_map[cfg.llm.name](cfg)
        else:
            self.model = model_name_map[cfg.llm.name](cfg.llm)

    def get_response(self, input_text, image_path):
        try:
            if self.prompt_cfg.name == "cantor":
                return cantor_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "cot":
                return cot_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "qa":
                return qa_forward(input_text, image_path, self.model)
            elif self.prompt_cfg.name == "our":
                return our_forward(input_text, image_path, self.model)
            elif "our1" in self.prompt_cfg.name:
                return our1_forward(input_text, image_path, self.model)
            elif "our2" in self.prompt_cfg.name:
                return our2_forward(input_text, image_path, self.model)
            elif "our3" in self.prompt_cfg.name:
                return our3_forward(input_text, image_path, self.model, self.cfg)
            elif "our4" in self.prompt_cfg.name:
                return our4_forward(input_text, image_path, self.model, self.cfg)
            elif "our5" in self.prompt_cfg.name:
                return our5_forward(input_text, image_path, self.model, self.cfg)
            elif "our6" in self.prompt_cfg.name:
                return our6_forward(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mcts", "mctsv1"]:
                return mcts_forward(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv2"]:
                return mcts_forward_v2(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv3"]:
                return mcts_forward_v3(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv4"]:
                return mcts_forward_v4(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv5"]:
                return mcts_forward_v5(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv6"]:
                return mcts_forward_v6(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv7"]:
                return mcts_forward_v7(input_text, image_path, self.model, self.cfg)
            elif self.prompt_cfg.name in ["mctsv8"]:
                return mcts_forward_v8(input_text, image_path, self.model, self.cfg)
            else:
                raise ValueError("Invalid prompt method")
        except Exception as e:
            print(e)
            raise e

