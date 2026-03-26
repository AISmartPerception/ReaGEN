from prompts.stage_n.prompt_teacher import compose_prompt
import json
import io
import base64
from PIL import Image
from typing import Dict, Any
from json import dumps
import re
from typing import Tuple

# from google.genai import types
# from google.genai.types import GenerationConfig


def _pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


class Teacher:
    def __init__(self, teacher, config, logger, chain):
        # self.model_nm = teacher[0]
        # self.client = teacher[1]
        if isinstance(teacher[0], str):
            self.model_nm = teacher[0]
            self.client = teacher[1]
        else:
            self.model_nm = "Qwen"
            self.processor, self.model = teacher

        self.system_prompt, self.use_instruction_lines = compose_prompt(config["inference"]["stages_pool"], 
                                                                        config["inference"]["attn"], 
                                                                        config["inference"]["prompt_refine"],
                                                                        config["inference"]["CoT"],
                                                                        config["inference"]["mini"])
    
        self.config = config
        self.logger = logger
        self.chain = chain
        
    
    def build_user_content_refine(self, sample: Dict[str, Any], current_CoT_info, past_iterations, optimal_CoT_dict, images):
        """
        Build the multimodal 'content' list for the user turn:
        [{"type":"text",...}, {"type":"image_url",...}]
        """

        state_lines = [
            f'"Stage Pool": {dumps(current_CoT_info.get("Stage Pool", []))}',
            # f'"stage_system_prompts": {dumps(current_CoT_info.get("Current stage_system_prompts", {}))}',
            # f'"cot_struct": {dumps(current_CoT_info.get("Current CoT_structure", []))}',
        ]
        state_block = "{\n  " + ",\n  ".join(state_lines) + "\n}"

        past_iterations_lines = ""
        past_iterations_lines += (
            f"CoT with the best score so far: "
            f"score={optimal_CoT_dict['score']} | "
            f"stages={','.join(map(str, optimal_CoT_dict['stages']))}, FINAL | "
            f"feedback={optimal_CoT_dict['feedback']}"
        ) + "\n"


        non_repeat_dict = {}
        for i, iteration in enumerate(past_iterations):
            if iteration["iteration"] == len(past_iterations) - 1:
                cat = "Current"
                past_iterations_lines += (
                    f"[{cat} iter] "
                    f"score={iteration['score']} | "
                    f"stages={','.join(map(str, iteration['stages']))}, FINAL | "
                    f"feedback={iteration['feedback']}"
                ) + "\n"
            elif iteration["iteration"] == len(past_iterations) - 2:
                cat = "Prev"
                past_iterations_lines += (
                    f"[{cat} iter] "
                    f"score={iteration['score']} | "
                    f"stages={','.join(map(str, iteration['stages']))}, FINAL | "
                    f"feedback={iteration['feedback']}"
                ) + "\n"
  
            

            if ','.join(map(str, iteration['stages'])) in non_repeat_dict:
                if non_repeat_dict[','.join(map(str, iteration['stages']))] < iteration['score']:
                    non_repeat_dict[','.join(map(str, iteration['stages']))] = iteration['score']
            else:
                non_repeat_dict[','.join(map(str, iteration['stages']))] = iteration['score']


        count = 0
        forbidden_list = ""
        for k, v in non_repeat_dict.items():
            count += 1
            if count == len(list(non_repeat_dict.keys())):
                forbidden_list+=f"[{k}, FINAL] with score {v}.\n"
            else:
                forbidden_list+=f"[{k}, FINAL] with score {v},\n"

        text = (
            "\n".join(self.use_instruction_lines) + "\n\n" +
            f"Question: {repr(sample.get('question'))}\n\n" +
            "EVOLUTION ITERATIONS:\n" + past_iterations_lines + "\n\n" +
            f"CoTs have been explored: (Avoid repeating these sequences redundant exploitation).\n {forbidden_list}Avoid repeating or permuting these sequences."
        )
        

        self.logger.info(f"\n\nTeacher input: {text}\n\n")

        if "Qwen" in self.model_nm:
            user_content = []
        
            # Add images to user content
            for i, img in enumerate(images):
                user_content.append({"type": "image"})
                if len(images) > 1:
                    # Add context about which image this is
                    if i == 0:
                        user_content.append({"type": "text", "text": f"IMG {i} = original image\n"})
                    else:
                        user_content.append({"type": "text", "text": f"IMG {i} = ROI crop\n"})

            user_content.append({"type": "text", "text": text})
            content = user_content
  
        elif "gpt" in self.model_nm:
            content = [
                {"type": "text", "text": text}
                # {"type": "image_url", "image_url": {"url": _pil_to_data_uri(sample["image"], fmt="PNG")}},
            ]
            
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_uri(img, fmt="PNG")}
                })
        

        elif "gemini" in self.model_nm:
            content = [sample["image"], text]
        

        return content


    def build_CoT_info(self):
        current_info = {}
        current_info["Stage Pool"] = self.config["inference"]["stages_pool"]
        current_info["Current CoT_structure"] = [stage.name for stage in self.chain.stages]
        # current_info["Current stage_system_prompts"] = {stage.name: stage.sys_prompt for stage in self.chain.stages}
        current_info["Current final_stage_system_prompt"] = self.chain.final_stage.sys_prompt

        return current_info

    def extract_thinking_and_json(self, text: str) -> Tuple[str, Dict[str, Any]]:
        text = text.strip()
        parts = re.split(r"</think>\s*", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            thinking_text, json_part = parts
        else:
            match = re.search(r"(\{[\s\S]*\})", text)
            if match:
                thinking_text = text[:match.start()].strip()
                json_part = match.group(1)
            else:
                thinking_text = text
                json_part = "{}"


        try:
            final_output = json.loads(json_part)

        except json.JSONDecodeError as e:
            matches = re.findall(r"(\{[\s\S]*?\})", text)
            final_output = {}
            for m in reversed(matches):
                try:
                    final_output = json.loads(m)
                    break
                except json.JSONDecodeError:
                    continue
            if not final_output:
                final_output = {"raw_json": json_part}
            # raise ValueError(f"Invalid JSON from teacher: {e}\nRaw output:\n{output}")

        # stage_prompts = parsed.get("stage_prompts")
        evo_finish = final_output.get("evo_finish")
        reason = final_output.get("reason")
        if evo_finish == "True":
            evo_finish = True

            return thinking_text, {
                "evo_finish": evo_finish,
                "reason": reason,
                # "stage_prompts": stage_prompts
            }

        elif evo_finish == "False":
            evo_finish = False
            cot = final_output.get("cot")

            return thinking_text, {
                "cot": cot,
                "evo_finish": evo_finish,
                "reason": reason,
                # "stage_prompts": stage_prompts
            }


        

    def _decode_new_tokens(self, processor, gen, input_len: int) -> str:
        """Helper function to decode new tokens from generation output"""
        tok = processor.tokenizer
        seqs = gen.sequences if hasattr(gen, "sequences") else gen
        new_tokens = seqs[:, input_len:]
        
        new_tokens = new_tokens.detach().cpu()
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        text = texts[0] if texts else ""
        text = text.strip()
        text = text.replace(tok.eos_token or "", "").strip()

        return text

    def generate(self, sample, past_iterations, optimal_CoT_dict, images):
        current_CoT_info = self.build_CoT_info()
        if self.model_nm == "Qwen":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_user_content_refine(sample, current_CoT_info, past_iterations, optimal_CoT_dict, images)}
            ]
            
            self.logger.info(f"Teacher messages: {messages}")
            
            chat = self.processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
            # self.logger.info(f"Teacher chat: {chat}")
            inputs = self.processor(text=[chat], images=images, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
            output = self.model.generate(**inputs, 
                                         max_new_tokens=self.config["inference"]["TEACHER"]["max_new_tokens"],
                                         do_sample=self.config["inference"]["TEACHER"]["do_sample"],
                                         num_beams=self.config["inference"]["TEACHER"]["num_beams"],
                                         temperature=self.config["inference"]["TEACHER"]["temperature"] if self.config["inference"]["TEACHER"]["temperature"] > 0.0 else None)
            # output = self.processor.decode(output[0], skip_special_tokens=True)
            output = self._decode_new_tokens(self.processor, output, inputs["input_ids"].shape[1])
            # self.logger.info(f"\n\nTeacher output: {output}\n\n")

        elif "gpt" in self.model_nm:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_user_content_refine(sample, current_CoT_info, past_iterations, optimal_CoT_dict, images)}
            ]
            self.logger.info(f"System prompt: {self.system_prompt}")
            
            
            resp = self.client.chat.completions.create(
            # resp = self.client.models.generate_content(
                model=self.model_nm,
                messages=messages,
                temperature=self.config["inference"]["TEACHER"]["temperature"],
                max_tokens=self.config["inference"]["TEACHER"]["max_new_tokens"]
            )
            
            output = resp.choices[0].message.content
            # 
            # self.logger.info(f"\n\nTeacher output: {output}\n\n")

        # elif "gemini" in self.model_nm:
        #     system_instruction = self.system_prompt
            # user_contents = self.build_user_content_refine(sample, feedback, current_CoT_info, past_iterations)

 
            # generation_config = types.GenerateContentConfig(
            #     temperature=self.config["inference"]["TEACHER"]["temperature"],
            #     max_output_tokens=self.config["inference"]["TEACHER"]["max_new_tokens"], 
            #     system_instruction=str(system_instruction)
      
            # )

            # resp = self.client.models.generate_content(
            #     model=self.model_nm,
            #     contents=["Explain how AI works"]
            # )
            # output = resp.text
            # self.logger.info(f"\n\nTeacher output: {output}\n\n")

    
        return self.extract_thinking_and_json(output)
    
    
    # def render_output(self, output):
    #     try:
    #         if isinstance(output, dict):
    #             parsed = json.loads(output)
    #         else:
    #             if not isinstance(output, str):
    #                 raise TypeError("output must be str or dict")

    #             s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", output.strip())
    #             s = re.sub(r"\n\s*```\s*$", "", s)


    #             parsed = json.loads(s)

    #     except json.JSONDecodeError as e:
    #         raise ValueError(f"Invalid JSON from teacher: {e}\nRaw output:\n{output}")

    #     stage_prompts = parsed.get("stage_prompts")
    #     evo_finish = parsed.get("evo_finish")
    #     reason = parsed.get("reason")
    #     if evo_finish == "True":
    #         evo_finish = True

    #         return {
    #             "evo_finish": evo_finish,
    #             "reason": reason,
    #             "stage_prompts": stage_prompts
    #         }

    #     elif evo_finish == "False":
    #         evo_finish = False
    #         cot = parsed.get("cot")

    #         return {
    #             "cot": cot,
    #             "evo_finish": evo_finish,
    #             "reason": reason,
    #             "stage_prompts": stage_prompts
    #         }


    def apply_edits(self, teacher_output, stages, base_system_prompts):
        # self.logger.info(f"Applying edits: {teacher_output}")
        # self.logger.info(f"Stages: {stages}")
        cot = teacher_output.get("cot")
        

        new_stages = list(cot)
    
        # if "DIRECT" in new_stages:
        #     new_stages.remove("DIRECT")
        

        if self.config["inference"]["prompt_refine"]:
            stage_prompts = teacher_output.get("stage_prompts")
            new_prompts = {}
            for stage in new_stages:
                if "DIRECT" in stage:
                    continue
                if stage in base_system_prompts:
                    new_prompts[stage] = stage_prompts[stage]
                elif stage in self.config["inference"]["stages_pool"]:
                    new_prompts[stage] = stage_prompts[stage]
                else:
                    raise ValueError(f"No prompt found for stage '{stage}'")

            return new_stages[:-1], new_prompts


        return new_stages[:-1], base_system_prompts

#         File "/home/XXX/gepa/vlm_evo/prompts/teacher.py", line 154, in apply_edits
#     if teacher_output["position"].has_key("index"):
# AttributeError: 'dict' object has no attribute 'has_key'


import random
from copy import deepcopy
from typing import List, Dict, Any

def edit_chain(
    chain: Dict[str, Any],
    searched_chains: List[Dict[str, Any]],
    config: Dict[str, Any],
):
    """
    Two-level search-aware chain editor (randomized):
    1. Determine available ops (add, delete, swap)
    2. Randomly choose one op
    3. Explore edits under that op in shuffled order until a unique new chain is found.

    Efficient and stochastic exploration to improve diversity.
    """

    # Collect already explored final sequences
    existing_final_seqs = {tuple(chain) for chain in searched_chains}

    orig_seq = list(chain.get("stage_seq", []))
    n = len(orig_seq)

    remaining_stages = [s for s in config["inference"]["stages_pool"] if s not in orig_seq]

    if chain['score'] == 0 and random.random() < 0.4:
        return orig_seq, "Update Reward", False

    # ---------- Level 1: determine feasible ops ----------
    available_ops = []
    if remaining_stages:
        available_ops.append("add")
    if n > 1:
        available_ops.extend(["delete", "swap"])

    if not available_ops:
        return orig_seq, None, False

    # Randomly pick one operation to explore
    op = random.choice(available_ops)
    new_seq = None

    # ---------- Level 2: explore the chosen op ----------
    if op == "add":
        # Shuffle both stage order and insertion positions
        shuffled_stages = random.sample(remaining_stages, len(remaining_stages))
        positions = list(range(n + 1))
        random.shuffle(positions)

        for stage_to_add in shuffled_stages:
            for pos in positions:
                seq = orig_seq[:pos] + [stage_to_add] + orig_seq[pos:]
                if tuple(seq) not in existing_final_seqs:
                    new_seq = seq
                    break
            if new_seq:
                break

    elif op == "delete" and n > 1:
        # Shuffle delete positions
        delete_positions = list(range(n))
        random.shuffle(delete_positions)

        for i in delete_positions:
            seq = orig_seq[:i] + orig_seq[i + 1:]
            if seq and tuple(seq) not in existing_final_seqs:
                new_seq = seq
                break

    elif op == "swap" and n > 1:
        # Shuffle all pair combinations
        indices = list(range(n))
        random.shuffle(indices)
        pairs = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
        random.shuffle(pairs)

        for i, j in pairs:
            if orig_seq[i] == orig_seq[j]:
                continue
            seq = deepcopy(orig_seq)
            seq[i], seq[j] = seq[j], seq[i]
            if tuple(seq) not in existing_final_seqs:
                new_seq = seq
                break

    # ---------- Return result ----------
    if new_seq is None:
        return orig_seq, op, False

    return new_seq, op, True


# def edit_chain_attn(
#     chain: Dict[str, Any],
#     attn_dict: Dict[str, Any],
#     searched_chains: List[Dict[str, Any]],
#     config: Dict[str, Any],
# ):
#     """
#     Two-level search-aware chain editor (randomized):
#     1. Determine available ops (add, delete, swap)
#     2. Randomly choose one op
#     3. Explore edits under that op in shuffled order until a unique new chain is found.

#     Efficient and stochastic exploration to improve diversity.
#     """
#     cot_str = ",".join(chain['stage_seq'])

#     current_attn_dict = attn_dict[cot_str]
#     contribution_dict = current_attn_dict.get("contribution_dict", {})
#     A = current_attn_dict.get("A", {})
#     a_to_final = current_attn_dict.get("a_to_final", {})

#     pred_score = chain['score']

#     # Collect already explored final sequences
#     existing_final_seqs = {tuple(chain) for chain in searched_chains}

#     orig_seq = list(chain.get("stage_seq", []))
#     n = len(orig_seq)

#     remaining_stages = [s for s in config["inference"]["stages_pool"] if s not in orig_seq]

#     if chain['score'] == 0 and random.random() < 0.25:
#         return orig_seq, "Update Reward", False

#     # ---------- Level 1: determine feasible ops ----------
#     available_ops = []
#     if remaining_stages:
#         available_ops.append("add")
#     if n > 1:
#         available_ops.extend(["delete", "swap"])

#     if not available_ops:
#         return orig_seq, None, False

#     # Randomly pick one operation to explore
#     op = random.choice(available_ops)
#     new_seq = None

#     # ---------- Level 2: explore the chosen op ----------
#     if op == "add":
#         # Shuffle both stage order and insertion positions
#         shuffled_stages = random.sample(remaining_stages, len(remaining_stages))
#         positions = list(range(n + 1))
#         random.shuffle(positions)

#         for stage_to_add in shuffled_stages:
#             for pos in positions:
#                 seq = orig_seq[:pos] + [stage_to_add] + orig_seq[pos:]
#                 if tuple(seq) not in existing_final_seqs:
#                     new_seq = seq
#                     break
#             if new_seq:
#                 break

#     elif op == "delete" and n > 1:
#         # Shuffle delete positions
#         delete_positions = list(range(n))
#         random.shuffle(delete_positions)

#         for i in delete_positions:
#             seq = orig_seq[:i] + orig_seq[i + 1:]
#             if seq and tuple(seq) not in existing_final_seqs:
#                 new_seq = seq
#                 break

#     elif op == "swap" and n > 1:
#         # Shuffle all pair combinations
#         indices = list(range(n))
#         random.shuffle(indices)
#         pairs = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
#         random.shuffle(pairs)

#         for i, j in pairs:
#             if orig_seq[i] == orig_seq[j]:
#                 continue
#             seq = deepcopy(orig_seq)
#             seq[i], seq[j] = seq[j], seq[i]
#             if tuple(seq) not in existing_final_seqs:
#                 new_seq = seq
#                 break

#     # ---------- Return result ----------
#     if new_seq is None:
#         return orig_seq, op, False

#     return new_seq, op, True


import random

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

def edit_chain_attn(
    chain: Dict[str, Any],
    attn_dict: Dict[str, Any],
    searched_chains: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[str], Optional[str], bool]:
   

    # ---------- helpers ----------
    def importance_from_contrib(seq, contrib_map):
        return [float(contrib_map.get(s, {}).get("total", 0.0)) for s in seq]

    def best_bridge(A, s):
        """argmax over i<j of A[i][j] * s[j]; returns (i, j) or None."""
        if A is None: return None
        N = len(s)
        if len(A) != N: return None
        best, best_val = None, -1.0
        for i in range(N - 1):
            row = A[i]
            for j in range(i + 1, N):
                v = float(row[j]) * s[j]
                if v > best_val:
                    best_val, best = v, (i, j)
        return best

    def weakest_adjacent(A, s):
        """Return adjacent pair (i, i+1) minimizing A[i][i+1] * s[i+1].
           If A missing, minimize s[i+1]."""
        N = len(s)
        if N < 2: return None
        best, best_val = None, float("inf")
        for i in range(N - 1):
            v = (float(A[i][i+1]) * s[i+1]) if (A is not None and len(A) == N) else s[i+1]
            if v < best_val:
                best_val, best = v, (i, i + 1)
        return best

    def pick_op(available_ops, final_correct: Optional[bool]):
        if final_correct is True:
            pref = {"delete": 0.6, "swap": 0.3, "add": 0.1}
        elif final_correct is False:
            pref = {"add": 0.6, "swap": 0.25, "delete": 0.15}
        else:
            pref = {"add": 1/3, "delete": 1/3, "swap": 1/3}
        weights = [pref.get(op, 0.0) for op in available_ops]
        if not any(weights):
            weights = [1.0 / len(available_ops)] * len(available_ops)
        return random.choices(available_ops, weights=weights, k=1)[0]

    # ---------- fetch attention/contribution data ----------
    cot_str = ",".join(chain.get("stage_seq", []))
    cur = attn_dict[cot_str]
    contribution_dict = cur.get("contribution_dict", {})    
    A = cur.get("A", None)                                
    # a_to_final = cur.get("a_to_final", {})                 # available if you need it

    pred_score = float(chain.get("score", 0.0))
    reward = float(chain.get("reward", 0.0))
    existing_final_seqs = {tuple(c.get("stage_seq", [])) for c in searched_chains}

    orig_seq = list(chain.get("stage_seq", []))
    n = len(orig_seq)
    remaining_stages = [s for s in config["inference"]["stages_pool"] if s not in orig_seq]

    # Optional fast-return you had
    if reward == 0.0 and random.random() < 0.25:
        return orig_seq, "Update Reward", False

    # ---------- determine feasible ops ----------
    available_ops: List[str] = []
    if remaining_stages:
        available_ops.append("add")
    if n > 1:
        available_ops.extend(["delete", "swap"])
    if not available_ops:
        return orig_seq, None, False

    # Decide correctness flag
    # thr = config.get("search", {}).get("correct_threshold", None)
    thr = 0.8
    final_correct = (pred_score >= thr)

    # Choose op with simple preference
    op = pick_op(available_ops, final_correct)

    # Importance vector aligned with current sequence
    s = importance_from_contrib(orig_seq, contribution_dict)
    new_seq: Optional[List[str]] = None

    # ---------- ADD ----------
    if op == "add" and remaining_stages:
        stage_to_add = random.choice(remaining_stages)

        # Position: strongest bridge i->j (A[i][j]*s[j]); fallback before argmax importance
        if n >= 1:
            pos = None
            br = best_bridge(A, s)
            if br is not None:
                _, j = br
                pos = j  # insert just before j
            else:
                pos = int(max(range(n), key=lambda t: s[t]))
        else:
            pos = 0

        seq = orig_seq[:pos] + [stage_to_add] + orig_seq[pos:]
        if tuple(seq) not in existing_final_seqs:
            new_seq = seq

        # Fallback: randomized insertion if collision
        if new_seq is None:
            positions = list(range(n + 1))
            random.shuffle(positions)
            for p in positions:
                seq = orig_seq[:p] + [stage_to_add] + orig_seq[p:]
                if tuple(seq) not in existing_final_seqs:
                    new_seq = seq
                    break

    # ---------- DELETE ----------
    elif op == "delete" and n > 1:
        # Delete the least-important stage (argmin total)
        idx = int(min(range(n), key=lambda t: s[t]))
        seq = orig_seq[:idx] + orig_seq[idx + 1:]
        if seq and tuple(seq) not in existing_final_seqs:
            new_seq = seq
        else:
            # Fallback randomized delete
            delete_positions = list(range(n))
            random.shuffle(delete_positions)
            for i in delete_positions:
                seq = orig_seq[:i] + orig_seq[i + 1:]
                if seq and tuple(seq) not in existing_final_seqs:
                    new_seq = seq
                    break

    # ---------- SWAP ----------
    elif op == "swap" and n > 1:
        pair = weakest_adjacent(A, s)
        if pair:
            i, j = pair
            seq = deepcopy(orig_seq)
            seq[i], seq[j] = seq[j], seq[i]
            if tuple(seq) not in existing_final_seqs:
                new_seq = seq

        # Fallback randomized swap
        if new_seq is None:
            idxs = list(range(n))
            random.shuffle(idxs)
            pairs = [(idxs[a], idxs[b]) for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
            random.shuffle(pairs)
            for i, j in pairs:
                if orig_seq[i] == orig_seq[j]:
                    continue
                seq = deepcopy(orig_seq)
                seq[i], seq[j] = seq[j], seq[i]
                if tuple(seq) not in existing_final_seqs:
                    new_seq = seq
                    break

    # ---------- return ----------
    if new_seq is None:
        return orig_seq, op, False
    return new_seq, op, True



def edit_chain_del_only(
    chain: Dict[str, Any],
    attn_dict: Dict[str, Any],
    searched_chains: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[str], Optional[str], bool]:
   

    # ---------- helpers ----------
    def importance_from_contrib(seq, contrib_map):
        return [float(contrib_map.get(s, {}).get("total", 0.0)) for s in seq]

    def best_bridge(A, s):
        """argmax over i<j of A[i][j] * s[j]; returns (i, j) or None."""
        if A is None: return None
        N = len(s)
        if len(A) != N: return None
        best, best_val = None, -1.0
        for i in range(N - 1):
            row = A[i]
            for j in range(i + 1, N):
                v = float(row[j]) * s[j]
                if v > best_val:
                    best_val, best = v, (i, j)
        return best

    def weakest_adjacent(A, s):
        """Return adjacent pair (i, i+1) minimizing A[i][i+1] * s[i+1].
           If A missing, minimize s[i+1]."""
        N = len(s)
        if N < 2: return None
        best, best_val = None, float("inf")
        for i in range(N - 1):
            v = (float(A[i][i+1]) * s[i+1]) if (A is not None and len(A) == N) else s[i+1]
            if v < best_val:
                best_val, best = v, (i, i + 1)
        return best

    def pick_op(available_ops, final_correct: Optional[bool]):
        if final_correct is True:
            pref = {"delete": 0.6, "swap": 0.3, "add": 0.1}
        elif final_correct is False:
            pref = {"add": 0.6, "swap": 0.25, "delete": 0.15}
        else:
            pref = {"add": 1/3, "delete": 1/3, "swap": 1/3}
        weights = [pref.get(op, 0.0) for op in available_ops]
        if not any(weights):
            weights = [1.0 / len(available_ops)] * len(available_ops)
        return random.choices(available_ops, weights=weights, k=1)[0]

    # ---------- fetch attention/contribution data ----------
    cot_str = ",".join(chain.get("stage_seq", []))
    cur = attn_dict[cot_str]
    contribution_dict = cur.get("contribution_dict", {})    
    A = cur.get("A", None)                                
    # a_to_final = cur.get("a_to_final", {})                 # available if you need it

    pred_score = float(chain.get("score", 0.0))
    reward = float(chain.get("reward", 0.0))
    existing_final_seqs = {tuple(c.get("stage_seq", [])) for c in searched_chains}

    orig_seq = list(chain.get("stage_seq", []))
    n = len(orig_seq)
    remaining_stages = [s for s in config["inference"]["stages_pool"] if s not in orig_seq]

    # Optional fast-return you had
    # if reward == 0.0 and random.random() < 0.25:
    #     return orig_seq, "Update Reward", False

    # ---------- determine feasible ops ----------
    available_ops: List[str] = []
    # if remaining_stages:
    #     available_ops.append("add")
    # if n > 1:
        # available_ops.extend(["delete", "swap"])
    # if not available_ops:
        # return orig_seq, None, False
    available_ops.extend(["delete"])

    # Decide correctness flag
    # thr = config.get("search", {}).get("correct_threshold", None)
    thr = 0.8
    final_correct = (pred_score >= thr)

    # Choose op with simple preference
    op = pick_op(available_ops, final_correct)

    # Importance vector aligned with current sequence
    s = importance_from_contrib(orig_seq, contribution_dict)
    new_seq: Optional[List[str]] = None

   
    if op == "delete" and n > 1:
        # Delete the least-important stage (argmin total)
        
        idx = int(min(range(n), key=lambda t: s[t]))
        # idx = int(max(range(n), key=lambda t: s[t]))
        new_seq = orig_seq[:idx] + orig_seq[idx + 1:]
        if new_seq is None:
            print(f"No new sequence found for delete operation")
        # if seq and tuple(seq) not in existing_final_seqs:
        #     new_seq = seq
        

    # # ---------- return ----------
    # if new_seq is None:
    #     return orig_seq, op, False
    return new_seq, op, True
