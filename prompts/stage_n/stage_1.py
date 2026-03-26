from typing import Dict, Any, Optional, Tuple
import json
import torch
import torch.nn.functional as F
import re
from utils.bboxes_tok import BBox, _maybe_to_pil, _pad_box
import numpy as np
from utils.draw_attn_heatmap import draw_attn_heatmap
from dataset.mmmu_pro import format_mmmu_pro_sample
from dataset.mmmu import format_mmmu_sample
from dataset.mmstar import format_mmstar_sample
from dataset.vstar import format_vstar_sample
from dataset.mathvision import format_mathvision_sample
from dataset.mathverse import format_mathverse_sample
import gc



def _decode_new_tokens(processor, gen, input_len: int) -> str:
    """Helper function to decode new tokens from generation output"""
    tok = processor.tokenizer
    seqs = gen.sequences if hasattr(gen, "sequences") else gen
    new_tokens = seqs[:, input_len:]
    texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
    return texts[0] if len(texts) else ""


def compute_stage_uncertainty(logits_sequence: torch.Tensor) -> float:
    """
    Compute stage-wise uncertainty following the algorithm:
    1. For each step t, get logits z_t over the vocab
    2. Compute p_t = log_softmax(z_t)
    3. M = mean_t((top1_p_t - top2_p_t))
    
    Higher uncertainty means the model is less confident (top1 and top2 probs are closer).
    We return the negative of the mean difference to make higher values = more uncertain.
    
    Args:
        logits_sequence: Tensor of shape (num_steps, vocab_size) containing logits for each generation step
        
    Returns:
        float: Mean uncertainty score across all steps (higher = more uncertain)
    """
    if logits_sequence is None or len(logits_sequence) == 0:
        return 0.0
    
    uncertainties = []
    
    for step_logits in logits_sequence:
        # Compute log probabilities
        log_probs = F.log_softmax(step_logits, dim=-1)
        
        # Get top 2 probabilities
        top2_probs, _ = torch.topk(log_probs, k=2, dim=-1)
        top1_prob = top2_probs[0]
        top2_prob = top2_probs[1]
        
        # Calculate confidence gap (larger gap = more confident = lower uncertainty)
        confidence_gap = (top1_prob - top2_prob).item()
        uncertainties.append(confidence_gap)
    
    # Return negative mean to make higher values indicate higher uncertainty
    mean_confidence_gap = np.mean(uncertainties) if uncertainties else 0.0
    return -mean_confidence_gap


def find_mem_spans_from_input_ids(processor, chat_text, input_ids, mem_names, anchor_len=32, logger=None):
    tok = processor.tokenizer

    def find_subseq(hay, needle):
        Lh, Ln = len(hay), len(needle)
        for s in range(Lh - Ln + 1):
            if hay[s:s+Ln] == needle:
                return s
        return -1

    # The Model-input token space: the input_ids fed the model
    # usually includes special/chat/image tokens inserted by the template/processor.

    # 1) tokenize full chat (no special tokens) WITH offsets
    # Plain-chat token space: a clean tokenization of chat_text with no special tokens.
    # Since these spaces differ, we re-tokenize chat_text in the plain space to get offsets,
    # then align it to the model input space with the anchor.
    enc = tok(chat_text, add_special_tokens=False, return_offsets_mapping=True)
    chat_ids = enc["input_ids"]            # list[int], length = N_chat
    offsets  = enc["offset_mapping"]       # list[(start_char, end_char)], length = N_chat

    # 2) align chat_ids into input_ids via an anchor slice (first N chat tokens)
    ids = list(input_ids)
    anchor = chat_ids[:min(anchor_len, len(chat_ids))]
    # logger.info(f"anchor: {len(anchor)}")
    # logger.info(f"ids: {len(ids)}")
    base = find_subseq(ids, anchor)
    # logger.info(f"base: {base}")
    # input_ids[base + i] == chart_ids[i]

    if base < 0 and len(chat_ids) > anchor_len:
        # fallback: try a middle slice if the template injected stuff at the front
        mid = len(chat_ids) // 2
        anchor = chat_ids[mid: mid + min(anchor_len, len(chat_ids)-mid)]
        base = find_subseq(ids, anchor)
        if base >= 0:
            base -= mid  # shift so chat_ids[0] maps to ids[base]
    if base < 0:
        raise ValueError("Could not align chat tokens inside input_ids; template likely wrapped text.")

    spans = {}
    for name in mem_names:
        start_marker = f"<MEM:{name}>"
        end_marker   = f"</MEM:{name}>"

        # 3) char indices for MEM content inside the *rendered chat_text*
        s_char = chat_text.find(start_marker)
        e_char = chat_text.find(end_marker)
        if s_char < 0 or e_char < 0 or e_char <= s_char:
            raise ValueError(f"Markers not found in chat_text for {name}")
        content_lo = s_char + len(start_marker)
        content_hi = e_char  # exclusive

        # 4) map char-range -> token-range in chat tokenization
        #    first token with hi > content_lo, last token with lo < content_hi
        try:
            lo_tok = next(i for i, (lo, hi) in enumerate(offsets) if hi > content_lo)
        except StopIteration:
            raise ValueError(f"Could not map start of MEM {name} to tokens")
        hi_tok = max(i for i, (lo, hi) in enumerate(offsets) if lo < content_hi)

        # 5) shift into model input_ids coordinates
        start_idx = base + lo_tok
        end_idx   = base + hi_tok
        # logger.info(f"MEM {name} span: {start_idx} to {end_idx}")
        # logger.info(f"content: {chat_text[content_lo:content_hi]}")
        if not (0 <= start_idx <= end_idx < len(ids)):
            raise ValueError(f"Computed MEM span out of bounds for {name}")

        spans[name] = (start_idx, end_idx)


    # if "BBOX" in spans:
    #     tokens = tok.convert_ids_to_tokens(input_ids)
    #     image_spans = []
    #     start = None
    #     for i, tok in enumerate(tokens):
    #         if tok == "<|vision_start|>":
    #             start = i + 1  # after vision_start
    #         elif tok == "<|vision_end|>" and start is not None:
    #             image_spans.append((start, i - 1))  # range of <|image_pad|> for one image
    #             start = None
    #     if len(image_spans) > 1:
    #         s_idx, e_idx = image_spans[1]  # cropped image is 2nd
    #         spans["BBOX"] = {
    #             "text_span": spans["BBOX"],
    #             "image_span": (s_idx, e_idx)
    #         }


    return spans



@torch.no_grad()
def attn_mass_per_layer(gen_attentions, spans, prompt_len, processor, input_ids, logger=None):
    """
    gen_attentions: tuple length T; each item is list[L] of tensors (B,H,Q,K)
    spans: {name: (lo, hi)} or {"name": {"text_span": (lo, hi), "image_span": (lo, hi)}}
    """
    T = len(gen_attentions)
    if T == 0:
        return {}
    eps = 1e-8

    # n_layers = len(gen_attentions[0])
    n_layers = 4
    per_layer_mass = {}
    for l in range(n_layers):
        per_layer_mass[l] = {k: 0.0 for k in spans}
        # else:
        #     per_layer_mass[l] = {k: 0.0 for k in spans}

    # Pre-normalize spans into list form to avoid per-token checks
    norm_spans = {}
    for name, span in spans.items():
        if isinstance(span, tuple):
            norm_spans[name] = [span]
        elif isinstance(span, dict):
            norm_spans[name] = [v for k, v in span.items() if isinstance(v, tuple)]
        else:
            raise ValueError(f"Unsupported span format for {name}: {span}")

    # Main computation loop
    for t in range(T):
        layer_list = gen_attentions[t]  # list of length L

        for ilayer, layer_attn in enumerate(layer_list):
            if ilayer >= n_layers:
                continue
            # layer_attn: (B,H,Q,K)
            A = layer_attn.mean(dim=(0, 1, 2))  # → (K,)
            K = A.shape[0]
            den = A[:min(prompt_len, K)].sum() + eps

            # Compute all spans in one GPU pass
            for name, span_list in norm_spans.items():
                total_mass = 0.0
                for (lo, hi) in span_list:
                    if lo < K:
                        hi = min(hi + 1, K)
                        total_mass += A[lo:hi].sum()

                per_layer_mass[ilayer][name] += (total_mass / den).item()

    # Average over time steps
    for l in range(n_layers):
        for name in spans:
            if l < n_layers:
                per_layer_mass[l][name] /= T
          

    return per_layer_mass


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from raw model output text that may contain markdown code blocks.
    
    Args:
        text: Raw text output from the model
        
    Returns:
        Parsed JSON object if found and valid, None otherwise
    """
    if not text or not text.strip():
        return None
    
    # First, try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        cleaned_json = match.strip()
        if cleaned_json:
            try:
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                # Try to fix common issues
                fixed_json = _fix_incomplete_json(cleaned_json)
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    return "Invalid JSON format"
                    continue
    
    # If no markdown blocks found, try to find JSON directly in text
    # Look for patterns that start with { or [
    json_candidates = re.findall(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)
    
    for candidate in json_candidates:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            # Try to fix and parse
            fixed = _fix_incomplete_json(candidate.strip())
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    return None


def _fix_incomplete_json(json_str: str) -> str:
    """Attempt to fix incomplete or malformed JSON strings."""
    json_str = json_str.strip()
    
    # Handle incomplete arrays
    if json_str.startswith('[') and not json_str.endswith(']'):
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        json_str += ']' * (open_brackets - close_brackets)
    
    # Handle incomplete objects
    if json_str.startswith('{') and not json_str.endswith('}'):
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        json_str += '}' * (open_braces - close_braces)
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Handle incomplete string values
    if json_str.count('"') % 2 == 1:
        json_str += '"'
    
    return json_str

class Stage_1:
    def __init__(self, name: str, model: Any, processor: Any, config: dict, base_system_prompt: str):
        self.name = name
        self.sys_prompt = base_system_prompt
        self.model = model
        self.processor = processor
        self.config = config["inference"][name]
        self.dataset_name = config["dataset"]["data_id"].split("/")[-1]
        if self.dataset_name == "MathVerse" and config["dataset"]["vison_only"]:
            self.vision_only = True
        else:
            self.vision_only = False

    def get_prompt(self, blackboard_text: str = None):
        """
        Get the system prompt, optionally with blackboard context.
        blackboard_text should be the formatted text from previous stages.
        """
        if blackboard_text is None or blackboard_text.strip() == "" or self.name == "DIRECT_ANSWER":
            return self.sys_prompt
        return self.sys_prompt + "\n" + blackboard_text
    

    # def get_user_text(self, question: str, W: int, H: int):
    #     wh = f"Image W={W}, Image H={H}\n"
    #     ofmt = "Output: ONE JSON object only; lowercase keys; no extra keys; no prose/markdown.\n"

    #     if self.name == "SCENE.SUMMARY":
    #         return (
    #             f"{wh}{ofmt}"
    #             f"Schema: {{\"caption\":\"<str>\", \"objects\":[\"<str>\", ...]}}\n"
    #             # f"Question: {question}\n"
    #             f"Task: Briefly caption the image (≤20 words) and list key objects."
    #         )
    #     elif self.name == "QUESTION.PARSING":
    #         return (
    #             f"{wh}{ofmt}"
    #             f"Schema: {{\"task\":\"<classify|count|compare|locate|other>\", "
    #             f"\"targets\":[\"<str>\"], \"refs\":[\"<str>\"], "
    #             f"\"attributes\":[\"<color|text|spatial|other>\"], \"text_required\":<true|false>}}\n"
    #             f"Question for parsing: {question}\n"
    #             f"Task: Parse the question into task/targets/refs/attributes/text_required."
    #         )
    #     elif self.name == "BBOX":
    #         return (
    #             f"{wh}{ofmt}"
    #             f"Schema: {{\"bbox\":[x1,y1,x2,y2]}} (ints; 0<=x1<x2<W; 0<=y1<y2<H; Must be positive values.)\n"
    #             f"Select the single tight bbox correlated with the question: \"{question}\".\n"
    #         )
    #     elif self.name == "TEXT.DETECTION":
    #         return (
    #             f"{wh}{ofmt}",
    #             f"Schema: {{\"texts\":[\"<str>\", \"<str>\", ...]}}\n",
    #             # f"Question for detection: {question}\n",
    #             f"Extract only visible text (use ROI if available) relevant to the question: \"{question}\".\n"
    #         )
    #     elif self.name == "COLOR.ATTRIBUTE":
    #         return (
    #             f"{wh}{ofmt}"
    #             f"Schema: {{\"object\":\"<short_noun|unknown>\", "
    #             f"\"color\":\"<red|green|blue|yellow|black|white|brown|gray|null>\"}}\n"
    #             f"Task: Identify the key object and its color mostly relavant to the question: \"{question}\".\n"
    #         )
    #     elif self.name == "SPATIAL.RELATION":
    #         return (
    #             f"{wh}{ofmt}"
    #             f"Schema: {{\"relation\":\"<left_of|right_of|above|below|inside|overlap|nearest|none>\"}}\n"
    #             f"Task: Determine the relation between target(s) and reference object(s) if the question: \"{question}\" requires it.",
    #             f"If the question does not require it, return \"No spatial relation required.\"."
    #         )
    #     elif self.name == "COUNT":
    #         return (
    #             f"{wh}{ofmt}"
    #             f"Schema: {{\"object\":\"<short_noun|unknown>\", \"count\":<int>}}\n"
    #             f"Task: Count the target objects relevant to the question: \"{question}\".\n"
    #         )

    #     elif self.name == "FINAL":
    #         return (
    #             f'{wh}{ofmt}' 
    #             f'Schema: {{"answer":"<str>", "evidence_ids":[1]}}\n'
    #             f'Task: Consult prior-stage outputs in <mem.*> (if present) and return a concise final answer to the question. evidence_ids must be [1].'
    #             f'Question: {question}\n'
    #         )
    #     elif self.name == "DIRECT_ANSWER":
    #         return (
    #             f'{wh}{ofmt}' 
    #             f'Schema: {{"answer":"<str>"}}\n'
    #             f'Task: Directly answer the question: \"{question}\".'
    #         )


    def get_user_text(self, sample: dict, W: int, H: int, dataset_name: str):
        """Build a unified user prompt for any reasoning stage."""
        
        # --- (1) Base question text (formatted for MMMU-Pro) ---
        if dataset_name == "MMMU_Pro":
            full_prompt, labels, gt = format_mmmu_pro_sample(sample)  # e.g., question + A)-J) options
        elif dataset_name == "MMStar":
            full_prompt, labels, gt = format_mmstar_sample(sample)  # e.g., question + A)-J) options
        elif dataset_name == "vstar_bench":
            full_prompt, labels, gt = format_vstar_sample(sample)  # e.g., question + A)-J) options
        elif dataset_name == "MathVision":
            full_prompt, labels, gt = format_mathvision_sample(sample)
        elif dataset_name == "MathVerse":
            full_prompt, labels, gt = format_mathverse_sample(sample)
        elif dataset_name == "MMMU":
            full_prompt, labels, gt = format_mmmu_sample(sample)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        full_prompt = full_prompt.strip()

        # --- (2) Context section: image info and question ---
        image_info = f"Image info: W={W}, H={H}."
        context_block = f"[CONTEXT]\n{full_prompt}\n{image_info}\n"

        # --- (3) Output formatting rules ---
        ofmt = (
            "Output: ONE JSON object only; lowercase keys; no prose/markdown/code fences.\n"
            "Do not include explanations outside JSON.\n"
        )

        # --- (4) Inject into each reasoning stage dynamically ---
        wh = f"{context_block}\n{ofmt}"
        
        full_prompt = (full_prompt, gt)

        # === Stage-Specific Templates ===
        if self.name == "TASK.INTERPRETATION":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"task_type":"<classify|count|compare|locate|reason|other>", '
            #         f'"rephrased_question":"<str>", "expected_output_type":"<text|numeric|boolean|choice>"}}\n'
            #         f'Task: Identify the core intent of the visual question shown in the image. '
            #         f'Determine its reasoning type, restate it briefly, and specify the expected output format.'
            #     ), full_prompt
            # else:
            return (
                f"{wh}"
                f'Schema: {{"task_type":"<classify|count|compare|locate|reason|other>", '
                f'"rephrased_question":"<str>", "expected_output_type":"<text|numeric|boolean|choice>"}}\n'
                f'Task: Interpret the question to identify the main task type, restate it concisely, and specify the expected output type.'
            ), full_prompt

        elif self.name == "VISUAL.OBSERVATION":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"objects":["<str>", ...], "relations":["<str>", ...], '
            #         f'"text_in_image":["<str>", ...], "key_regions":["<str>", ...]}}\n'
            #         f'Task: Describe key visual elements visible in the image related to the question shown in the image.'
            #     ), full_prompt
            # else:
            return (
                f"{wh}"
                f'Schema: {{"objects":["<str>", ...], "relations":["<str>", ...], '
                f'"text_in_image":["<str>", ...], "key_regions":["<str>", ...]}}\n'
                f'Task: List key visual objects, visible relations, readable text, and notable regions relevant to the question.'
            ), full_prompt

        elif self.name == "TEXTUAL.UNDERSTANDING":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"main_concept":"<str>", "keywords":["<str>", ...], '
            #         f'"question_type":"<classify|compare|count|reason|locate>", '
            #         f'"option_entities":["<str>", ...]}}\n'
            #         f'Task: Analyze the text of the question and visible options in the image. '
            #         f'Extract main concepts, essential keywords, and reasoning type.'
            #     ), full_prompt
            # else:
            return (
                f"{wh}"
                f'Schema: {{"main_concept":"<str>", "keywords":["<str>", ...], '
                f'"question_type":"<classify|compare|count|reason|locate>", '
                f'"option_entities":["<str>", ...]}}\n'
                f'Task: Parse the question and options to extract main concepts, keywords, and type of reasoning required.'
            ), full_prompt

        elif self.name == "CONTEXTUAL.LINKING":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"linked_concepts":[["<term>","<visual_obj>"], ...], '
            #         f'"unlinked_terms":["<str>", ...], "link_confidence":<0-1>}}\n'
            #         f'Task: Link textual entities from the question in the image to corresponding visual objects or text in the image.'
            #     ), full_prompt
            # else:
            return (
                f"{wh}"
                f'Schema: {{"linked_concepts":[["<term>","<visual_obj>"], ...], '
                f'"unlinked_terms":["<str>", ...], "link_confidence":<0-1>}}\n'
                f'Task: Link linguistic entities in the question to visible objects or text in the image.'
            ), full_prompt

        elif self.name == "FACT.EXTRACTION":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"facts":["<str>", ...], "measurements":[["<quantity>","<unit>"], ...], "labels":["<str>", ...]}}\n'
            #         f'Task: Extract factual and measurable information from the image or visible text relevant to solving the question shown in the image.'
            #     ), full_prompt
            # else:
            return (
                f"{wh}"
                f'Schema: {{"facts":["<str>", ...], "measurements":[["<quantity>","<unit>"], ...], '
                f'"labels":["<str>", ...]}}\n'
                f'Task: Extract factual and measurable information from the image and text relevant to solving the question.'
            ), full_prompt

        elif self.name == "VARIABLE.DEFINITION":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"variables":[{{"name":"<str>","meaning":"<str>","value":"<num|null>","unit":"<str|null>"}}]}}\n'
            #         f'Task: Define symbolic or numeric variables representing important quantities or entities observed in the image.'
            #     ), full_prompt
            # else:
            return (
                f"{wh}"
                f'Schema: {{"variables":[{{"name":"<str>", "meaning":"<str>", "value":"<num|null>", "unit":"<str|null>"}}]}}\n'
                f'Task: Define clear variables representing important quantities, labels, or symbolic entities derived from visible evidence.'
            ), full_prompt

        elif self.name == "RELATIONAL.REASONING":
            return (
                f"{wh}"
                f'Schema: {{"relations":[{{"subject":"<str>", "relation":"<str>", "object":"<str>"}}], '
                f'"supporting_evidence":["<str>", ...]}}\n'
                f'Task: Infer logical or spatial relations between objects and provide supporting evidence.'
            ), full_prompt  

        elif self.name == "QUANTITATIVE.REASONING":
            return (
                f"{wh}"
                f'Schema: {{"equations":["<str>", ...], "derived_values":[["<var>","<value>","<unit>"], ...], '
                f'"final_numeric":"<num|null>"}}\n'
                f'Task: Perform quantitative reasoning, derive intermediate numeric results, and produce a final value if applicable.'
            ), full_prompt

        elif self.name == "LOGICAL.FILTERING":
            return (
                f"{wh}"
                f'Schema: {{"eliminated_options":["<str>", ...], "remaining_options":["<str>", ...], '
                f'"rationale":"<str>"}}\n'
                f'Task: Eliminate inconsistent or irrelevant options using logical reasoning and summarize why they were removed.'
            ), full_prompt

        elif self.name == "HYPOTHESIS.GENERATION":
            return (
                f"{wh}"
                f'Schema: {{"hypotheses":["<str>", ...], "justifications":["<str>", ...], '
                f'"confidence":[<float>, ...]}}\n'
                f'Task: Generate plausible answer hypotheses with short justifications and confidence scores (0–1).'
            ), full_prompt

        elif self.name == "CROSSMODAL.ALIGNMENT":
            return (
                f"{wh}"
                f'Schema: {{"alignment_score":<float>, "conflicts":["<str>", ...], '
                f'"resolved_interpretation":"<str>"}}\n'
                f'Task: Ensure consistency between visual, textual, and numerical reasoning, resolve conflicts, and report alignment quality.'
            ), full_prompt

        elif self.name == "SELFCONSISTENCY.CHECK":
            return (
                f"{wh}"
                f'Schema: {{"recomputed_values":[["<var>","<value>"]], "agreement_score":<float>, '
                f'"validation_status":"<consistent|inconsistent|uncertain>"}}\n'
                f'Task: Reassess prior reasoning steps for consistency; recompute or verify earlier outputs if needed.'
            ), full_prompt

        elif self.name == "COMPARATIVE.EVALUATION":
            return (
                f"{wh}"
                f'Schema: {{"option_scores":{{"<option>":<float>}}, "ranking":["<option>", ...], '
                f'"selection_reasoning":"<str>"}}\n'
                f'Task: Compare all answer options and rank them according to reasoning confidence and evidence.'
            ), full_prompt

        elif self.name == "ANSWER.CONSOLIDATION":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"final_answer":"<str>", "confidence":<float>, "supporting_stages":["<str>", ...]}}\n'
            #         f'Task: Integrate reasoning outputs from all stages to produce the most probable answer for the question shown in the image. '
            #         f'If the question is a multiple choice question, output the chosen option letter.'
            #         f'If the question is not a multiple choice question, output the final answer.'
            #     ), full_prompt
            # else:
            return (
                f"{wh}"
                f'Schema: {{"final_answer":"<str>", "confidence":<float>, "supporting_stages":["<str>", ...]}}\n'
                f'Task: Integrate reasoning outputs from all relevant stages to produce the most probable final answer.'
                f'If the question is a multiple choice question, output the captical letter <A|B|C|D|...> of the choosen option.'
                f'If the question is not a multiple choice question, output the final answer <str>.'
            ), full_prompt
        

        elif self.name == "EXPLANATION.GENERATION":
            return (
                f"{wh}"
                f'Schema: {{"rationale":"<str>", "evidence":["<str>", ...], '
                f'"reasoning_summary":"<str>"}}\n'
                f'Task: Generate a concise explanation summarizing the reasoning process and evidence behind the final answer.'
            ), full_prompt

        elif self.name == "DIRECT_ANSWER":
            # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name']:
            #     return (
            #         f"{wh}{ofmt}"
            #         f'Schema: {{"answer":"<str>"}}\n'
            #         f'Task: Directly answer the visual question shown in the image using only the image(s) and visible options. '
            #         f'Do not restate the question or options.'
            #         f'If the question is a multiple choice question, output the chosen option letter.'
            #         f'If the question is not a multiple choice question, output the final answer.'
            #     ), full_prompt
            # else:
            
            return (
                f"{wh}"
                f'Schema: {{"answer":"<str>"}}\n'
                f'Task: Directly answer the question using the visible image(s) and options only. '
                f'Do not repeat the question or options.'
                f'If the question is a multiple choice question, output the chosen option letter <A|B|C|D|...>.'
                f'If the question is not a multiple choice question, output the final answer <str>.'
            ), full_prompt
            

    def render_output(self, parsed_json: dict, W: int = None, H: int = None, logger = None):
        """
        Render the raw text output from each stage into a formatted string for the blackboard.
        """

        try:
            if self.name == "ANSWER.CONSOLIDATION":
                answer = parsed_json.get("final_answer", "unknown")
                return answer

            elif self.name == "DIRECT_ANSWER":
                answer = parsed_json.get("answer", "unknown")
                return answer
            
        except Exception as e:
            if logger: 
                logger.warning(f"{self.name} parse failed: {e}")

            return f"{self.name} parsing failed: {parsed_json}"

    def get_iq_embeddings(self, inputs, question_inputs):
        # raw_vision_embeds = self.model.visual.get_vision_embeddings(inputs["pixel_values"])
        # image_emb = self.model.visual.mm_projector(raw_vision_embeds)
        raw_vision_embeds = inputs["pixel_values"]
        grid_sizes = inputs["image_grid_thw"]

        # If your processor already returns flattened embeddings (as you've shown),
        # just feed them to the vision transformer with grid_thw
        vision_out = self.model.model.visual(raw_vision_embeds, grid_thw=grid_sizes)

        aligned_vision_embeds = vision_out[0].unsqueeze(0)

        token_counts = []
        for thw in grid_sizes:
            _, h, w = thw.tolist()
            token_counts.append(h * w)
        token_counts = torch.tensor(token_counts)

        # Adjust for downsampling (PatchMerger)
        ratio = aligned_vision_embeds.shape[1] / token_counts.sum().item()
        merged_token_counts = (token_counts.float() * ratio).round().long().tolist()

        # Split embeddings correctly
        image_emb = torch.split(aligned_vision_embeds, merged_token_counts, dim=1)

        image_emb = [emb.detach().to(torch.float32).cpu().numpy() for emb in image_emb]

        text_out = self.model.model.language_model(
            input_ids=question_inputs["input_ids"],
            attention_mask=question_inputs["attention_mask"],
            output_hidden_states=True,
        )
        mask = question_inputs["attention_mask"].unsqueeze(-1) # [batch, seq_len, 1]
        question_emb = text_out.hidden_states[-1]  * mask   # [batch, seq_len, 3584]

        question_emb = question_emb.detach().to(torch.float32).cpu().numpy()

        return question_emb, image_emb

    def run(self, sample_id, sample: Dict[str, Any], blackboard_text: str, images: list, mem_names: list, logger = None):
        """
        Run the stage with given images and blackboard context.
        
        Args:
            sample: The input sample containing question
            blackboard_text: Formatted text from previous stages  
            images: List of images [original_image] or [original_image, cropped_image]
            logger: Optional logger
        """
        
        # if self.dataset_name == "MMMU_Pro" and "vision" in sample['config_name'] or self.vision_only:
        #     question = ""
        # else:
        #     question = sample["question"] if "question" in sample else sample["text"]
        
        # Use the first image for dimensions (original image)
        
        
        
        primary_img = images[0]
        W, H = primary_img.size
        
        # user_text = self.get_user_text(question, W, H)
        user_text, full_prompt = self.get_user_text(sample, W, H, self.dataset_name)
        prompt = self.get_prompt(blackboard_text)
        # logger.info(f"User text: {prompt}")

        # Prepare messages with multiple images if available
        user_content = []
        
        # Add images to user content
        for i, img in enumerate(images):
            user_content.append({"type": "image"})
            if len(images) > 1 or self.dataset_name == "MathVision":
                user_content.append({"type": "text", "text": f"IMG {i} = image {i+1}\n"})
        
        # Add the user text
        user_content.append({"type": "text", "text": user_text})
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]

        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=images, return_tensors="pt", truncation=True, padding=True).to(self.model.device)

        question_inputs = self.processor.tokenizer(
            user_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        input_ids = inputs["input_ids"][0].tolist()
        if self.name != "DIRECT_ANSWER":
            mem_spans = find_mem_spans_from_input_ids(self.processor, chat, input_ids, mem_names, logger=logger)
            
        with torch.inference_mode():
            self.model.set_attn_implementation("eager")
            gen = self.model.generate(**inputs, 
                                do_sample=self.config["do_sample"],
                                temperature=self.config["temperature"] if self.config["temperature"] > 0.0 else None, 
                                num_beams=self.config["num_beams"],
                                max_new_tokens=self.config["max_new_tokens"],
                                return_dict_in_generate=True,
                                output_scores=True,
                                output_attentions=True,
                            )

            if self.name == "ANSWER.CONSOLIDATION":
                question_emb, image_emb = self.get_iq_embeddings(inputs, question_inputs)


            text = _decode_new_tokens(self.processor, gen, inputs["input_ids"].shape[1])
            if self.name != "DIRECT_ANSWER":
                gen_attention = gen.attentions
                attn_mass = attn_mass_per_layer(gen_attention, mem_spans, inputs["input_ids"].shape[1], self.processor, input_ids, logger=logger)
                # if len(mem_spans.keys()) > 0:
                #     draw_attn_heatmap(attn_mass, f"imgs/attn_mass_per_layer_{sample_id}_{self.name}.png")
                # mean_attention_mass = 0.0
                
            elif self.name == "DIRECT_ANSWER":
                attn_mass = 0.0
            

        if self.name == "ANSWER.CONSOLIDATION":
            parsed_json = extract_json_from_text(text)
            rendered_output = self.render_output(parsed_json, W=W, H=H, logger=logger)

            return {
                "raw_text": text,
                "output": rendered_output,
                "num_images": len(images),
                # "uncertainty_score": uncertainty_score,
                "mean_attention_mass": attn_mass,
                "mem_names": mem_names,
                "mem_span": mem_spans,
                "prompt_len": inputs["input_ids"].shape[1],
                "question_emb": question_emb,
                "image_emb": image_emb,
                "gt": full_prompt[1],
                
            }

        elif self.name == "DIRECT_ANSWER":
            parsed_json = extract_json_from_text(text)
            rendered_output = self.render_output(parsed_json, W=W, H=H, logger=logger)

            return {
                "raw_text": text,
                "output": rendered_output,
                "gt": full_prompt[1],
            }
            
        else:
            
            return {
                "output": text,
                "num_images": len(images),
                "mean_attention_mass": attn_mass,
                "mem_names": mem_names,
                "mem_span": mem_spans,
                "prompt_len": inputs["input_ids"].shape[1],
            }
