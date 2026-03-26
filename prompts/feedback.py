from prompts.stage_n.chain import Chain

import math
import numpy as np

def _to_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return list(x)
    except Exception:
        return [x]

def _as_text(x):
    if isinstance(x, dict):
        # common keys we might surface; fall back to repr
        for k in ("text", "output", "answer", "content", "message"):
            if k in x:
                return str(x[k])
        return str(x)
    return str(x)

def feedback_fn(chain, sample, score, out, answer, config, calc_iou=None, base_system_prompts=None):

    question = (sample or {}).get("question", "")
    answer_raw = (out or {}).get("answer_raw", "")
    stage_outputs = (out or {}).get("stage_outputs") or []
    blackboard_text = (out or {}).get("blackboard_text", "")
    # importance = (out or {}).get("importance")
    # a_to_final = (out or {}).get("a_to_final")
    importance_dict = (out or {}).get("importance_dict")
    
    # imp_list = _to_list(importance)
    # att_list = _to_list(a_to_final)

    # n_stages = len(stage_outputs)

    header_lines = [
        "Feedback from multi-stage reasoning on the question:",
        f"Question: {question}",
        f"Ground-truth answer: {answer}",
        f"Predicted answer: {answer_raw}",
        (f"Prediction Correctness: {score}" if score is not None else ""),
        # (f"IoU: {calc_iou}" if calc_iou is not None else "")
    ]
    header_lines = [ln for ln in header_lines if ln != ""]

    body_lines = [] 
    for i, stage_name in enumerate(stage_outputs.keys()):
        stage_text = _as_text(stage_outputs[stage_name])

        # imp_val = imp_list[i] if i < len(imp_list) else None
        # att_val = att_list[i] if i < len(att_list) else None
        if stage_name != "ANSWER.CONSOLIDATION" and stage_name != "DIRECT_ANSWER":
            body_lines.append(f"Stage {i+1}, {stage_name}:")
            body_lines.append(f"    Output of {stage_name}: {stage_text}")
        elif stage_name == "ANSWER.CONSOLIDATION":
            body_lines.append(f"Final answer: {stage_text}")
        elif stage_name == "DIRECT_ANSWER":
            body_lines.append(f"Direct answer: {stage_text}")
        
        
        if stage_name != "ANSWER.CONSOLIDATION" and stage_name != "DIRECT_ANSWER" and config["inference"]["attn"]:
            imp_info = importance_dict.get(stage_name, None)
            if imp_info:
                body_lines.append(
                    f"    Direct influence: {imp_info['direct']:.4f}, "
                    f"Indirect influence: {imp_info['indirect']:.4f}, "
                    f"Total influence: {imp_info['total']:.4f}"
                )
    
    body_lines = [line for line in body_lines if line != ""]

    output = "\n".join(header_lines + [""] + body_lines + ["\n"])

    return output