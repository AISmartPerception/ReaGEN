sys_prompts = {
"SCENE.SUMMARY": r"""briefly describe the scene and list key objects.""",

"QUESTION.PARSING": r"""parse the question.""",

"BBOX": r"""
select the single bounding box most relevant to answering the question.
schema: {"bbox":[x1,y1,x2,y2]}
constraints:
- integers; 0 <= x1 < x2 < W; 0 <= y1 < y2 < H. Must be positive values.
""",

"TEXT.DETECTION": r"""extract TEXT in the image that is relevant to answering the question.""",

"COLOR.ATTRIBUTE": r"""identify the key object and its color from the roi (or full image).""",

"SPATIAL.RELATION": r"""determine the spatial relation between target(s) and reference(s).""",

"COUNT": r"""count instances of the target object.""",

"FINAL": r"""Produce the final concise answer using the available evidence.
schema: {"answer":"<str>", "evidence_ids":[1]}
rules:
- evidence_ids must be [1].
- answer ≤ 10 words; if not enough evidence, return "unknown".
"""

}


def compose_prompt_simple(stage_key: str) -> str:
    return sys_prompts[stage_key].strip()
