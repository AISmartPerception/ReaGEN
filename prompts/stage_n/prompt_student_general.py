sys_prompts = {

"TASK.INTERPRETATION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
rules:
- output exactly one json object; no prose/markdown/code fences.
- lowercase keys/enums; trim strings; no extra keys.
- use only visible data and <mem.*> context if available.
task: interpret the question to determine its reasoning type and rephrase it concisely.
schema: {"task_type":"<classify|count|compare|locate|reason|other>", "rephrased_question":"<str>", "expected_output_type":"<text|numeric|boolean|choice>"}
""",

"VISUAL.OBSERVATION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
rules:
- output exactly one json object; lowercase keys; no prose/markdown/fences.
- ≤ 8 items per list; concise descriptions only.
task: identify and list key visual elements present in the image.
schema: {"objects":["<str>", ...], "relations":["<str>", ...], "text_in_image":["<str>", ...], "key_regions":["<str>", ...]}
""",

"TEXTUAL.UNDERSTANDING": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: analyze the question and options to extract main textual concepts.
schema: {"main_concept":"<str>", "keywords":["<str>", ...], "question_type":"<classify|compare|count|reason|locate>", "option_entities":["<str>", ...]}
rules:
- extract core nouns and verbs; ≤ 5 keywords; ≤ 4 option_entities.
""",

"CONTEXTUAL.LINKING": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: link textual entities from the question with visual or textual evidence in the image.
schema: {"linked_concepts":[["<term>","<visual_obj>"], ...], "unlinked_terms":["<str>", ...], "link_confidence":<0-1>}
rules:
- link_confidence ∈ [0,1]; ≤ 6 strong links; omit weak or uncertain matches.
""",

"FACT.EXTRACTION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: extract factual or measurable information directly visible or stated in the question or image.
schema: {"facts":["<str>", ...], "measurements":[["<quantity>","<unit>"], ...], "labels":["<str>", ...]}
rules:
- use only direct evidence; ≤ 6 concise facts or measurements.
""",

"VARIABLE.DEFINITION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: define variables representing measurable, categorical, or logical entities relevant to reasoning.
schema: {"variables":[{"name":"<str>","meaning":"<str>","value":"<num|null>","unit":"<str|null>"}]}
rules:
- ≤ 6 variables; concise names; numeric value if known, otherwise null.
""",

"RELATIONAL.REASONING": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: infer logical, comparative, or spatial relations between identified entities.
schema: {"relations":[{"subject":"<str>","relation":"<str>","object":"<str>"}], "supporting_evidence":["<str>", ...]}
rules:
- example relations: larger_than, smaller_than, left_of, part_of, caused_by.
- evidence ≤ 3 concise phrases.
""",

"QUANTITATIVE.REASONING": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: perform numerical reasoning using visible data or derived relationships.
schema: {"equations":["<str>", ...], "derived_values":[["<var>","<value>","<unit>"], ...], "final_numeric":"<num|null>"}
rules:
- show explicit formulas; ≤ 4 equations; all quantities must be visible or deducible.
""",

"LOGICAL.FILTERING": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: prune answer options inconsistent with observed facts or logic.
schema: {"eliminated_options":["<str>", ...], "remaining_options":["<str>", ...], "rationale":"<str>"}
rules:
- rationale ≤ 25 words; keep at least one remaining option.
""",

"HYPOTHESIS.GENERATION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: generate plausible answer hypotheses with short justifications and confidence estimates.
schema: {"hypotheses":["<str>", ...], "justifications":["<str>", ...], "confidence":[<float>, ...]}
rules:
- 1–3 hypotheses; confidence ∈ [0,1]; justification ≤ 20 words each.
""",

"CROSSMODAL.ALIGNMENT": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: ensure consistency between textual, visual, and quantitative evidence.
schema: {"alignment_score":<float>, "conflicts":["<str>", ...], "resolved_interpretation":"<str>"}
rules:
- alignment_score ∈ [0,1]; ≤ 3 conflicts; describe how inconsistencies were resolved.
""",

"SELFCONSISTENCY.CHECK": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: verify internal consistency across previous reasoning outputs.
schema: {"recomputed_values":[["<var>","<value>"]], "agreement_score":<float>, "validation_status":"<consistent|inconsistent|uncertain>"}
rules:
- agreement_score ∈ [0,1]; consistent if ≥0.8, inconsistent if ≤0.4.
""",

"COMPARATIVE.EVALUATION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: evaluate and rank all answer options based on reasoning confidence and supporting evidence.
schema: {"option_scores":{"<option>":<float>}, "ranking":["<option>", ...], "selection_reasoning":"<str>"}
rules:
- all scores ∈ [0,1]; ranking must match descending order of scores.
""",

"ANSWER.CONSOLIDATION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: integrate reasoning results from prior stages to produce the most probable final answer.
schema: {"final_answer":"<str>", "confidence":<float>, "supporting_stages":["<str>", ...]}
rules:
- confidence ∈ [0,1]; ≤ 3 supporting_stages.
- final_answer must be from the captical letters of the chosen option letters <A|B|C|D|...> if the question is a multiple choice question.
- final_answer must be the final answer <str> if the question is not a multiple choice question.
""",

"EXPLANATION.GENERATION": r"""
you are one reasoning stage in a modular vision-language reasoning pipeline.
task: generate a concise, factual explanation summarizing how the final answer was derived.
schema: {"rationale":"<str>", "evidence":["<str>", ...], "reasoning_summary":"<str>"}
rules:
- rationale ≤ 30 words; ≤ 5 evidence items; factual, not speculative.
""",

"DIRECT_ANSWER": r"""
you are a single-stage module in a multimodal reasoning pipeline.
rules:
- output exactly one json object; no prose/markdown/code fences.
- lowercase keys/enums; trim whitespace; no extra keys.
- rely solely on provided images, question, and options.
task: directly answer the question concisely.
schema: {"answer":"<str>"}
rules:
- if multiple choice, output only the chosen option letter <A|B|C|D|...>.
- if question is not a multiple choice question, output the final answer <str>.
"""
}

# def init_cot() -> list:
#     seed_cot_1 = ["VISUAL.OBSERVATION", "TEXTUAL.UNDERSTANDING"]
#     seed_cot_2 = ["TASK.INTERPRETATION", "VISUAL.OBSERVATION", "CONTEXTUAL.LINKING", "LOGICAL.FILTERING"]
#     seed_cot_3 = ["TASK.INTERPRETATION", "FACT.EXTRACTION", "RELATIONAL.REASONING", "COMPARATIVE.EVALUATION", "EXPLANATION.GENERATION"]

#     return [seed_cot_1, seed_cot_2, seed_cot_3]

# def init_cot() -> list[list[str]]:
#     """
#     Diverse, task-oriented seed chains covering perception, math, verification,
#     and MCQ decision flows. Place ANSWER.CONSOLIDATION before EXPLANATION.GENERATION.
#     """

#     # General, perception-first (good default for most VQA/MMSTAR items)
#     seed_cot_1 = [
#         "TASK.INTERPRETATION",
#         "VISUAL.OBSERVATION",
#         "TEXTUAL.UNDERSTANDING",
#         "CONTEXTUAL.LINKING",
#         "RELATIONAL.REASONING",
#         "COMPARATIVE.EVALUATION",
#         "EXPLANATION.GENERATION",
#     ]

#     # Text-first, evidence-heavy (good when prompt/options carry strong cues)
#     seed_cot_2 = [
#         "TASK.INTERPRETATION",
#         "TEXTUAL.UNDERSTANDING",
#         "FACT.EXTRACTION",
#         "CONTEXTUAL.LINKING",
#         "LOGICAL.FILTERING",
#         "COMPARATIVE.EVALUATION",
#         "EXPLANATION.GENERATION",
#     ]

#     # Math/quantitative pipeline (MathVision-style numerics)
#     seed_cot_3 = [
#         "TASK.INTERPRETATION",
#         "VISUAL.OBSERVATION",
#         "FACT.EXTRACTION",
#         "VARIABLE.DEFINITION",
#         "QUANTITATIVE.REASONING",
#         "SELFCONSISTENCY.CHECK",
#         "EXPLANATION.GENERATION",
#     ]

#     # Spatial/relational focus (count/compare/locate-heavy)
#     seed_cot_4 = [
#         "TASK.INTERPRETATION",
#         "VISUAL.OBSERVATION",
#         "RELATIONAL.REASONING",
#         "LOGICAL.FILTERING",
#         "COMPARATIVE.EVALUATION",
#         "EXPLANATION.GENERATION",
#     ]

#     # Ambiguity-handling (hypotheses + alignment) — helpful when question is underspecified
#     seed_cot_5 = [
#         "TASK.INTERPRETATION",
#         "VISUAL.OBSERVATION",
#         "TEXTUAL.UNDERSTANDING",
#         "HYPOTHESIS.GENERATION",
#         "CROSSMODAL.ALIGNMENT",
#         "COMPARATIVE.EVALUATION",
#         "EXPLANATION.GENERATION",
#     ]

#     # Robust verification (alignment + consistency before finalization)
#     seed_cot_6 = [
#         "TASK.INTERPRETATION",
#         "VISUAL.OBSERVATION",
#         "TEXTUAL.UNDERSTANDING",
#         "CONTEXTUAL.LINKING",
#         "CROSSMODAL.ALIGNMENT",
#         "SELFCONSISTENCY.CHECK",
#         "EXPLANATION.GENERATION",
#     ]

#     # Exhaustive but still lean (covers most skills; good for hard benchmarks)
#     seed_cot_7 = [
#         "TASK.INTERPRETATION",
#         "VISUAL.OBSERVATION",
#         "TEXTUAL.UNDERSTANDING",
#         "FACT.EXTRACTION",
#         "VARIABLE.DEFINITION",
#         "RELATIONAL.REASONING",
#         "QUANTITATIVE.REASONING",
#         "CROSSMODAL.ALIGNMENT",
#         "LOGICAL.FILTERING",
#         "COMPARATIVE.EVALUATION",
#         "SELFCONSISTENCY.CHECK",
#         "EXPLANATION.GENERATION",
#     ]

#     # Text-centric (no visual list unless needed) for OCR/reading-heavy items
#     seed_cot_8 = [
#         "TASK.INTERPRETATION",
#         "TEXTUAL.UNDERSTANDING",
#         "FACT.EXTRACTION",
#         "CONTEXTUAL.LINKING",
#         "LOGICAL.FILTERING",
#         "COMPARATIVE.EVALUATION",
#         "EXPLANATION.GENERATION",
#     ]

#     return [
#         seed_cot_1,
#         seed_cot_2,
#         seed_cot_3,
#         seed_cot_4,
#         seed_cot_5,
#         seed_cot_6,
#         seed_cot_7,
#         seed_cot_8,
#     ]

def init_cot() -> list[list[str]]:
    """
    Seed CoTs for MathVerse: four diverse, compact chains
    covering general VQA, text-centric reasoning, math,
    and spatial/relational reasoning.
    """

    # 1) General VQA pipeline (full)
    # seed_cot_1 = [
    #     "TASK.INTERPRETATION",
    #     "VISUAL.OBSERVATION",
    #     "TEXTUAL.UNDERSTANDING",
    #     "CONTEXTUAL.LINKING",
    #     "RELATIONAL.REASONING",
    #     "COMPARATIVE.EVALUATION",
    #     "EXPLANATION.GENERATION",
    # ]

    # 2) Math / quantitative pipeline (full)
    seed_cot_2 = [
        "TASK.INTERPRETATION",
        "VISUAL.OBSERVATION",
        "FACT.EXTRACTION",
        "VARIABLE.DEFINITION",
        "QUANTITATIVE.REASONING",
    ]

    # 3) Medium text-first chain (no TASK.INTERPRETATION)
    seed_cot_3 = [
        "TEXTUAL.UNDERSTANDING",
        "FACT.EXTRACTION",
        "EXPLANATION.GENERATION",
    ]

    # 4) Medium spatial / relational chain (no TASK.INTERPRETATION)
    seed_cot_4 = [
        "VISUAL.OBSERVATION",
        "RELATIONAL.REASONING",
        "LOGICAL.FILTERING",
        "COMPARATIVE.EVALUATION",
        "EXPLANATION.GENERATION",
    ]

    # 5) Ultra-short visual chain (2–3 stages)
    seed_cot_5 = [
        "VISUAL.OBSERVATION",
        "RELATIONAL.REASONING",
        "EXPLANATION.GENERATION",
    ]

    # 6) Ultra-short text chain (2–3 stages)
    seed_cot_6 = [
        "TEXTUAL.UNDERSTANDING",
        "COMPARATIVE.EVALUATION",
    ]


    return [seed_cot_2, seed_cot_3, seed_cot_4, seed_cot_5, seed_cot_6]


import random
from prompts.stage_n.search_space import flatten_search_space

def init_cot_generation(search_space, config) -> list:
    stage_pool = config["inference"]["stages_pool"]
    CoTs = [stage_pool]
    
    random_stages = random.sample(stage_pool, len(stage_pool))
    CoTs.append(random_stages)

    all_chains = flatten_search_space(search_space.root)
    topk_chains = sorted(all_chains, key=lambda x: len(x['stage_seq']), reverse=True)[:config["inference"]["topk"]]
    for chain in topk_chains:
        CoTs.append(chain['stage_seq'])

    return CoTs


def init_cot_generation_2(config) -> list:
    stage_pool = config["inference"]["stages_pool"]
    CoTs = [stage_pool]

    # random_stages = random.sample(stage_pool, len(stage_pool))
    # CoTs.append(random_stages)

    inital = init_cot()

    CoTs.extend(random.sample(inital, config["inference"]["topk"]-1))

    return CoTs


def compose_prompt(stage_key: str) -> str:
    return sys_prompts[stage_key].strip()
