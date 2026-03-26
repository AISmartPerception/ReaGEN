import os
import json
from tqdm import tqdm, trange
from pydantic import BaseModel
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from .prompts.scoring_inst import SCORING_INST, SYSTEM_SCORING_INST, USER_SCORING_INST

class Extract_Answer(BaseModel):
    extracted_answer: str
    score: int

def get_majority_answer(answers):
    answers_extracted = [answer["extracted_answer"] for answer in answers]
    answers_score = [answer["score"] for answer in answers]
    majority_score = Counter(answers_score).most_common(1)[0][0]
    remained_answers = [answers_extracted[i] for i in range(len(answers_extracted)) if answers_score[i] == majority_score]
    majority_answer = Counter(remained_answers).most_common(1)[0][0]
    return majority_answer, majority_score

def ext_yes_or_no(question, ground_truth, model_response, model):
    prompt = USER_SCORING_INST.format(question=question, ground_truth=ground_truth, model_response=model_response)
    messages = [
        {"role": "user", "content": prompt}
    ]
    max_patience = 10
    patience = 0
    while patience < max_patience:
        try:
            responses = model.get_yes_or_no(messages)["response"]
            if isinstance(responses, list):
                yes_or_no = Counter(responses).most_common(1)[0][0]
                response_json = {"extracted_answer": yes_or_no, "score": yes_or_no == "Yes"}
            else:
                response_json = {"extracted_answer": responses, "score": responses == "Yes"}
            break
        except Exception as e:
            print(f"Error: {str(e)}")
        patience += 1
    if patience == max_patience:
        print(f"Failed to get response for prompt: {prompt}")
        response_json = {"extracted_answer": "Extract Error!", "score": False}
        responses = [response_json]
    return response_json, responses

def ext_ans_and_score(question, ground_truth, model_response, model):
    prompt = SCORING_INST.format(question=question, ground_truth=ground_truth, model_response=model_response)
    messages = [
        {"role": "system", "content": SYSTEM_SCORING_INST}, 
        {"role": "user", "content": prompt}
    ]
    max_patience = 10
    patience = 0
    while patience < max_patience:
        try:
            responses = model.get_response(messages, json_format=Extract_Answer)["response"]
            if isinstance(responses, list):
                responses = [json.loads(response) for response in responses]
                extracted_answer, score = get_majority_answer(responses)
                response_json = {"extracted_answer": extracted_answer, "score": score == 1}
            else:
                responses = json.loads(responses)
                response_json = responses
                response_json["score"] = response_json["score"] == 1
            break
        except Exception as e:
            print(f"Error: {str(e)}")
        patience += 1
    if patience == max_patience:
        print(f"Failed to get response for prompt: {prompt}")
        response_json = {"extracted_answer": "Extract Error!", "score": False}
        responses = [response_json]
    return response_json, responses

def ext_ans_and_score_with_llm(cfg, question, ground_truth, model_response, model):
    if cfg.extract_method == "yes_or_no":
        return ext_yes_or_no(question, ground_truth, model_response, model)
    else:
        return ext_ans_and_score(question, ground_truth, model_response, model)
    
def ext_ans_charxiv(cfg, model):
    data_cfg = cfg.data

    # input_file
    input_file = os.path.join(data_cfg.data_dir, f"{data_cfg.mode}_{data_cfg.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    
    resp_file = os.path.join(cfg.output_dir, 
            f'gen-{data_cfg.mode}_{data_cfg.split}.json')

    # output file
    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'scores-{data_cfg.mode}_{data_cfg.split}.json')
    
    data, response = json.load(open(input_file)), json.load(open(resp_file))
    mode = 'descriptive' if 'descriptive' in resp_file.split('-')[-1] else 'reasoning'

    complete_ids = []
    saved_queries = {}
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping generation.")
        with open(output_file) as f:
            queries = json.load(f)
        complete_ids = [k for k in queries if "extracted_answer" in queries[k]]
        print(f"{len(complete_ids)} problems skipped.")
        saved_queries = queries

    if mode == 'descriptive':
        pass
    
    elif mode == 'reasoning':
        from data.CharXiv.src.reasoning_utils import build_reasoning_grading_queries, get_reasoning_result_gpt
        # dict of figure_id -> {figure_id, grading_query}
        queries = build_reasoning_grading_queries(data, response) 
        # merge
        for k in queries:
            if k in saved_queries and k in complete_ids:
                queries[k] = saved_queries[k]

        # 写成多线程
        lock = threading.Lock()

        def process_query(figure_id):
            if figure_id in complete_ids:
                return None
            query = queries[figure_id]
            if hasattr(cfg, "extract_field") and 'decision' in query:
                if cfg.extract_field == "max_mean_terminal":
                    model_response = query['decision']['memory']['max_mean_terminal']['sub_answers'][-1]
                elif cfg.extract_field == "max_terminal":
                    model_response = query['decision']['memory']['max_terminal']['sub_answers'][-1]
                else:
                    model_response = query['response']
            else:
                model_response = query['response']
            response, responses = ext_ans_and_score_with_llm(cfg, query['question'], query['ground_truth'], model_response, model)
            with lock:
                queries[figure_id]['extracted_answer'] = response['extracted_answer']
                queries[figure_id]['llm_score'] = response['score']
                queries[figure_id]['extracted_votes'] = responses
                queries[figure_id].pop('grading_query')
            return figure_id
        
        def save_results():
            try:
                print(f"Saving results to {output_file}...")
                with lock:
                    with open(output_file, "w+") as f:
                        json.dump(queries, f, indent=4)
                print(f"Results saved.")
            except Exception as e:
                print(e)
                print(f"Error in saving {output_file}")   

        with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
            futures = {executor.submit(process_query, figure_id): figure_id for figure_id in queries}
            for i, future in enumerate(tqdm(as_completed(futures), total=len(queries))):
                print(f"Completed {i} queries.")
                if future.result() is not None and i % 10 == 0:
                    save_results()

        save_results()

    else:
        raise ValueError("Mode not supported")

from data.MathVista.evaluation.extract_answer import extract_answer
def ext_ans_mathvista(cfg, model):
    data_cfg = cfg.data

    # input_file
    result_file = os.path.join(cfg.output_dir, f'generation.json')
    print(f"Reading {result_file}...")
    with open(result_file) as f:
        results = json.load(f)
    
    
    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'extraction.json')
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved_results = json.load(open(output_file))
        results.update(saved_results)
    
    # full pids
    full_pids = list(results.keys())
    print("Number of testing problems:", len(full_pids))

    test_pids = []
    for pid in full_pids:
        if 'extracted_answer' not in results[pid]:
            test_pids.append(pid)
    
    test_num = len(test_pids)
    print("Number of problems to run:", test_num)
        
    # 写成多线程
    lock = threading.Lock()

    def process_query(pid):
        if hasattr(cfg, "extract_field") and 'decision' in results[pid]:
            if cfg.extract_field == "max_mean_terminal":
                model_response = results[pid]['decision']['memory']['max_mean_terminal']['sub_answers'][-1]
            elif cfg.extract_field == "max_terminal":
                model_response = results[pid]['decision']['memory']['max_terminal']['sub_answers'][-1]
            else:
                model_response = results[pid]['response']
        else:
            model_response = results[pid]['response']
        response, responses = ext_ans_and_score_with_llm(cfg, results[pid]['query'], results[pid]['answer'], model_response, model)
        with lock:
            results[pid]['extracted_answer'] = response['extracted_answer']
            results[pid]['llm_score'] = response['score']
            results[pid]['extracted_votes'] = responses
        return pid
    
    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(results, f, indent=4)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")

    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = {executor.submit(process_query, pid): pid for pid in test_pids}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
            print(f"Completed {i} queries.")
            if future.result() is not None and i % 10 == 0:
                save_results()

    save_results()

from .mathvision_utils import find_math_answer, is_number, is_equal
def ext_ans_mathvision(cfg, model):
    data_cfg = cfg.data

    # input_file
    result_file = os.path.join(cfg.output_dir, f'generation.json')
    print(f"Reading {result_file}...")
    with open(result_file) as f:
        results = json.load(f)
    
    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'scores.json')
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved_results = json.load(open(output_file))
        results.update(saved_results)

    # full pids
    full_pids = list(results.keys())
    print("Number of testing problems:", len(full_pids))

    test_pids = []
    for pid in full_pids:
        if 'extracted_answer' not in results[pid]:
            test_pids.append(pid)
    
    test_num = len(test_pids)
    print("Number of problems to run:", test_num)
    
    # 多线程
    lock = threading.Lock()

    def process_query(pid):
        raw_exampe = results[pid]
        gt_answer = str(raw_exampe['answer'])
        if len(raw_exampe['options']) > 0:
            gt_answer_value = raw_exampe['options'][ord(gt_answer)-ord('A')]
        else:
            gt_answer_value = ''
        
        if hasattr(cfg, "extract_field") and 'decision' in results[pid]:
            if cfg.extract_field == "max_mean_terminal":
                model_response = results[pid]['decision']['memory']['max_mean_terminal']['sub_answers'][-1]
            elif cfg.extract_field == "max_terminal":
                model_response = results[pid]['decision']['memory']['max_terminal']['sub_answers'][-1]
            else:
                model_response = results[pid]['response']
        else:
            model_response = results[pid]['response']

        response, responses = ext_ans_and_score_with_llm(cfg, results[pid]['question'], results[pid]['answer'], model_response, model)
        with lock:
            results[pid]['extracted_answer'] = response['extracted_answer']
            results[pid]['llm_score'] = response['score']
            extraction = response['extracted_answer']
            results[pid]["rule_score"] = is_equal(gt_answer, extraction) or is_equal(gt_answer_value, extraction)
            results[pid]['extracted_votes'] = responses
        return pid
    
    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(results, f, indent=4)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")
    
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = {executor.submit(process_query, pid): pid for pid in test_pids}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
            print(f"Completed {i} queries.")
            if future.result() is not None and i % 10 == 0:
                save_results()
    save_results()

def ext_ans_vstar(cfg, model):
    data_cfg = cfg.data

    result_file = os.path.join(cfg.output_dir, "generation.json")
    print(f"Reading {result_file}...")
    with open(result_file) as f:
        results = json.load(f)

    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scores.json")
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved_results = json.load(open(output_file))
        results.update(saved_results)

    # which pids still need extraction?
    full_pids = list(results.keys())
    test_pids = [pid for pid in full_pids if "extracted_answer" not in results[pid]]
    print("Number of testing problems:", len(test_pids))

    lock = threading.Lock()

    def process_query(pid):
        # gold answer (letter)
        gt_answer = results[pid].get("answer", "")
        gt_answer_value = ""  # VStar is letter-based

        # pick model_response: either from mcts memory or plain response
        if hasattr(cfg, "extract_field") and "decision" in results[pid]:
            if cfg.extract_field == "max_mean_terminal":
                model_response = results[pid]["decision"]["memory"]["max_mean_terminal"]["sub_answers"][-1]
            elif cfg.extract_field == "max_terminal":
                model_response = results[pid]["decision"]["memory"]["max_terminal"]["sub_answers"][-1]
            else:
                model_response = results[pid]["response"]
        else:
            model_response = results[pid]["response"]

        # use existing helper that LLM-scores the answer
        response, responses = ext_ans_and_score_with_llm(
            cfg,
            results[pid]["question"],
            results[pid]["answer"],
            model_response,
            model,
        )

        with lock:
            results[pid]["extracted_answer"] = response["extracted_answer"]
            results[pid]["llm_score"] = response["score"]
            extraction = response["extracted_answer"]
            # rule-level check: exact letter match
            results[pid]["rule_score"] = is_equal(gt_answer, extraction) or is_equal(gt_answer_value, extraction)
            results[pid]["extracted_votes"] = responses
        return pid

    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(results, f, indent=4)
            print("Results saved.")
        except Exception as e:
            print(e)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in test_pids]
        from tqdm import tqdm
        for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
            if future.result() is not None and i % 10 == 0:
                save_results()
    save_results()

def ext_ans_mmstar(cfg, model):
    data_cfg = cfg.data

    result_file = os.path.join(cfg.output_dir, "generation.json")
    print(f"Reading {result_file}...")
    with open(result_file, "r") as f:
        results = json.load(f)

    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scores.json")
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved = json.load(open(output_file, "r"))
        results.update(saved)

    # figure out which ones still need extraction
    test_pids = [pid for pid in results if "extracted_answer" not in results[pid]]
    print("Number of testing problems:", len(test_pids))

    lock = threading.Lock()

    def process_query(pid):
        # GT is the letter (A/B/C/...)
        gt_answer = results[pid].get("answer", "")
        gt_answer_value = ""  # no secondary numeric target

        # pick model output
        if hasattr(cfg, "extract_field") and "decision" in results[pid]:
            if cfg.extract_field == "max_mean_terminal":
                model_response = results[pid]["decision"]["memory"]["max_mean_terminal"]["sub_answers"][-1]
            elif cfg.extract_field == "max_terminal":
                model_response = results[pid]["decision"]["memory"]["max_terminal"]["sub_answers"][-1]
            else:
                model_response = results[pid]["response"]
        else:
            model_response = results[pid]["response"]

        # ask LLM to extract/score
        response, responses = ext_ans_and_score_with_llm(
            cfg,
            results[pid]["question"],
            results[pid]["answer"],
            model_response,
            model,
        )

        with lock:
            results[pid]["extracted_answer"] = response["extracted_answer"]
            results[pid]["llm_score"] = response["score"]
            extraction = response["extracted_answer"]
            # rule-level = exact match on the letter
            results[pid]["rule_score"] = is_equal(gt_answer, extraction) or is_equal(gt_answer_value, extraction)
            results[pid]["extracted_votes"] = responses
        return pid

    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
            print("Results saved.")
        except Exception as e:
            print(e)

    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in test_pids]
        for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
            if future.result() is not None and i % 10 == 0:
                save_results()
    save_results()

def ext_ans_mmmu_pro(cfg, model):
    result_file = os.path.join(cfg.output_dir, "generation.json")
    print(f"Reading {result_file}...")
    with open(result_file, "r") as f:
        results = json.load(f)

    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scores.json")
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved = json.load(open(output_file, "r"))
        results.update(saved)

    # figure out which pids to process
    test_pids = [pid for pid in results if "extracted_answer" not in results[pid]]
    print("Number of testing problems:", len(test_pids))

    lock = threading.Lock()

    def process_query(pid):
        gt_answer = results[pid].get("answer", "")
        gt_answer_value = ""

        # pick model response
        if hasattr(cfg, "extract_field") and "decision" in results[pid]:
            mem = results[pid]["decision"]["memory"]
            if cfg.extract_field == "max_mean_terminal":
                model_response = mem["max_mean_terminal"]["sub_answers"][-1]
            elif cfg.extract_field == "max_terminal":
                model_response = mem["max_terminal"]["sub_answers"][-1]
            else:
                model_response = results[pid]["response"]
        else:
            model_response = results[pid]["response"]

        # call the shared helper
        response, responses = ext_ans_and_score_with_llm(
            cfg,
            results[pid]["question"],
            results[pid]["answer"],
            model_response,
            model,
        )

        with lock:
            results[pid]["extracted_answer"] = response["extracted_answer"]
            results[pid]["llm_score"] = response["score"]
            extraction = response["extracted_answer"]
            # exact letter match
            results[pid]["rule_score"] = is_equal(gt_answer, extraction) or is_equal(gt_answer_value, extraction)
            results[pid]["extracted_votes"] = responses
        return pid

    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
            print("Results saved.")
        except Exception as e:
            print(e)

    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in test_pids]
        for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
            if future.result() is not None and i % 10 == 0:
                save_results()
    save_results()

import json
from tqdm import tqdm

def ext_ans_mathverse(cfg, model):
    """
    Extract answers from MathVerse Vision-only dataset.
    """
    data_cfg = cfg.data
    result_file = os.path.join(cfg.output_dir, "generation.json")
    print(f"Reading {result_file}...")
    with open(result_file) as f:
        results = json.load(f)

    output_dir = cfg.extract_result_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scores.json")

    # If extraction already done, skip
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping extraction.")
        saved_results = json.load(open(output_file))
        results.update(saved_results)

    # Process pids
    full_pids = list(results.keys())
    test_pids = [pid for pid in full_pids if "extracted_answer" not in results[pid]]
    print("Number of testing problems:", len(test_pids))

    lock = threading.Lock()

    def process_query(pid):
        # gold answer (from "answer")
        gt_answer = results[pid].get("answer", "")
        model_response = results[pid]["response"]

        # Scoring using LLM
        response, responses = ext_ans_and_score_with_llm(
            cfg,
            results[pid]["question"],
            gt_answer,
            model_response,
            model,
        )

        with lock:
            results[pid]["extracted_answer"] = response["extracted_answer"]
            results[pid]["llm_score"] = response["score"]
            results[pid]["rule_score"] = gt_answer == response["extracted_answer"]
            results[pid]["extracted_votes"] = responses
        return pid

    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(results, f, indent=4)
            print("Results saved.")
        except Exception as e:
            print(e)

    # Process queries using multithreading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in test_pids]
        from tqdm import tqdm
        for i, future in enumerate(tqdm(as_completed(futures), total=len(test_pids))):
            if future.result() is not None and i % 10 == 0:
                save_results()
    save_results()


def extract(cfg, model):
    if cfg.data.name == "charxiv":
        ext_ans_charxiv(cfg, model)
    elif cfg.data.name == "mathvista":
        ext_ans_mathvista(cfg, model)
    elif cfg.data.name == "mathvision":
        ext_ans_mathvision(cfg, model)
    elif cfg.data.name == "vstar":
        ext_ans_vstar(cfg, model)
    elif cfg.data.name == "mmstar":
        ext_ans_mmstar(cfg, model)
    elif cfg.data.name == "mmmu_pro_standard_10" or cfg.data.name == "mmmu_pro_standard_4":
        ext_ans_mmmu_pro(cfg, model)
    elif cfg.data.name == "mathverse":
        ext_ans_mathverse(cfg, model)
    else:
        raise ValueError("Dataset not supported")