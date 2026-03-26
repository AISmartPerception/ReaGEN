import os
import json
import asyncio
import pickle
from tqdm import tqdm, trange
import aiofiles
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import ast

def gen_charxiv(cfg, model):
    data_cfg = cfg.data

    # input_file
    input_file = os.path.join(data_cfg.data_dir, f"{data_cfg.mode}_{data_cfg.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    # output file
    output_file = os.path.join(cfg.output_dir, 
            f'gen-{data_cfg.mode}_{data_cfg.split}.json')
    
    if "mcts" in cfg.prompt_method.name:
        tree_save_dir = os.path.join(cfg.output_dir, f'tree_results')
        os.makedirs(tree_save_dir, exist_ok=True)

    complete_ids = []
    saved_queries = {}
    if os.path.exists(output_file):
        print(f"{output_file} exists. Skipping generation.")
        with open(output_file) as f:
            queries = json.load(f)
        complete_ids = [k for k in queries if "response" in queries[k] and queries[k]['response'] != "Failed to generate response!"]
        print(f"{len(complete_ids)} problems skipped.")
        saved_queries = queries
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if data_cfg.mode == 'descriptive':
        from data.CharXiv.src.descriptive_utils import build_descriptive_quries
        queries = build_descriptive_quries(data, data_cfg.image_dir)
    elif data_cfg.mode == 'reasoning':
        from data.CharXiv.src.reasoning_utils import build_reasoning_queries
        queries = build_reasoning_queries(data, data_cfg.image_dir)
    else: 
        raise ValueError("Mode not supported")
    
    # merge queries
    for k in queries:
        if k in saved_queries and k in complete_ids:
            queries[k] = saved_queries[k]
    
    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", data_cfg.mode)
    print("Output file:", output_file)
    
    # 创建一个线程安全的字典和锁
    lock = threading.Lock()

    def process_query(k):
        if k in complete_ids:
            return None
        
        query = queries[k]['raw_question']
        image = queries[k]["figure_path"]
        try:
            response = model.get_response(query, image)
        except Exception as e:
            print(e)
            response = {"response": "Failed to generate response!"}

        if "root" in response:
            tree_save_path = os.path.join(tree_save_dir, f"{k}.pkl")
            with open(tree_save_path, "wb") as f:
                pickle.dump(response["root"], f)
            del response["root"]

        with lock:
            queries[k]['decision'] = response
            queries[k]['response'] = response["response"]

            queries[k].pop("figure_path", None)
            queries[k].pop("question", None)

        return k

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

    # 使用线程池来并发处理查询
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = {executor.submit(process_query, k): k for k in queries}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(queries))):
            if future.result() is not None:
                print(f"Completed {i} queries.")
                if i % cfg.save_steps == 0:
                    save_results()
    
    save_results()

#### ------- To handle errors -------
def load_caption_data():
    # Return an empty dictionary, no captions will be provided
    return {}

def load_ocr_data():
    # Return an empty dictionary, no OCR data will be provided
    return {}

import argparse

def parse_args(cfg):
    # Create a Namespace object
    args = argparse.Namespace()
    
    # Set the attributes as in your dictionary
    args.shot_num = 0  # Example from the config
    args.shot_type = ""  # Default, can be adjusted as needed
    args.use_caption = False  # Default, don't use captions
    args.use_ocr = False  # Default, don't use OCR data
    
    return args


# -----------------------------------


from data.MathVista.evaluation.build_query import create_query_data
def gen_mathvista(cfg, model):
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    print("Data:", data['1'])

    # #### TODO: Added these to handle errors ----------
    # # Add minimal functions to handle missing data
    # caption_data = load_caption_data()  # Return empty data
    # ocr_data = load_ocr_data()          # Return empty data
    # args = parse_args(cfg)
    # # -----------------
    query_data = create_query_data(data)
    # query_data = create_query_data(data, caption_data, ocr_data, args)

    # output file
    output_file = os.path.join(cfg.output_dir, f'generation.json')
    
    if "mcts" in cfg.prompt_method.name:
        tree_save_dir = os.path.join(cfg.output_dir, f'tree_results')
        os.makedirs(tree_save_dir, exist_ok=True)

    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        with open(output_file) as f:
            results = json.load(f)
    else:
        results = {}
    
    data.update(results)
    results = data
    
    complited_ids = []
    
    # build final test pid list
    test_pids = list(results.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    print("\nRemoving problems with existing valid response...")
    for pid in test_pids:
        if pid in results and 'response' in results[pid] and results[pid]['response'] != "Failed to generate response!":
            complited_ids.append(pid)
    
    # 写成多线程
    lock = threading.Lock()

    def process_query(pid):
        if pid in complited_ids:
            return None
        problem = data[pid]
        query = query_data[pid]
        image = problem['image']
        image_path = os.path.join(data_cfg.data_dir, image)

        try:
            response = model.get_response(query, image_path)
            
            if "root" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{pid}.pkl")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["root"], f)
                del response["root"]

            with lock:
                results[pid]['query'] = query

                results[pid]['decision'] = response
                results[pid]['response'] = response["response"]

        except Exception as e:
            print(str(e))
            with lock:
                results[pid]['query'] = query
                results[pid]['error'] = str(e)
                results[pid]['response'] = "Failed to generate response!"

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
        futures = {executor.submit(process_query, pid): pid for pid in results}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(results))):
            print(f"Completed {i} queries.")
            if future.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()


def load_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data
def gen_mathvision(cfg, model):
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")
    data = load_jsonl(input_file)
    questions = {}
    for line in data:
        questions[line['id']] = line
    output_file = os.path.join(cfg.output_dir, f'generation.json')
    
    if "mcts" in cfg.prompt_method.name:
        tree_save_dir = os.path.join(cfg.output_dir, f'tree_results')
        os.makedirs(tree_save_dir, exist_ok=True)

    old_id = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
            for k in questions:
                if k in results and 'response' in results[k] and results[k]['response'] != "Failed to generate response!":
                    questions[k] = results[k]
                    old_id.append(k)
    completed_ids = old_id

    # benchmark_prompt = "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n"

    # 写成多线程
    lock = threading.Lock()

    def process_query(idx):
        if idx in old_id:
            return None
        
        line = questions[idx]
        question = line['question']
        options = ''
        if len(line['options']) > 0:
            assert len(line['options']) == 5, f"len(line['options']) == {len(line['options'])} != 5"
            options = f"(A) {line['options'][0]}\n(B) {line['options'][1]}\n(C) {line['options'][2]}\n(D) {line['options'][3]}\n(E) {line['options'][4]}\n"
        question = f"{question}\n{options}"
        image_path = os.path.join(data_cfg.image_dir, line['image'])

        try:
            response = model.get_response(question, image_path)

            if "root" in response:
                tree_save_path = os.path.join(tree_save_dir, f"{idx}.pkl")
                with open(tree_save_path, "wb") as f:
                    pickle.dump(response["root"], f)
                del response["root"]

            with lock:
                questions[idx]["question"] = question
                questions[idx]["decision"] = response
                questions[idx]["response"] = response["response"]

        except Exception as e:
            print(str(e))
            with lock:
                questions[idx]["error"] = str(e)
                questions[idx]["response"] = "Failed to generate response!"

        return idx
    
    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w+") as f:
                    json.dump(questions, f, indent=4)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")

    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = {executor.submit(process_query, idx): idx for idx in questions}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(questions))):
            print(f"Completed {i} queries.")
            if future.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()

def gen_vstar(cfg, model):
    """
    Generate answers for the VStar benchmark.
    VStar jsonl fields (from your file):
        - question_id: string/int id
        - text: full question + (A)... + instruction
        - image: relative path
        - label: gold letter, e.g. "A"
    """
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")
    data = load_jsonl(input_file)

    # make a dict like other datasets do
    questions = {}
    for line in data:
        pid = str(line["question_id"])
        questions[pid] = line

    output_file = os.path.join(cfg.output_dir, "generation.json")

    # mcts storage if needed (same as mathvision)
    if "mcts" in cfg.prompt_method.name:
        tree_save_dir = os.path.join(cfg.output_dir, "tree_results")
        os.makedirs(tree_save_dir, exist_ok=True)

    # resume logic (same pattern as mathvision)
    old_id = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        for k in questions:
            if (
                k in results
                and "response" in results[k]
                and results[k]["response"] != "Failed to generate response!"
            ):
                questions[k] = results[k]
                old_id.append(k)
    completed_ids = old_id

    lock = threading.Lock()

    def process_query(pid):
        if pid in completed_ids:
            return None

        line = questions[pid]
        # VStar already gives fully formatted text (question + options)
        question = line["text"]
        # build image path
        image_path = os.path.join(data_cfg.image_dir, line["image"])

        try:
            response = model.get_response(question, image_path)

            # same pattern as mathvision: strip "root" if it's an MCTS response
            if "root" in response:
                if "mcts" in cfg.prompt_method.name:
                    tree_save_path = os.path.join(tree_save_dir, f"{pid}.pkl")
                    with open(tree_save_path, "wb") as f:
                        pickle.dump(response["root"], f)
                del response["root"]

            with lock:
                # store everything we need for extraction/scoring
                questions[pid]["question"] = question
                questions[pid]["answer"] = line.get("label", "")  # gold letter
                questions[pid]["decision"] = response
                questions[pid]["response"] = response["response"]

        except Exception as e:
            print(str(e))
            with lock:
                questions[pid]["error"] = str(e)
                questions[pid]["response"] = "Failed to generate response!"
        return pid

    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w") as f:
                    json.dump(questions, f, indent=4)
            print("Results saved.")
        except Exception as e:
            print(e)

    # multithreaded like mathvision
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in questions]
        for i, future in enumerate(tqdm(as_completed(futures), total=len(questions))):
            if future.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()


def gen_mmstar(cfg, model):
    """
    Generate results for MMStar (mmstar.jsonl)
    Each line looks like:
    {
      "id": 0,
      "image": "images/00000.jpg",
      "question": "Which option ... \nOptions: A: ..., B: ..., ...",
      "answer": "A",
      "category": "...",
      "l2_category": "...",
      ...
    }
    """
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")

    # reuse the helper in this file if you already have load_jsonl,
    # otherwise this is what it does:
    def _load_jsonl(fname):
        data = []
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    data = _load_jsonl(input_file)

    # make it a dict keyed by string id, like other datasets
    problems = {str(item["id"]): item for item in data}

    output_file = os.path.join(cfg.output_dir, "generation.json")

    # MCTS stuff, keep same style as mathvision / vstar
    if "mcts" in cfg.prompt_method.name:
        tree_save_dir = os.path.join(cfg.output_dir, "tree_results")
        os.makedirs(tree_save_dir, exist_ok=True)
    else:
        tree_save_dir = None

    # resume support (like mathvision)
    completed_ids = []
    if os.path.exists(output_file):
        print(f"{output_file} exists. Try to resume...")
        with open(output_file, "r") as f:
            old_results = json.load(f)
        for k, v in old_results.items():
            if "response" in v and v["response"] != "Failed to generate response!":
                problems[k] = v
                completed_ids.append(k)
        print(f"{len(completed_ids)} problems already done, skipping them.")

    lock = threading.Lock()

    def process_query(pid):
        if pid in completed_ids:
            return None
        item = problems[pid]

        # MMStar question already contains the options, so we can ask as-is
        question = item["question"]

        # build image path: data/MMStar + images/00000.jpg
        image_path = os.path.join(data_cfg.data_dir, item["image"])

        try:
            response = model.get_response(question, image_path)

            # strip mcts root if present
            if "root" in response:
                if tree_save_dir is not None:
                    with open(os.path.join(tree_save_dir, f"{pid}.pkl"), "wb") as f:
                        pickle.dump(response["root"], f)
                del response["root"]

            with lock:
                item["question"] = question
                # save GT answer so extract/scoring can see it
                item["answer"] = item.get("answer", "")
                item["response"] = response["response"]
                item["decision"] = response

        except Exception as e:
            print(e)
            with lock:
                item["response"] = "Failed to generate response!"
                item["error"] = str(e)

        return pid

    def save_results():
        print(f"Saving to {output_file} ...")
        with lock:
            with open(output_file, "w") as f:
                json.dump(problems, f, indent=4)
        print("Saved.")

    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in problems]
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(problems))):
            if fut.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()

import ast
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

def gen_mmmu_pro(cfg, model):
    """
    Generate for MMMU-Pro (standard_4 / standard_10).
    Handles:
      - options as string repr of list
      - options as real list
      - missing / empty generation.json (resume)
    """
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")

    # --- load jsonl ---
    def _load_jsonl(fname):
        items = []
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    raw_data = _load_jsonl(input_file)

    # make dict keyed by string id
    problems = {str(item["id"]): item for item in raw_data}

    output_file = os.path.join(cfg.output_dir, "generation.json")

    # mcts dir
    tree_save_dir = None
    if "mcts" in cfg.prompt_method.name:
        tree_save_dir = os.path.join(cfg.output_dir, "tree_results")
        os.makedirs(tree_save_dir, exist_ok=True)

    # --- resume logic, but tolerant ---
    completed_ids = []
    if os.path.exists(output_file):
        print(f"{output_file} exists, trying to resume...")
        try:
            with open(output_file, "r") as f:
                old_results = json.load(f)
            for k, v in old_results.items():
                if "response" in v and v["response"] != "Failed to generate response!":
                    problems[k] = v
                    completed_ids.append(k)
            print(f"Resuming: {len(completed_ids)} items already done.")
        except json.JSONDecodeError:
            # file was empty / half-written: ignore and start fresh
            print(f"WARNING: {output_file} is not valid JSON. Starting fresh.")
    else:
        print("No previous results. Starting fresh.")

    lock = threading.Lock()

    def process_query(pid):
        if pid in completed_ids:
            return None

        item = problems[pid]

        # 1) base question
        question = item["question"]

        # 2) options: could be string or list
        options_raw = item.get("options", None)
        options_list = []
        if options_raw:
            if isinstance(options_raw, list):
                options_list = options_raw
            else:
                # string like "['A', 'B', ...]"
                try:
                    options_list = ast.literal_eval(options_raw)
                except Exception:
                    # leave empty if totally weird
                    options_list = []

        # 3) append options as (A) ...
        if options_list:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            opt_lines = []
            for i, opt in enumerate(options_list):
                opt_lines.append(f"({letters[i]}) {opt}")
            question = question.rstrip() + "\n" + "\n".join(opt_lines)

        # 4) image: MMMU-Pro uses list under "images"
        image_rel = None
        if "images" in item and item["images"]:
            image_rel = item["images"][0]
        elif "image" in item:
            image_rel = item["image"]

        if image_rel:
            image_path = os.path.join(data_cfg.image_dir, image_rel)
        else:
            image_path = None

        try:
            response = model.get_response(question, image_path)

            if "root" in response:
                if tree_save_dir is not None:
                    with open(os.path.join(tree_save_dir, f"{pid}.pkl"), "wb") as f:
                        pickle.dump(response["root"], f)
                del response["root"]

            with lock:
                item["question"] = question
                item["answer"] = item.get("answer", "")  # GT letter
                item["response"] = response["response"]
                item["decision"] = response
        except Exception as e:
            print(e)
            with lock:
                item["response"] = "Failed to generate response!"
                item["error"] = str(e)

        return pid

    def save_results():
        print(f"Saving to {output_file} ...")
        with lock:
            with open(output_file, "w") as f:
                json.dump(problems, f, indent=4)
        print("Saved.")

    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in problems]
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(problems))):
            if fut.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()

# utils/generate.py

def gen_mathverse(cfg, model):
    """
    Generate answers for the MathVerse benchmark.
    MathVerse jsonl fields (from your file):
        - question_for_eval: full question text
        - query_wo: additional question for eval
        - image: relative path
        - answer: gold standard answer
    """
    data_cfg = cfg.data
    input_file = os.path.join(data_cfg.data_dir, data_cfg.input_file)
    print(f"Reading {input_file}...")
    data = load_jsonl(input_file)  # This function should load JSONL into a list

    # make a dict like other datasets do
    questions = {}
    for line in data:
        pid = str(line["sample_index"])
        questions[pid] = line

    output_file = os.path.join(cfg.output_dir, "generation.json")

    # mcts storage if needed (same as mathvision)
    if "mcts" in cfg.prompt_method.name:
        tree_save_dir = os.path.join(cfg.output_dir, "tree_results")
        os.makedirs(tree_save_dir, exist_ok=True)

    # resume logic (same pattern as mathvision)
    old_id = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        for k in questions:
            if (
                k in results
                and "response" in results[k]
                and results[k]["response"] != "Failed to generate response!"
            ):
                questions[k] = results[k]
                old_id.append(k)
    completed_ids = old_id

    lock = threading.Lock()

    def process_query(pid):
        if pid in completed_ids:
            return None

        line = questions[pid]
        # MathVerse uses "question_for_eval" and "query_wo"
        question = line["question_for_eval"] + " " + line.get("query_wo", "")
        image_path = os.path.join(data_cfg.data_dir, "images" ,line["image"])
        
        try:
            response = model.get_response(question, image_path)

            # same pattern as mathvision: strip "root" if it's an MCTS response
            if "root" in response:
                if "mcts" in cfg.prompt_method.name:
                    tree_save_path = os.path.join(tree_save_dir, f"{pid}.pkl")
                    with open(tree_save_path, "wb") as f:
                        pickle.dump(response["root"], f)
                    del response["root"]

                with lock:
                    # store everything we need for extraction/scoring
                    questions[pid]["question"] = question
                    questions[pid]["answer"] = line.get("answer", "")  # gold answer
                    questions[pid]["decision"] = response
                    questions[pid]["response"] = response["response"]

            else:
                with lock:
                    questions[pid]["response"] = response["response"]
                    questions[pid]["decision"] = response

        except Exception as e:
            print(str(e))
            with lock:
                questions[pid]["error"] = str(e)
                questions[pid]["response"] = "Failed to generate response!"
        return pid

    def save_results():
        try:
            print(f"Saving results to {output_file}...")
            with lock:
                with open(output_file, "w") as f:
                    json.dump(questions, f, indent=4)
            print("Results saved.")
        except Exception as e:
            print(e)

    # multithreaded like mathvision
    with ThreadPoolExecutor(max_workers=cfg.worker_num) as executor:
        futures = [executor.submit(process_query, pid) for pid in questions]
        for i, future in enumerate(tqdm(as_completed(futures), total=len(questions))):
            if future.result() is not None and i % cfg.save_steps == 0:
                save_results()

    save_results()

# utils/generate.py

def generate(cfg, model):
    if cfg.data.name == "charxiv":
        gen_charxiv(cfg, model)
    elif cfg.data.name == "mathvista":
        gen_mathvista(cfg, model)
    elif cfg.data.name == "mathvision" or cfg.data.name == "mathvision_100":
        gen_mathvision(cfg, model)
    elif cfg.data.name == "vstar" or cfg.data.name == "vstar_100":
        gen_vstar(cfg, model)
    elif cfg.data.name == "mmstar" or cfg.data.name == "mmstar_100":
        gen_mmstar(cfg, model)
    elif cfg.data.name == "mmmu_pro_standard_10" or cfg.data.name == "mmmu_pro_standard_4" or cfg.data.name == "mmmu_pro_standard_10_100" or cfg.data.name == "mmmu_pro_standard_4_100":
        gen_mmmu_pro(cfg, model)
    elif cfg.data.name == "mathverse":
        gen_mathverse(cfg, model)
    else:
        raise ValueError("Dataset not supported")
