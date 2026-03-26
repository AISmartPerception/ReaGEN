import os
import json, logging
from pathlib import Path
import random
import numpy as np
from prompts.stage_n import stage
import torch

from model.model_loader_evo import load_model
from dataset.data_loader_evo import load_data


# from prompts.stage_n.prompt_student_textvqa import compose_prompt
from prompts.stage_n.prompt_student_general import compose_prompt
# from prompts.stage_n.prompt_student_textvqa_simple import compose_prompt_simple

from prompts.feedback import feedback_fn
from utils.bboxes_tok import BBox, _iou
from utils.vqa_soft_acc import combined_accuracy, composite_reward
from utils.config_loader import get_config
from tqdm import tqdm

from prompts.stage_n.chain import Chain

from prompts.stage_n.chain_1 import Chain_1
from prompts.stage_n.prompt_student_general import init_cot, init_cot_generation, init_cot_generation_2
from prompts.stage_n.search_space import SearchSpace, flatten_search_space
from prompts.teacher import edit_chain, edit_chain_attn
from utils.vqa_soft_acc import compute_token_cost

from prompts.teacher import Teacher
# from mutation.mutation import rewritter
# from feedback.feedback import feedback

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
# from utils.draw_attn_heatmap import draw_attn_heatmap_dataset
import copy
import pickle


# Load global configuration
config = get_config()

def set_random_seed(seed: int = None):
    """
    Set random seed for reproducible generation across all libraries.
    
    Args:
        seed: Random seed value. If None, uses config value or defaults to 42.
    """
    
    if seed is None:
        seed = config.get("generation", {}).get("seed", 42)
    
    # Skip seed setting if explicitly set to None
    if seed is None:
        print("Random seed not set - using random behavior")
        return
    
    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        
    # Make CUDA operations deterministic (may impact performance)
    deterministic = config.get("generation", {}).get("deterministic", True)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed} (deterministic mode)")
    else:
        print(f"Random seed set to: {seed} (non-deterministic mode)")


def setup_split_logger(out_dir: str, split: str) -> logging.Logger:
    """
    Creates a logger that writes JSON lines to logs/eval_<split>.jsonl.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"eval.{split}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't duplicate to root

    # Clear existing handlers if re-created in notebooks
    logger.handlers = []

    fh = logging.FileHandler(out_dir / f"eval_{split}.jsonl", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))  # raw JSON per line
    logger.addHandler(fh)

    return logger



def setup_subset(ds_split, orignal = True, partition=0):
    length = config["inference"]["len_data"]
    if orignal:
        subset = ds_split
        my_ids = [i for i in range(len(ds_split))]
        
        # if partition == 0:
        #     id_list = [i for i in range(len(ds_split)) if i % 4 == 0]
        # elif partition == 1:
        #     id_list = [i for i in range(len(ds_split)) if i % 4 == 1]
        # elif partition == 2:
        #     id_list = [i for i in range(len(ds_split)) if i % 4 == 2]
        # elif partition == 3:
        #     id_list = [i for i in range(len(ds_split)) if i % 4 == 3]
        
        # subset = [ds_split[i] for i in id_list]
        # my_ids = id_list
        
    # else:
    #     start_idx = 1
    #     subset = [ds_split[i] for i in TESTING_IDS[start_idx:(start_idx+length)]]
    #     my_ids = TESTING_IDS[start_idx:(start_idx+length)]
    #     # subset = [ds_split[i] for i in test_id]
    #     # my_ids = test_id
 
    return subset, my_ids

def iou_calculation(sample, out):
    gt_box = sample.get("bboxs")[0]
    gt_box = BBox(gt_box[0], gt_box[1], gt_box[2], gt_box[3], conf=None)
    pred_box = BBox(out["bbox_coords"][0], out["bbox_coords"][1], out["bbox_coords"][2], out["bbox_coords"][3], conf=None)
    calc_iou = _iou(gt_box, pred_box)
    return calc_iou, gt_box, pred_box

def run_evolution_1(student_pair, teacher_pair, ds_split, logger: logging.Logger | None = None, 
                  split_name: str = "val", stages: list = None, heatmap: bool = False, ref_key: str = None, partition: int = 0):
    
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]

    s_processor, s_model = student_pair
        
    subset, my_ids = setup_subset(ds_split, orignal=True, partition=partition)
    # subset, my_ids = setup_subset(ds_split, orignal=False)

    # print(len(subset))
    # print(my_ids)
    scores, direct_scores = [], []

    # if config["inference"]["prompt_refine"]:
    #     base_system_prompts = {stage: compose_prompt_simple(stage) for stage in config["inference"]["stages_pool"]}
    #     base_system_prompts["FINAL"] = compose_prompt_simple("FINAL")
    # else:
    #     base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    #     base_system_prompts["FINAL"] = compose_prompt("FINAL")
    #     base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")
    
    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")


    initial_stages = copy.deepcopy(stages)
    # atten_mass_dataset = []
    num_iterations, num_student_calls = 0, 0
    CoT_lengths = []

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:
                # if os.path.exists(config["paths"]["output_dir"] + f"/record_sample_{my_ids[i]}.pkl"):
                #     continue
                # if i < 2:
                #     continue

                stages = copy.deepcopy(initial_stages)
                iterations = config["inference"]["iterations"]
                past_iterations = []
                sorted_dict = []
                
                record_sample_dict = {
                    "idx": my_ids[i],
                    # "question": sample.get("question"),
                    # "refs": sample.get(ref_key),
                    "initial_stages": stages,
                    "final_stages": [],
                    "iteration": {},
                }
                
                with tqdm(total=iterations, desc=f"Iterations", leave=False) as inner_tqdm:
                    for iteration in range(iterations):
                        logger.info(f"Iteration {iteration} of sample {my_ids[i]}")
                        chain = Chain(stages, logger, s_model, s_processor, config, base_system_prompts=base_system_prompts)
                        out = chain.run(i, sample, return_debug=True)
                        record_sample_dict["iteration"][iteration] = {
                            "stages": stages,
                            "stage_outputs": out['stage_outputs'],
                            "question_emb": out['question_emb'],
                            "image_emb": out['image_emb'],
                            "A": out["A"],
                            "a_to_final": out["a_to_final"],
                        }
                        # if iteration == 0:
                        #     record_sample_dict["question_emb"] = out["question_emb"]
                        #     record_sample_dict["image_emb"] = out["image_emb"]
                        #     record_sample_dict["A"] = out["A"]
                        #     record_sample_dict["a_to_final"] = out["a_to_final"]
                        # atten_mass_dataset.append(out["stage_outputs"])
                        
                        # pred = out["answer_raw"]
                        images = out["images"]
                        # refs = sample.get(ref_key)
                        # pred = out["rendered_answer"]
                        answer = sample.get('answer')
                        # score = combined_accuracy(pred, answer)

                        # direct_pred = out["direct_answer_rendered"]
                        # direct_score = combined_accuracy(direct_pred, answer)
                        score, direct_score = combined_accuracy(out, answer)
                
                        # if "BBOX" in stages:
                        #     calc_iou, gt_box, pred_box = iou_calculation(sample, out)
                        #     record["gt_box"] = [gt_box.x1, gt_box.y1, gt_box.x2, gt_box.y2] 
                        #     record["pred_box"] = [pred_box.x1, pred_box.y1, pred_box.x2, pred_box.y2] 
                        #     record["iou"] = calc_iou
                        #     feedback = feedback_fn(chain, sample, score, out, refs, config, calc_iou=calc_iou)

                        # feedback = feedback_fn(chain, sample, score, out, refs, config, base_system_prompts=base_system_prompts)
                        feedback = feedback_fn(chain, sample, score, out, answer, config, base_system_prompts=base_system_prompts)
                        iteration_dict = {
                            "iteration": iteration,
                            'score': score,
                            # 'stages': ', '.join(map(str, stages)),
                            'stages': stages,
                            'feedback': feedback,
                            'images': images,
                            "pred": out["rendered_answer"],
                        }

                        sorted_dict.append(iteration_dict)
                        sorted_dict.sort(key=lambda x: x['score']+(1/len(x['stages'])), reverse=True)
                        optimal_CoT_dict = sorted_dict[0]
                        past_iterations.append(iteration_dict)

                        teacher = Teacher(teacher_pair, config, logger, chain)
                        thinking_text, teacher_output = teacher.generate(sample, past_iterations, optimal_CoT_dict, images)

                        evo_finish = teacher_output.get("evo_finish", False)
                        old_stages = copy.deepcopy(stages)
                        if not evo_finish:
                            stages, base_system_prompts = teacher.apply_edits(teacher_output, stages, base_system_prompts)


                        record = {
                            "idx": my_ids[i],
                            "score": float(score),
                            "question": sample.get("question"),
                            "thinking_text": thinking_text,
                            "pred": out["rendered_answer"],
                            "answer": answer,
                            "teacher_output": teacher_output,
                            "old_stages": old_stages,
                            "new_stages": stages,
                            "direct_pred": out["direct_answer_rendered"],
                            "direct_score": direct_score,
                        }
                        logger.info(json.dumps(record, ensure_ascii=False, indent=4))
                        num_student_calls += len(old_stages)
                        inner_tqdm.update(1)

                        if evo_finish or iteration == iterations - 1:
                            scores.append(optimal_CoT_dict['score'])
                            direct_scores.append(direct_score)
                            num_iterations += iteration
                            CoT_lengths.append(len(optimal_CoT_dict['stages']))
                            logger.info(f"Evolution of sample {my_ids[i]} is done in iteration {iteration}. Score: {optimal_CoT_dict['score']}.")
                            record = {
                                "Idx": my_ids[i],
                                "score": float(optimal_CoT_dict['score']),
                                "question": sample.get("question"),
                                "pred": optimal_CoT_dict['pred'],
                                "answer": answer,
                                "optimal_cot": optimal_CoT_dict['stages'],
                                "direct_pred": out["direct_answer_rendered"],
                                "direct_score": direct_score,
                                # "stage_uncertainties": out.get("stage_uncertainties", {}),
                            }

                            record_sample_dict["final_stages"] = optimal_CoT_dict['stages']
                            # record_sample_dict["images"] = optimal_CoT_dict['images']
                            # record_sample_dict["image_id"] = sample.get("image_id")
                            record_sample_dict["score"] = optimal_CoT_dict['score']
                            record_sample_dict["pred"] = optimal_CoT_dict['pred']
                            # record_sample_dict["direct_pred"] = direct_pred
                            # record_sample_dict["direct_score"] = direct_score
                            
                            os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}", exist_ok=True)
                            output_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/record_sample_{sample['id']}.pkl"
                            with open(output_path, "wb") as f:
                                pickle.dump(record_sample_dict, f)
                            
                            # os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}", exist_ok=True)
                            # with open(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/record_sample_{sample['id']}.json", "w") as f:
                            #     json.dump(record_sample_dict, f, ensure_ascii=False, indent=4)

                            # scores.append(score)

                            logger.info(json.dumps(record, ensure_ascii=False, indent=4))
                            break


            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"CUDA OOM error on sample {i} (idx: {my_ids[i]}): {e}")
                logger.info("Clearing CUDA cache and skipping this sample...")
                torch.cuda.empty_cache()

                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Memory error on sample {i} (idx: {my_ids[i]}): {e}")
                    logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    mean_acc = sum(scores) / max(len(scores), 1)
    mean_direct_score = sum(direct_scores) / max(len(direct_scores), 1)
    if logger is not None:
        logger.info(json.dumps({
            "split": split_name, "summary": True,
            "mean_acc": float(mean_acc), "n": len(scores),
            "total_num_iterations": num_iterations,
            "total_num_student_calls": num_student_calls,
            "mean_CoT_length": sum(CoT_lengths) / max(len(CoT_lengths), 1),
            "mean_direct_score": float(mean_direct_score), "n": len(direct_scores),
        }, ensure_ascii=False))

    # draw_attn_heatmap_dataset(atten_mass_dataset)
    
    
    
def run_evolution_2(student_pair, teacher_pair, ds_split, logger: logging.Logger | None = None, 
                  split_name: str = "val", stages: list = None, heatmap: bool = False, ref_key: str = None, partition: int = 0):
    
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]
        
    os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}", exist_ok=True)

    s_processor, s_model = student_pair
        
    subset, my_ids = setup_subset(ds_split, orignal=True, partition=partition)

    scores, direct_scores = [], []

    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")



    initial_stages = copy.deepcopy(stages)
    num_iterations, num_student_calls = 0, 0
    CoT_lengths = []

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:
    
                output_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/record_sample_{sample['id']}.pkl"
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        record_sample_dict = pickle.load(f)
                    score = record_sample_dict['score']
                    direct_score = record_sample_dict['direct_score']
                    scores.append(score)
                    direct_scores.append(direct_score)
                    logger.info(f"Sample {record_sample_dict['sample']['id']} is already evaluated, score: {score}, direct_score: {direct_score}")
                    logger.info(json.dumps({
                        "idx": record_sample_dict['sample']['id'],
                        "score": float(score),
                        "question": sample.get("question"),
                        "pred": record_sample_dict['pred'],
                        "direct_score": direct_score,
                        "direct_pred": record_sample_dict['direct_pred'],
                    }, ensure_ascii=False, indent=4))
                    # break
                    continue


                search_space_path = config["paths"]["search_space_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/search_space_{sample['id']}.pkl"
                stage_output_path = config["paths"]["stage_outputs_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/stage_outputs_{sample['id']}.pkl"
                os.makedirs(config["paths"]["search_space_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}", exist_ok=True)
                os.makedirs(config["paths"]["stage_outputs_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}", exist_ok=True)
                if os.path.exists(search_space_path):
                    with open(search_space_path, "rb") as f:
                        search_space = pickle.load(f)
                else:
                    search_space = SearchSpace()

                if os.path.exists(stage_output_path):
                    with open(stage_output_path, "rb") as f:
                        stage_outputs = pickle.load(f)
                else:
                    stage_outputs = {}

                # search_space = SearchSpace()
                # stage_outputs = {}

                CoTs = init_cot()
                iterations = config["inference"]["iterations"]
                
                
                answer = sample.get('answer')
                searched_chains = []
                best_score = 0
                shortest_chain, best_chain = None, None
                with tqdm(total=iterations, desc=f"Iterations", leave=False) as inner_tqdm:
                    for iteration in range(iterations):
                        logger.info(f"Iteration {iteration} of sample {my_ids[i]}")
                        
                        for cot in CoTs:
                            logger.info(f"Running for CoT: {cot}")
                            chain = Chain_1(cot, logger, s_model, s_processor, config, search_space=search_space, base_system_prompts=base_system_prompts)

                            if ",".join(cot) in stage_outputs:
                                out = stage_outputs[",".join(cot)]
                            else:
                                out = chain.run(i, sample, return_debug=True)
                                stage_outputs[",".join(cot)] = out

                            # out = chain.run(i, sample, return_debug=True)
                            search_space = chain.search_space
                            reward, pred_score, di_reward = composite_reward(out, answer, searched_chains=searched_chains, stages=cot)
                            # search_space.add_reward(cot, reward, pred_score, out["rendered_answer"])
                            
                            feedback = feedback_fn(chain, sample, pred_score, out, answer, config, base_system_prompts=base_system_prompts)
                            iteration_dict = {
                                "iteration": iteration,
                                'score': pred_score,
                                # 'stages': ', '.join(map(str, stages)),
                                'stages': cot,
                                'feedback': feedback,
                                "pred": out["rendered_answer"],
                            }
                            # sorted_dict.append(iteration_dict)
                            # sorted_dict.sort(key=lambda x: x['score']+(1/len(x['stages'])), reverse=True)
                            # optimal_CoT_dict = sorted_dict[0]
                            # past_iterations.append(iteration_dict)
                            past_iterations = []

                            teacher = Teacher(teacher_pair, config, logger, chain)
                            thinking_text, teacher_output = teacher.generate(sample, past_iterations, iteration_dict, out['images'])
                            
                            # images = out["images"]

                            logger.info(f"Chain {cot} has composite reward: {reward} (with direct pred score: {di_reward})\n")
                            logger.info(f"  -> CoT Pred: {out['rendered_answer']}; Direct Pred: {out['direct_answer_rendered']}\n")
                            # logger.info(f"Search space: {json.dumps(search_space.to_dict(), ensure_ascii=False, indent=4)}")
                            
                        all_chains = flatten_search_space(search_space.root)
                        searched_chains = [chain['stage_seq'] for chain in all_chains]
                        topk_chains = sorted(all_chains, key=lambda x: x['reward'], reverse=True)[:(config["inference"]["topk"])]
                        # topk_chains.append(random.choice(all_chains))
                        CoTs = []
                        
                        for j, chain in enumerate(topk_chains):
                            
                            if shortest_chain is None or len(chain['stage_seq']) < len(shortest_chain['stage_seq']):
                                shortest_chain = chain
                            
                            if best_chain is None or chain['reward'] > best_chain['reward']:
                                best_chain = chain
                                
                            logger.info(f"Top {j+1} Chain: {json.dumps(chain, ensure_ascii=False, indent=4)}")
                            new_chain, op, unique = edit_chain(chain, searched_chains=searched_chains, config=config)
                            CoTs.append(new_chain)
                            if unique:
                                logger.info(f"    -> Edited CoT: {new_chain} (Op: {op})\n")
                            else:
                                logger.info(f"    -> Edited CoT: {new_chain} (Op: {op}) is not unique\n")


                        # for chain in chains:
                        #     logger.info(f"Chain: {json.dumps(chain, ensure_ascii=False, indent=4)}")
                        num_iterations += 1
                        if shortest_chain is not None and len(shortest_chain['stage_seq']) == 1 and di_reward == 1:
                            logger.info(f"Shortest chain {shortest_chain['stage_seq']} is the best chain, and it is unique, so we break the loop\n\n")
                            break

                        inner_tqdm.update(1)

                if best_chain is not None:
                    scores.append(best_chain['score'])
                    CoT_lengths.append(len(best_chain['stage_seq']))
                    
                else:
                    scores.append(0)

                direct_scores.append(di_reward)
                
                logger.info(json.dumps({
                    "idx": sample['id'],
                    "score": float(best_chain['score']),
                    "question": sample.get("question"),
                    "Answer": answer,
                    "pred": best_chain['final_output'],
                    "best_chain": best_chain['stage_seq'],
                    "direct_score": di_reward,
                    "direct_pred": out['direct_answer_rendered'],
                }, ensure_ascii=False, indent=4))
                
                
                record_sample_dict = {
                    "sample": sample,
                    "question_emb": out['question_emb'],
                    "image_emb": out['image_emb'],
                    "score": best_chain['score'],
                    "direct_score": di_reward,
                    "direct_pred": out['direct_answer_rendered'],
                    "pred": best_chain['final_output'],
                    "search_space": search_space,
                }

                
                with open(output_path, "wb") as f:
                    pickle.dump(record_sample_dict, f)

                with open(search_space_path, "wb") as f:
                    pickle.dump(search_space, f)

                with open(stage_output_path, "wb") as f:
                    pickle.dump(stage_outputs, f)


            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"CUDA OOM error on sample {i} (idx: {my_ids[i]}): {e}")
                logger.info("Clearing CUDA cache and skipping this sample...")
                torch.cuda.empty_cache()

                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Memory error on sample {i} (idx: {my_ids[i]}): {e}")
                    logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    mean_acc = sum(scores) / max(len(scores), 1)
    mean_direct_score = sum(direct_scores) / max(len(direct_scores), 1)
    if logger is not None:
        logger.info(json.dumps({
            "split": split_name, "summary": True,
            "mean_acc": float(mean_acc), "n": len(scores),
            "total_num_iterations": num_iterations,
            "total_num_student_calls": num_student_calls,
            "mean_CoT_length": sum(CoT_lengths) / max(len(CoT_lengths), 1),
            "mean_direct_score": float(mean_direct_score), "n": len(direct_scores),
        }, ensure_ascii=False))

    

    
def run_evolution_3(s_model, s_processor, ds_split, logger: logging.Logger | None = None):
    
    os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_attn", exist_ok=True)

    subset, my_ids = setup_subset(ds_split, orignal=True)

    scores, direct_scores = [], []

    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")

    num_iterations, num_student_calls = 0, 0
    iterations = config["inference"]["iterations"]
    CoT_lengths = []
    
    score_iter = {}

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:
                # if os.path.exists(config["paths"]["output_dir"] + f"/record_sample_{my_ids[i]}.pkl"):
                #     continue
                # if i < 2:
                #     continue
                if config["inference"]["attn"]:
                    output_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_evo_4_attn"
                else:
                    output_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_evo_4"
                    
                os.makedirs(output_path, exist_ok=True)
                output_path += f"/record_sample_{sample['id']}.pkl"

                CoTs = init_cot()
                CoTs = random.sample(CoTs, config["gen_training"]["branch"])


                # iterations = config["inference"]["iterations"]

                if 'options' in config['dataset']:
                    if "vision" in config['dataset']['options'][0] or "4" in config['dataset']['options'][0]:
                        dataset_name = config['dataset']['data_id'].split('/')[-1]+ "_" + config['dataset']['options'][0] 
                    elif "10" in config['dataset']['options'][0]:
                        dataset_name = config['dataset']['data_id'].split('/')[-1]
                elif config["dataset"]["vison_only"] and config["dataset"]["data_id"] == "AI4Math/MathVerse":
                    dataset_name = config['dataset']['data_id'].split('/')[-1] + "_vision"
                else:
                    dataset_name = config['dataset']['data_id'].split('/')[-1]
                
                # search_space_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/record_sample_{sample['id']}.pkl"
                search_space_path = config["paths"]["search_space_dir"] + f"/ablation_{dataset_name}/search_space_{sample['id']}.pkl"
                # output_search_space_path = config["paths"]["search_space_dir"] + f"/ablation_{dataset_name}/search_space_{sample['id']}.pkl"
                os.makedirs(config["paths"]["search_space_dir"] + f"/ablation_{dataset_name}", exist_ok=True)
                # if os.path.exists(output_path):
                #     with open(search_space_path, "rb") as f:
                #         sample_dict = pickle.load(f)
                #     search_space = sample_dict['search_space']
                #     all_chains = flatten_search_space(search_space.root)
                #     searched_chains = [chain['stage_seq'] for chain in all_chains]
                #     topk_chains = sorted(all_chains, key=lambda x: x['reward'], reverse=True)[:config["inference"]["topk"]]
                #     best_chain_old.append(topk_chains[0]['stage_seq'])
                #     scores_old.append(topk_chains[0]['score'])
                #     search_space.clear_all_finals()
                    
                if os.path.exists(search_space_path):
                    with open(search_space_path, "rb") as f:
                        # sample_dict = pickle.load(f)
                        search_space = pickle.load(f)


                    all_chains_old = flatten_search_space(search_space.root)
                    searched_chains_old = [chain['stage_seq'] for chain in all_chains_old]
                        
                    # search_space = sample_dict['search_space']
                    # all_chains = flatten_search_space(search_space.root)
                    # searched_chains = [chain['stage_seq'] for chain in all_chains]
                    # topk_chains = sorted(all_chains, key=lambda x: x['reward'], reverse=True)[:config["inference"]["topk"]]
                    # best_chain_old.append(topk_chains[0]['stage_seq'])
                    # scores_old.append(topk_chains[0]['score'])
                    search_space.clear_all_finals()

                else:
                    logger.info(f"Search space for sample {sample['id']} not found, skipping...")
                    
                    continue


                stage_output_path = config["paths"]["stage_outputs_dir"] + f"/ablation_{dataset_name}/stage_outputs_{sample['id']}.pkl"
                os.makedirs(config["paths"]["stage_outputs_dir"] + f"/ablation_{dataset_name}", exist_ok=True)
                if os.path.exists(stage_output_path) and i != 1:
                    with open(stage_output_path, "rb") as f:
                        stage_outputs = pickle.load(f)
                else:
                    stage_outputs = {}

                
                score_iter[sample['id']] = {}
                
                answer = sample.get('answer')
                searched_chains = []
                attn_dict = {}
                shortest_chain, best_chain = None, None
                with tqdm(total=iterations, desc=f"Iterations", leave=False) as inner_tqdm:
                    for iteration in range(iterations):
                        logger.info(f"Iteration {iteration} of sample {my_ids[i]}")
                        number_of_stages, token_cost = 0, 0
                        for cot in CoTs:
                            logger.info(f"Running for CoT: {cot}")
                            number_of_stages += len(cot)
                            chain = Chain_1(cot, logger, s_model, s_processor, config, search_space=search_space, base_system_prompts=base_system_prompts)

                            all_chains = flatten_search_space(search_space.root)
                            searched_chains = [chain['stage_seq'] for chain in all_chains]
                            
                            # out = chain.run(i, sample, return_debug=True)
                            # stage_outputs[",".join(cot)] = out
                            if ",".join(cot) in stage_outputs and cot in searched_chains_old:
                                out = stage_outputs[",".join(cot)]
                            else:
                                out = chain.run(i, sample, return_debug=True)
                                stage_outputs[",".join(cot)] = out
                                searched_chains_old.append(cot)
                            
                            # out = chain.run(i, sample, return_debug=True)
                            token_cost += compute_token_cost(out, s_processor)


                            search_space = chain.search_space
                            reward, pred_score, di_reward = composite_reward(out, answer, searched_chains=searched_chains, stages=cot)
                            search_space.add_reward(cot, reward, pred_score, out["rendered_answer"])

                            attn_dict[",".join(cot)] = {
                                "contribution_dict": out['importance_dict'],
                                "A": out["A"],
                                "a_to_final": out["a_to_final"],
                            }
               
                            # images = out["images"]

                            logger.info(f"Chain {cot} has composite reward: {reward} (with direct pred score: {di_reward})\n")
                            logger.info(f"  -> CoT Pred: {out['rendered_answer']}; Direct Pred: {out['direct_answer_rendered']}\n")
                            # logger.info(f"Search space: {json.dumps(search_space.to_dict(), ensure_ascii=False, indent=4)}")
                        
                        all_chains = flatten_search_space(search_space.root)
                        searched_chains = [chain['stage_seq'] for chain in all_chains]
                        topk_chains = sorted(all_chains, key=lambda x: x['reward'], reverse=True)[:config["inference"]["topk"]]
                        CoTs = []
                        
                        score_iter[sample['id']][iteration] = {
                            "score": topk_chains[0]['score'],
                            "length": number_of_stages,
                            "token_cost": token_cost
                        }
                        
                        for j, chain in enumerate(topk_chains):
                            
                            if shortest_chain is None or len(chain['stage_seq']) < len(shortest_chain['stage_seq']):
                                shortest_chain = chain
                            
                            if best_chain is None or chain['reward'] > best_chain['reward']:
                                best_chain = chain
                                
                            logger.info(f"Top {j+1} Chain: {json.dumps(chain, ensure_ascii=False, indent=4)}")
                            if config["inference"]["attn"]:
                                new_chain, op, unique = edit_chain_attn(chain, attn_dict=attn_dict, searched_chains=all_chains, config=config)
                            else:
                                new_chain, op, unique = edit_chain(chain, searched_chains=searched_chains, config=config)

                            if unique:
                                CoTs.append(new_chain)
                                logger.info(f"    -> Edited CoT: {new_chain} (Op: {op})\n")
                                logger.info(f"    -> Attn Dict: {attn_dict[','.join(chain['stage_seq'])]}\n")
                            else:
                                logger.info(f"    -> Edited CoT: {new_chain} (Op: {op}) is not unique\n")
                                logger.info(f"    -> Attn Dict: {attn_dict[','.join(chain['stage_seq'])]}\n")


                        # for chain in chains:
                        #     logger.info(f"Chain: {json.dumps(chain, ensure_ascii=False, indent=4)}")
                        num_iterations += 1
                        if shortest_chain is not None and len(shortest_chain['stage_seq']) == 1 and di_reward == 1:
                            logger.info(f"Shortest chain {shortest_chain['stage_seq']} is the best chain, and it is unique, so we break the loop\n\n")
                            break

                        inner_tqdm.update(1)

                


                if best_chain is not None:
                    scores.append(best_chain['score'])
                    CoT_lengths.append(len(best_chain['stage_seq']))
                    
                else:
                    scores.append(0)
                direct_scores.append(di_reward)
                
                logger.info(json.dumps({
                    "idx": my_ids[i],
                    "score": float(best_chain['score']),
                    "question": sample.get("question"),
                    "Answer": answer,
                    "pred": best_chain['final_output'],
                    "best_chain": best_chain['stage_seq'],
                    "direct_score": di_reward,
                    "direct_pred": out['direct_answer_rendered'],
                }, ensure_ascii=False, indent=4))
                
                record_sample_dict = {
                    "sample": sample,
                    "question_emb": out['question_emb'],
                    "image_emb": out['image_emb'],
                    "score": best_chain['score'],
                    "direct_score": di_reward,
                    "direct_pred": out['direct_answer_rendered'],
                    "pred": best_chain['final_output'],
                    "search_space": search_space,
                    "score_iter": score_iter[sample['id']],
                }

                
                # with open(output_path, "wb") as f:
                #     pickle.dump(record_sample_dict, f)
                with open(stage_output_path, "wb") as f:
                    pickle.dump(stage_outputs, f)

                with open(search_space_path, "wb") as f:
                    # search_space.clear_all_finals()
                    pickle.dump(search_space, f)


            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"CUDA OOM error on sample {i} (idx: {my_ids[i]}): {e}")
                logger.info("Clearing CUDA cache and skipping this sample...")
                torch.cuda.empty_cache()

                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Memory error on sample {i} (idx: {my_ids[i]}): {e}")
                    logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    mean_acc = sum(scores) / max(len(scores), 1)
    mean_direct_score = sum(direct_scores) / max(len(direct_scores), 1)
    
    sum_sample_scores_iter = [0 for _ in range(iterations)]
    sum_sample_lengths_iter = [0 for _ in range(iterations)]
    sum_sample_token_costs_iter = [0 for _ in range(iterations)]
    for _, sample_scores in score_iter.items():
        for iter in range(iterations):
            if iter in sample_scores:
                sum_sample_scores_iter[iter] += sample_scores[iter]['score']
                sum_sample_lengths_iter[iter] += sample_scores[iter]['length']
                sum_sample_token_costs_iter[iter] += sample_scores[iter]['token_cost']

                # old_length = sample_scores[iter]['length']

            else:
                sum_sample_scores_iter[iter] += 1
                sum_sample_lengths_iter[iter] += 0
                sum_sample_token_costs_iter[iter] += 0
        # list_scores = list(sample_scores.values()["score"])
        # list_lengths = list(sample_scores.values()["length"])
        # if len(list_scores) != iterations:
        #     list_scores.extend([1 for _ in range(iterations - len(list_scores))])
        # if len(list_lengths) != iterations:
        #     list_lengths.extend([1 for _ in range(iterations - len(list_lengths))])
        # sum_sample_scores_iter = [sum_sample_scores_iter[i] + list_scores[i] for i in range(iterations)]
        # sum_sample_lengths_iter = [sum_sample_lengths_iter[i] + list_lengths[i] for i in range(iterations)]

    
    mean_sample_score_iter = [sum_sample_scores_iter[i]/max(len(direct_scores), 1) for i in range(iterations)]
    mean_sample_length_iter = [sum_sample_lengths_iter[i]/max(len(direct_scores), 1) for i in range(iterations)]
    mean_sample_token_cost_iter = [sum_sample_token_costs_iter[i]/max(len(direct_scores), 1) for i in range(iterations)]
    
    # cumulative summation:
    for iter in range(iterations-1, -1, -1):
        # if iter == 0:
        #     mean_sample_length_iter[iter] = mean_sample_length_iter[iter]
        #     mean_sample_token_cost_iter[iter] = mean_sample_token_cost_iter[iter]
        # else:
        #     mean_sample_length_iter[iter] = mean_sample_length_iter[iter] + mean_sample_length_iter[iter-1]
        #     mean_sample_token_cost_iter[iter] = sum(mean_sample_token_cost_iter[iter])
        mean_sample_length_iter[iter] = sum(mean_sample_length_iter[:iter+1])
        mean_sample_token_cost_iter[iter] = sum(mean_sample_token_cost_iter[:iter+1])
    
    if logger is not None:
        logger.info(json.dumps({
            "summary": True,
            "mean_acc": float(mean_acc), "n": len(scores),
            "total_num_iterations": num_iterations,
            "total_num_student_calls": num_student_calls,
            "mean_CoT_length": sum(CoT_lengths) / max(len(CoT_lengths), 1),
            "mean_direct_score": float(mean_direct_score), "n": len(direct_scores),
        }, ensure_ascii=False))
        

        logger.info(mean_sample_score_iter)
        logger.info(mean_sample_length_iter)
        logger.info(mean_sample_token_cost_iter)

    

def run_inference_attn_gen(model, processor, ds_split, ref_key=None, logger: logging.Logger | None = None):
    '''
    run inference for a single or two stages
    '''
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]

    s_processor, s_model = processor, model

    subset, my_ids = setup_subset(ds_split, orignal=True)

    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:
              
                output_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_cot_init_attn/cot_init_attn_{sample['id']}.pkl"
                os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_cot_init_attn", exist_ok=True)
                if os.path.exists(output_path):
                    continue
                
                search_space_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/record_sample_{sample['id']}.pkl"
                os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}", exist_ok=True)
                if not os.path.exists(search_space_path):
                    continue

                with open(search_space_path, "rb") as f:
                    sample_dict = pickle.load(f)
                    
                    
                search_space = sample_dict['search_space']
                CoTs = init_cot_generation(search_space, config)

                record_init_cot_dict = {
                    "sample": sample,
                    "score": sample_dict['score'],
                    "direct_score": sample_dict['direct_score'],
                    "direct_pred": sample_dict['direct_pred'],
                    "pred": sample_dict['pred'],
                    "init": {}
                }

                answer = sample.get('answer')
                searched_chains = []
                for j, cot in enumerate(CoTs):
                    logger.info(f"Running for CoT #{j}: {cot}")
                    chain = Chain_1(cot, logger, s_model, s_processor, config, search_space=search_space, base_system_prompts=base_system_prompts)
                    out = chain.run(i, sample, return_debug=True)
                    search_space = chain.search_space
                    reward, pred_score, _ = composite_reward(out, answer, searched_chains=searched_chains)
                    search_space.add_reward(cot, reward, pred_score, out["rendered_answer"])

                    record_init_cot_dict["init"][" ".join(cot)] = {
                        "question_emb": out['question_emb'],
                        "image_emb": out['image_emb'],
                        "A": out["A"],
                        "a_to_final": out["a_to_final"],
                        "reward": reward,
                        "pred_score": pred_score,
                        "pred": out["rendered_answer"],
                    }

                record_init_cot_dict['search_space'] = search_space

                with open(output_path, "wb") as f:
                    pickle.dump(record_init_cot_dict, f)
                

            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"CUDA OOM error on sample {i} (idx: {my_ids[i]}): {e}")
                logger.info("Clearing CUDA cache and skipping this sample...")
                torch.cuda.empty_cache()
                continue

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Memory error on sample {i} (idx: {my_ids[i]}): {e}")
                    logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise


def run_inference_attn_gen_2(model, processor, ds_split, ref_key=None, logger: logging.Logger | None = None):
    '''
    run inference for a single or two stages
    '''
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]

    s_processor, s_model = processor, model

    subset, my_ids = setup_subset(ds_split, orignal=True)
    
    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:
              
                if 'options' in config['dataset']:
                    dataset_name = config['dataset']['data_id'].split('/')[-1]+ "_" + config['dataset']['options'][0] 
                elif config["dataset"]["vison_only"] and config["dataset"]["data_id"] == "AI4Math/MathVerse":
                    dataset_name = config['dataset']['data_id'].split('/')[-1] + "_vision"
                else:
                    dataset_name = config['dataset']['data_id'].split('/')[-1]
                    
                output_path = config["paths"]["output_dir"] + f"/{dataset_name}_cot_init_attn/cot_init_attn_{sample['id']}.pkl"
                os.makedirs(config["paths"]["output_dir"] + f"/{dataset_name}_cot_init_attn", exist_ok=True)
                if os.path.exists(output_path):
                    continue
                
                stage_output_path = config["paths"]["stage_outputs_dir"] + f"/{dataset_name}/stage_outputs_{sample['id']}.pkl"
                os.makedirs(config["paths"]["stage_outputs_dir"] + f"/{dataset_name}", exist_ok=True)
                if os.path.exists(stage_output_path):
                    with open(stage_output_path, "rb") as f:
                        sample_CoT_outputs = pickle.load(f)
                else:
                    sample_CoT_outputs = {}

                search_space_path = config["paths"]["search_space_dir"] + f"/{dataset_name}/search_space_{sample['id']}.pkl"
                os.makedirs(config["paths"]["search_space_dir"] + f"/{dataset_name}", exist_ok=True)
                if os.path.exists(search_space_path):
                    with open(search_space_path, "rb") as f:
                        search_space = pickle.load(f)
                else:
                    search_space = SearchSpace()
                    


                
                # search_space_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}/record_sample_{sample['id']}.pkl"
                # os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}", exist_ok=True)
                # if not os.path.exists(search_space_path):
                #     continue

                # with open(search_space_path, "rb") as f:
                #     sample_dict = pickle.load(f)
                    
                
                # search_space = sample_dict['search_space']
                # CoTs = init_cot_generation(search_space, config)
                CoTs = init_cot_generation_2(config)

                record_init_cot_dict = {
                    "sample": sample,
                    "init": {}
                }

                answer = sample.get('answer')
                searched_chains = []
                for j, cot in enumerate(CoTs):
                    logger.info(f"Running for CoT #{j}: {cot}")
                    chain = Chain_1(cot, logger, s_model, s_processor, config, search_space=search_space, base_system_prompts=base_system_prompts)
                    
                    if ','.join(cot) in sample_CoT_outputs:
                        out = sample_CoT_outputs[','.join(cot)]
                    else:
                        out = chain.run(i, sample, return_debug=True)
                        sample_CoT_outputs[','.join(cot)] = out
                    
                    # if ','.join(cot) not in sample_CoT_outputs:
                    #     sample_CoT_outputs[','.join(cot)] = out
                    
                    search_space = chain.search_space
                    reward, pred_score, direct_score = composite_reward(out, answer, searched_chains=searched_chains, stages=cot)
                    search_space.add_reward(cot, reward, pred_score, out["rendered_answer"])

                    record_init_cot_dict["init"][" ".join(cot)] = {
                        "question_emb": out['question_emb'],
                        "image_emb": out['image_emb'],
                        "A": out["A"],
                        "a_to_final": out["a_to_final"],
                        "reward": reward,
                        "pred_score": pred_score,
                        "pred": out["rendered_answer"],
                    }

                record_init_cot_dict['search_space'] = search_space
                record_init_cot_dict['direct_score'] = direct_score
                record_init_cot_dict['direct_pred'] = out['direct_answer_rendered']
  

                with open(output_path, "wb") as f:
                    pickle.dump(record_init_cot_dict, f)

                with open(search_space_path, "wb") as f:
                    pickle.dump(search_space, f)
                    
                with open(stage_output_path, "wb") as f:
                    pickle.dump(sample_CoT_outputs, f)

            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"CUDA OOM error on sample {i} (idx: {my_ids[i]}): {e}")
                logger.info("Clearing CUDA cache and skipping this sample...")
                torch.cuda.empty_cache()
                continue

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Memory error on sample {i} (idx: {my_ids[i]}): {e}")
                    logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise




def run_inference(model, processor, ds_split, ref_key=None, logger: logging.Logger | None = None, 
                split_name: str = "val", stages: list = None, heatmap: bool = False, teacher_pair: tuple = None):
    '''
    run inference for a single or two stages
    '''
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]

    s_processor, s_model = processor, model

    subset, my_ids = setup_subset(ds_split, orignal=True)
    scores, direct_scores = [], []
    
    # if config["inference"]["prompt_refine"]:
    #     base_system_prompts = {stage: compose_prompt_simple(stage) for stage in config["inference"]["stages_pool"]}
    #     base_system_prompts["FINAL"] = compose_prompt_simple("FINAL")
    #     base_system_prompts["DIRECT_ANSWER"] = compose_prompt_simple("DIRECT_ANSWER")
    # else:   
    #     base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    #     base_system_prompts["FINAL"] = compose_prompt("FINAL")
    #     base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")
    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:
                # if i != 94:
                #     continue
                search_space = SearchSpace()
                # chain = Chain(stages, logger, s_model, s_processor, config, base_system_prompts=base_system_prompts)
                chain = Chain_1(stages, logger, s_model, s_processor, config, search_space=search_space, base_system_prompts=base_system_prompts)

                out = chain.run(i, sample, return_debug=True)

                # atten_mass_dataset.append(out["stage_outputs"])
                
                
                # pred = out["answer_raw"]
                # refs = sample.get(ref_key)
                # pred = out["rendered_answer"]
                # answer = sample.get('answer')
                # score = combined_accuracy(pred, answer)
                
                # direct_pred = out["direct_answer_rendered"]
                # direct_score = combined_accuracy(direct_pred, answer)
                # score, direct_score = combined_accuracy(out, answer)
                _, direct_score = combined_accuracy(out, out["gt"])
                
                
                scores.append(0)
                direct_scores.append(direct_score)
                record = {
                    "idx": my_ids[i],
                    # "score": float(score),
                    # "final_validity": out["final_validity"],
                    "question": sample.get("question"),
                    "options": sample.get("options"),
                    "difficulty": sample.get("topic_difficulty"),
                    "subject": sample.get("subject"),
                    # "pred": out["rendered_answer"],
                    # "pred_raw": out["answer_raw"],
                    "answer": out["gt"],
                    "direct_pred": out["direct_answer_rendered"],
                    "direct_pred_raw": out["direct_answer_raw"],
                    "direct_score": direct_score,
                }

                if len(stages) > 0 and config["inference"]["attn"]:
                    record["A"] = out.get("A", {}).tolist()
                    record["a_to_final"] = out.get("a_to_final", {}).tolist()
                    record["importance_dict"] = out.get("importance_dict", {})
                    record["blackboard_text"] = out.get("blackboard_text", "")

                logger.info(json.dumps(record, ensure_ascii=False, indent=4))
        
                '''
                if "BBOX" in stages:
                    gt_box = sample.get("bboxs")[0]
                    gt_box = BBox(gt_box[0], gt_box[1], gt_box[2], gt_box[3], conf=None)
                    pred_box = BBox(out["bbox_coords"][0], out["bbox_coords"][1], out["bbox_coords"][2], out["bbox_coords"][3], conf=None)
                    calc_iou = _iou(gt_box, pred_box)
            
                    ious.append(calc_iou)
                '''

            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"CUDA OOM error on sample {i} (idx: {my_ids[i]}): {e}")
                logger.info("Clearing CUDA cache and skipping this sample...")
                torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Memory error on sample {i} (idx: {my_ids[i]}): {e}")
                    logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
                
    mean_acc = sum(scores) / max(len(scores), 1)
    mean_acc_direct = sum(direct_scores) / max(len(direct_scores), 1)
    # iou = sum(ious) / max(len(ious), 1)
    if logger is not None:
        logger.info(json.dumps({
            "split": split_name, "summary": True,
            # "mean_iou": float(iou),
            "mean_acc": float(mean_acc),
            "mean_direct_acc": float(mean_acc_direct), "n": len(direct_scores),
            # "success_ids": success_ids,
        }, ensure_ascii=False))

    # draw_attn_heatmap_dataset(atten_mass_dataset)
    

from prompts.stage_n.internVL import run_internVL
def run_inference_internVL(model, tokenizer, ds_split, ref_key=None, logger: logging.Logger | None = None, 
                split_name: str = "val", stages: list = None, heatmap: bool = False, teacher_pair: tuple = None):
    '''
    run inference for a single or two stages
    '''
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]


    subset, my_ids = setup_subset(ds_split, orignal=True)
    direct_scores = []
    

    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:

                out = run_internVL(model, tokenizer, sample, return_debug=True, config=config, system_prompt=base_system_prompts["DIRECT_ANSWER"], logger=logger)

     
                _, direct_score = combined_accuracy(out, out["gt"])
                
                
                direct_scores.append(direct_score)
                record = {
                    "idx": sample['id'],
                    "answer": out["gt"],
                    "direct_pred": out["direct_answer_rendered"],
                    "direct_pred_raw": out["direct_answer_raw"],
                    "direct_score": direct_score,
                }


                logger.info(json.dumps(record, ensure_ascii=False, indent=4))
        


            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"CUDA OOM error on sample {i} (idx: {my_ids[i]}): {e}")
                logger.info("Clearing CUDA cache and skipping this sample...")
                torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Memory error on sample {i} (idx: {my_ids[i]}): {e}")
                    logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
                
    mean_acc_direct = sum(direct_scores) / max(len(direct_scores), 1)
    if logger is not None:
        logger.info(json.dumps({
            "split": split_name, "summary": True,
            "mean_direct_acc": float(mean_acc_direct), "n": len(direct_scores),
        }, ensure_ascii=False))


def main(stages=None, evo=0, seed=None, device_id_student=0, device_id_teacher=1, partition=0):
    # Set random seed for reproducible generation
    set_random_seed(seed)
    
    # Use config values if not provided
    if stages is None:
        stages = config["inference"]["stages"]
    
    model_name = config["model"]["model_id_student"].split("/")[-1]
    ds = load_data(config["dataset"]["data_id"], config["dataset"]["local_data_dir"], config, args)
    student = load_model(config["model"]["model_id_student"], device=device_id_student, API=False, 
                                      local_dir=config["model"]["local_model_dir_student"],
                                      use_flash=config["model"]["use_flash"])

    if evo == 1:
        s_name, t_name = model_name, config["model"]["model_id_teacher"].split("/")[-1] if "Qwen" in config["model"]["model_id_teacher"] else config["model"]["model_id_teacher"]
        if "Qwen" in config["model"]["model_id_teacher"]:
            teacher = load_model(config["model"]["model_id_teacher"], device=device_id_teacher, API=False, 
                                      local_dir=config["model"]["local_model_dir_teacher"],
                                      quant_conf=config["model"]["quant_conf"],
                                      use_flash=config["model"]["use_flash"])
        else:
            teacher = load_model(config["model"]["model_id_teacher"], device=device_id_teacher, API=True)
        
        log_path = config["paths"]["logs_dir_evo"] + "_" + config["dataset"]["data_id"].split("/")[-1] + "_evo"
        # if config["inference"]["prompt_refine"]:
        #     log_path += "_prompt_refine"
        
        if config["inference"]["attn"]:
            log_path += "_Attn"
        
        name = [stage.split(".")[0] for stage in stages]
        if "Qwen" in config["model"]["model_id_teacher"]:
            evo_logger = setup_split_logger(log_path, f"s_{s_name}_t_{t_name}_q_{config['model']['quant_conf']}_stages-{'-'.join(name)}")
        else:
            evo_logger = setup_split_logger(log_path, f"s_{s_name}_t_{t_name}_stages-{'-'.join(name)}")
        # evo_logger = setup_split_logger(log_path, f"s_{s_name}_t_{t_name}_q_{config['model']['quant_conf']}_stages-{'-'.join(name)}")

        # run_evolution(student, teacher, ds["train"], logger=evo_logger, split_name="train", stages=stages)
        # run_evolution_1(student, teacher, ds["train"], logger=evo_logger, split_name="train", stages=stages, partition=partition)
        
        run_evolution_1(student, teacher, ds, logger=evo_logger, split_name="train", stages=stages, partition=partition)
        # run_evolution_2(student, teacher, ds, logger=evo_logger, split_name="train", stages=stages, partition=partition)

    elif evo == 2:
        s_name, t_name = model_name, config["model"]["model_id_teacher"].split("/")[-1] if "Qwen" in config["model"]["model_id_teacher"] else config["model"]["model_id_teacher"]
        if "Qwen" in config["model"]["model_id_teacher"]:
            teacher = load_model(config["model"]["model_id_teacher"], device=device_id_teacher, API=False, 
                                      local_dir=config["model"]["local_model_dir_teacher"],
                                      quant_conf=config["model"]["quant_conf"],
                                      use_flash=config["model"]["use_flash"])
        else:
            teacher = load_model(config["model"]["model_id_teacher"], device=device_id_teacher, API=True)
        
        log_path = config["paths"]["logs_dir_evo"] + "_" + config["dataset"]["data_id"].split("/")[-1] + f"_evo3"
        # log_path = config["paths"]["logs_dir_evo"] + "_" + config["dataset"]["data_id"].split("/")[-1] + f"_evo2_datagen"
        # if config["inference"]["prompt_refine"]:
        #     log_path += "_prompt_refine"
        
        if config["inference"]["attn"]:
            log_path += "_Attn"
        
        name = [stage.split(".")[0] for stage in stages]
        if "Qwen" in config["model"]["model_id_teacher"]:
            evo_logger = setup_split_logger(log_path, f"s_{s_name}_t_{t_name}_q_{config['model']['quant_conf']}_stages-{'-'.join(name)}")
        else:
            evo_logger = setup_split_logger(log_path, f"s_{s_name}_partition_{partition}")
            # evo_logger = setup_split_logger(log_path, f"s_{s_name}_partition_{partition}_reverse")
            
        run_evolution_2(student, teacher, ds, logger=evo_logger, split_name="train", stages=stages, partition=partition)
        # processor, model = student
        # Prepare initial states for GEN training
        # run_inference_attn_gen_2(model, processor, ds, logger=evo_logger)


    elif evo == 3:
        processor, model = student

        if 'options' in config['dataset']:
            dataset_name = config['dataset']['data_id'].split('/')[-1]+ "_" + config['dataset']['options'][0] 
        elif config["dataset"]["vison_only"] and config["dataset"]["data_id"] == "AI4Math/MathVerse":
            dataset_name = config['dataset']['data_id'].split('/')[-1] + "_vision"
        else:
            dataset_name = config['dataset']['data_id'].split('/')[-1]
        # log_path = config["paths"]["logs_dir_evo"] + "_" + config["dataset"]["data_id"].split("/")[-1] + "_cot_init_attn_options_" + config["dataset"]["options"][0]
        log_path = config["paths"]["logs_dir_evo"] + "_" + dataset_name + "_cot_init_attn"
        # log_path = log_path + "_reverse"
        # logger = setup_split_logger(log_path, f"{model_name}_partition_{partition}")
        logger = setup_split_logger(log_path, f"{model_name}")

        # run_inference_attn_gen(model, processor, ds, logger=logger)
        run_inference_attn_gen_2(model, processor, ds, logger=logger)

    elif evo == 4:
        processor, model = student
        log_path = config["paths"]["logs_dir_evo"] + "_" + config["dataset"]["data_id"].split("/")[-1] + "_evo4"
        if config["inference"]["attn"]:
            # log_path += "_Attn"
            logger = setup_split_logger(log_path, f"ablation_{model_name}_Attn_branch_{config['gen_training']['branch']}")
        else:
            logger = setup_split_logger(log_path, f"ablation_{model_name}_branch_{config['gen_training']['branch']}")
        run_evolution_3(model, processor, ds, logger=logger)

    elif evo == 0:
        # processor, model = student
        # if config["inference"]["CoT"]:
        #     teacher = load_model(config["model"]["model_id_teacher"], device_id=device_id_teacher, API=True)
        #     teacher_pair = (teacher[0], teacher[1])

        log_path = config["paths"]["logs_dir_evo"] + "_direct_answer"
        # if config["inference"]["attn"]:
        #     log_path += "_Attn"
        
        # name = [stage.split(".")[0] for stage in stages]
        stages = []
        logger = setup_split_logger(log_path, f"{model_name}_stages-direct_answer_{config['dataset']['data_id'].split('/')[-1]}")
        # if config["inference"]["CoT"]:
        #     run_inference(model, processor, ds["train"], logger=logger, split_name="train", stages=stages, teacher_pair=teacher_pair)
        # else:
        
        if "Qwen" in config["model"]["model_id_student"]:
            processor, model = student
            run_inference(model, processor, ds, logger=logger, split_name="train", stages=stages)
        
        if "InternVL" in config["model"]["model_id_student"]:
            model, tokenizer = student
            run_inference_internVL(model, tokenizer, ds, logger=logger, split_name="train", stages=stages)
        



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GEPA inference with optional seed control')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible generation')
    parser.add_argument('--evo', type=int, default=0, help='Run evolution optimization')
    # parser.add_argument('--stage_evo', action='store_true', help='Run evolution optimization')
    parser.add_argument('--gpu-student', type=int, default=0, help='Student device ID')
    parser.add_argument('--gpu-teacher', type=int, default=0, help='Teacher device ID')
    parser.add_argument('--partition', type=int, default=0, help='Partition')
    parser.add_argument('--attn', type=int, default=0, help='Attn')
    args = parser.parse_args()
    
    if args.attn:
        config["inference"]["attn"] = True
    else:
        config["inference"]["attn"] = False

    # Now uses config values by default, but can still be overridden
    main(evo=args.evo, seed=args.seed, device_id_student=args.gpu_student, device_id_teacher=args.gpu_teacher, partition=args.partition)

