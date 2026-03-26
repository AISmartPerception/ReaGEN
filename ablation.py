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
from prompts.teacher import edit_chain, edit_chain_attn, edit_chain_del_only
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
        
    return subset, my_ids


    
def run_ablation(s_model, s_processor, ds_split, logger: logging.Logger | None = None):
    
    os.makedirs(config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_attn", exist_ok=True)

    subset, my_ids = setup_subset(ds_split, orignal=True)

    direct_scores = []

    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")

    iterations = 3
    
    score_iter = {}

    with tqdm(enumerate(subset), desc=f"Evaluating {config['dataset']['data_id'].split('/')[-1]}", total=len(subset)) as outer_tqdm:
        for i, sample in outer_tqdm:
            try:

                if config["inference"]["attn"]:
                    output_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_evo_4_attn"
                else:
                    output_path = config["paths"]["output_dir"] + f"/{config['dataset']['data_id'].split('/')[-1]}_evo_4"
                    
                os.makedirs(output_path, exist_ok=True)
                output_path += f"/record_sample_{sample['id']}.pkl"

                # CoTs = init_cot()
                # CoTs = random.sample(CoTs, config["gen_training"]["branch"])


                if 'options' in config['dataset']:
                    if "vision" in config['dataset']['options'][0] or "4" in config['dataset']['options'][0]:
                        dataset_name = config['dataset']['data_id'].split('/')[-1]+ "_" + config['dataset']['options'][0] 
                    elif "10" in config['dataset']['options'][0]:
                        dataset_name = config['dataset']['data_id'].split('/')[-1]
                elif config["dataset"]["vison_only"] and config["dataset"]["data_id"] == "AI4Math/MathVerse":
                    dataset_name = config['dataset']['data_id'].split('/')[-1] + "_vision"
                else:
                    dataset_name = config['dataset']['data_id'].split('/')[-1]
                
                search_space_path = config["paths"]["search_space_dir"] + f"/ablation_{dataset_name}/search_space_{sample['id']}.pkl"
                os.makedirs(config["paths"]["search_space_dir"] + f"/ablation_{dataset_name}", exist_ok=True)
                
                CoTs = []
                if os.path.exists(search_space_path):
                    with open(search_space_path, "rb") as f:
                        search_space = pickle.load(f)

                    all_chains_old = flatten_search_space(search_space.root)
                    
                    searched_chains_old = [chain['stage_seq'] for chain in all_chains_old]
                    for chain in all_chains_old:
                        if len(chain['stage_seq']) >= 3 and len(chain['stage_seq']) <= 7 and chain['reward'] > 0.5:
                            CoTs.append(chain['stage_seq'])
                            
                    # if len(CoTs) == 0:
                    #     CoTs.append(searched_chains_old[0])

                    search_space.clear_all_finals()

                else:
                    logger.info(f"Search space for sample {sample['id']} not found, skipping...")
                    
                    continue


                stage_output_path = config["paths"]["stage_outputs_dir"] + f"/ablation_{dataset_name}/stage_outputs_{sample['id']}.pkl"
                os.makedirs(config["paths"]["stage_outputs_dir"] + f"/ablation_{dataset_name}", exist_ok=True)
                if os.path.exists(stage_output_path):
                    with open(stage_output_path, "rb") as f:
                        stage_outputs = pickle.load(f)
                else:
                    stage_outputs = {}

                
                score_iter[sample['id']] = {}

                # CoTs = init_cot()
                CoTs = random.sample(CoTs, 3 if len(CoTs) > 3 else len(CoTs))
                
                answer = sample.get('answer')
                searched_chains = []
                attn_dict = {}
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
               

                            logger.info(f"Chain {cot} has composite reward: {reward} (with direct pred score: {di_reward})\n")
                            logger.info(f"  -> CoT Pred: {out['rendered_answer']}; Direct Pred: {out['direct_answer_rendered']}\n")
                        
                        all_chains = flatten_search_space(search_space.root)
                        # searched_chains = [chain if chain['stage_seq'] in CoTs else None for chain in all_chains]

                        affect_performance = False
                        searched_chains = []
                        for chain in all_chains:
                            if chain['stage_seq'] in CoTs:
                                searched_chains.append(chain)

                                if chain['reward'] < 0.5:
                                    affect_performance = True

                        CoTs = []

                        score_iter[sample['id']][iteration] = {
                            "score": 1 if not affect_performance else 0,
                            "length": number_of_stages,
                            "token_cost": token_cost
                        }
                        
                        for j, chain in enumerate(searched_chains):
                                
                            logger.info(f"Top {j+1} Chain: {json.dumps(chain, ensure_ascii=False, indent=4)}")
                            # if config["inference"]["attn"]:
                            #     new_chain, op, unique = edit_chain_attn(chain, attn_dict=attn_dict, searched_chains=all_chains, config=config)
                            # else:
                            #     new_chain, op, unique = edit_chain(chain, searched_chains=searched_chains, config=config)
                            new_chain, op, unique = edit_chain_del_only(chain, attn_dict=attn_dict, searched_chains=searched_chains, config=config)

                            if unique:
                                CoTs.append(new_chain)
                                logger.info(f"    -> Edited CoT: {new_chain} (Op: {op})\n")
                                logger.info(f"    -> Attn Dict: {attn_dict[','.join(chain['stage_seq'])]}\n")
                            else:
                                logger.info(f"    -> Edited CoT: {new_chain} (Op: {op}) is not unique\n")
                                logger.info(f"    -> Attn Dict: {attn_dict[','.join(chain['stage_seq'])]}\n")

                        inner_tqdm.update(1)


                # if best_chain is not None:
                #     scores.append(best_chain['score'])
                # else:
                #     scores.append(0)
                direct_scores.append(di_reward)
                
                logger.info(json.dumps({
                    "idx": my_ids[i],
                    "question": sample.get("question"),
                    "Answer": answer,
                    "direct_score": di_reward,
                    "direct_pred": out['direct_answer_rendered'],
                }, ensure_ascii=False, indent=4))


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


    sum_sample_scores_iter = [0 for _ in range(iterations)]
    sum_sample_lengths_iter = [0 for _ in range(iterations)]
    sum_sample_token_costs_iter = [0 for _ in range(iterations)]
    for _, sample_scores in score_iter.items():
        for iter in range(iterations):
            if iter in sample_scores:
                sum_sample_scores_iter[iter] += sample_scores[iter]['score']
                sum_sample_lengths_iter[iter] += sample_scores[iter]['length']
                sum_sample_token_costs_iter[iter] += sample_scores[iter]['token_cost']

            else:
                sum_sample_scores_iter[iter] += 1
                sum_sample_lengths_iter[iter] += 0
                sum_sample_token_costs_iter[iter] += 0

    
    mean_sample_score_iter = [sum_sample_scores_iter[i]/max(len(direct_scores), 1) for i in range(iterations)]
    mean_sample_length_iter = [sum_sample_lengths_iter[i]/max(len(direct_scores), 1) for i in range(iterations)]
    mean_sample_token_cost_iter = [sum_sample_token_costs_iter[i]/max(len(direct_scores), 1) for i in range(iterations)]
    
    # cumulative summation:
    for iter in range(iterations-1, -1, -1):
        mean_sample_length_iter[iter] = sum(mean_sample_length_iter[:iter+1])
        mean_sample_token_cost_iter[iter] = sum(mean_sample_token_cost_iter[:iter+1])
    
    if logger is not None:
        logger.info(json.dumps({
            "summary": True,
            # "mean_acc": float(mean_acc), "n": len(scores),
            # "mean_CoT_length": sum(CoT_lengths) / max(len(CoT_lengths), 1),
            # "mean_direct_score": float(mean_direct_score), "n": len(direct_scores),
        }, ensure_ascii=False))
        

        logger.info(mean_sample_score_iter)
        logger.info(mean_sample_length_iter)
        logger.info(mean_sample_token_cost_iter)



def main(stages=None, seed=None, device_id_student=0):
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


    processor, model = student
    log_path = config["paths"]["logs_dir_evo"] + "_" + config["dataset"]["data_id"].split("/")[-1] + "_evo4"
    file_name = f"ablation_{model_name}_branch_{config['gen_training']['branch']}"
    if config["inference"]["attn"]:
        file_name += "_Attn"


    logger = setup_split_logger(log_path, file_name + "_stage_ablation")
    run_ablation(model, processor, ds, logger=logger)

   


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GEPA inference with optional seed control')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible generation')
    parser.add_argument('--gpu-student', type=int, default=0, help='Student device ID')
    parser.add_argument('--attn', type=int, default=0, help='Attn')
    args = parser.parse_args()
    
    if args.attn:
        config["inference"]["attn"] = True
    else:
        config["inference"]["attn"] = False

    # Now uses config values by default, but can still be overridden
    main(seed=args.seed, device_id_student=args.gpu_student)

