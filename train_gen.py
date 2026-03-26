import argparse
import os
from utils.config_loader import get_config
from dataset.data_loader_evo import load_data
from dataset.data_loader_gen import get_gen_dataloader
# from model.model_loader_gen import GENModel_naive, GENModel_1, GENModel_2, GENModel_3
from model.model_loader_gen import GENModel_1
from model.model_loader_evo import load_model

import json
import random
import numpy as np
import torch
import logging
from pathlib import Path
from torch.optim import AdamW
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pickle
# from prompts.stage_n.prompt_student_textvqa import compose_prompt
from prompts.stage_n.prompt_student_general import compose_prompt
from utils.vqa_soft_acc import composite_reward

from prompts.stage_n.chain import Chain
from prompts.stage_n.chain_1 import Chain_1

from utils.vector_cot import encode_stages, decode_stages
from utils.vqa_soft_acc import combined_accuracy
from prompts.stage_n.search_space import SearchSpace
from utils.vqa_soft_acc import compute_token_cost
# from prompts.stage_n.prompt_student_textvqa_simple import compose_prompt_simple

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


def get_gen_loss(logits, length_logits, cot_final, model, criterion):
    """
    Compute total loss for GEN model:
    - Stage loss: Cross-entropy over reasoning tokens (ignore EOS)
    - Length loss: Global length prediction
    - Tail loss: Encourage EOS after correct length

    Args:
        logits:         [B, L, C]   (C = num_stages + 1, last = EOS)
        length_logits:  [B, model.num_stages]  or  [B, L+1]
        cot_final:      [B, L]      (targets with EOS index where reasoning ends)
        model.num_stages: int       (EOS class index)
        criterion:      nn.CrossEntropyLoss with ignore_index=EOS (recommended)
    """
    EOS = model.num_stages
    B, L, C = logits.shape

    # -----------------------------
    # Stage loss (main CE loss)
    # -----------------------------
    # Ignore EOS tokens if criterion has ignore_index=EOS
    stage_loss = criterion(
        logits.view(-1, C),
        cot_final[:, :model.max_cot_len].reshape(-1)
    )

    # -----------------------------
    # Length loss (predict total reasoning length)
    # -----------------------------
    # true length = number of non-EOS tokens per sample
    len_target = (cot_final != EOS).sum(dim=1).clamp(min=1, max=L)  # [B]
    len_ce = nn.CrossEntropyLoss()
    len_loss = len_ce(length_logits, (len_target - 1).clamp(0, model.num_stages - 1))

    # -----------------------------
    # Tail loss (EOS confidence beyond true length)
    # -----------------------------
    # Build token index grid: [B, L]
    idxs = torch.arange(L, device=cot_final.device).unsqueeze(0).expand(B, L)
    tail_mask = (idxs > len_target.unsqueeze(1))  # mask tail positions only
    

    # Compute log-probability of EOS at each position
    log_p_eos = F.log_softmax(logits, dim=-1)[..., EOS]

    # Average NLL over tail tokens (only if tail_mask not empty)
    if tail_mask.any():
        tail_loss = -(log_p_eos.masked_select(tail_mask)).mean()
    else:
        tail_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # -----------------------------
    # Combine weighted losses
    # -----------------------------
    total_loss = stage_loss + 0.5 * len_loss + 0.5 * tail_loss

    return total_loss, stage_loss, len_loss, tail_loss

def get_gen_acc(logits, length_logits, cot_final):
    B, L, C = logits.shape
    pred = logits.argmax(dim=-1)
    valid_mask = (cot_final != 7)

    correct = (pred == cot_final[:, :L]) & valid_mask[:, :L]
    stage_acc = correct.sum().float() / valid_mask[:, :L].sum().clamp(min=1)

    len_pred = length_logits.argmax(dim=-1)
    len_target = (cot_final != 7).sum(dim=1).clamp(min=1, max=L)
    len_target = (len_target - 1).clamp(0, L - 1)
    len_acc = (len_pred == len_target).float().mean()


    idxs = torch.arange(L, device=cot_final.device).unsqueeze(0).expand(B, L)
    tail_mask = idxs >= len_target.unsqueeze(1)  # tail region (including EOS itself)
    correct_tail = (pred == 7) & tail_mask
    tail_acc = correct_tail.sum().float() / tail_mask.sum().clamp(min=1)
    return stage_acc, len_acc, tail_acc


def log_info(acc, loss, len_loader, epoch, mode = 'Train', gen_logger=None, writer = None):
    epoch_loss, epoch_stage_loss, epoch_len_loss, epoch_tail_penalty = loss
    epoch_stage_acc, epoch_len_acc, epoch_tail_acc = acc

    if gen_logger is not None:
        gen_logger.info(f"Epoch {epoch}: {mode}")
        gen_logger.info(f"  Toal {mode} Loss: {epoch_loss / len_loader}, Stage Loss: {epoch_stage_loss / len_loader}, Length Loss: {epoch_len_loss / len_loader}, Tail Penalty: {epoch_tail_penalty / len_loader}")
        gen_logger.info(f"  {mode} Stage Accuracy: {epoch_stage_acc / len_loader}, {mode} Length Accuracy: {epoch_len_acc / len_loader}, {mode} Tail Accuracy: {epoch_tail_acc / len_loader}")
        gen_logger.info("-" * 100)
    if writer is not None:  
        writer.add_scalar(f"Loss/{mode}", epoch_loss / len_loader, epoch)
        writer.add_scalar(f"Stage Loss/{mode}", epoch_stage_loss / len_loader, epoch)
        writer.add_scalar(f"Length Loss/{mode}", epoch_len_loss / len_loader, epoch)
        writer.add_scalar(f"Tail Loss/{mode}", epoch_tail_penalty / len_loader, epoch)
        writer.add_scalar(f"Stage Acc/{mode}", epoch_stage_acc / len_loader, epoch)  
        writer.add_scalar(f"Length Acc/{mode}", epoch_len_acc / len_loader, epoch)  
        writer.add_scalar(f"Tail Acc/{mode}", epoch_tail_acc / len_loader, epoch)  


def train_gen(model, train_loader, val_loader, optimizer, criterion, gen_logger, config, writer):
    model.train()
    best_stage_acc, best_len_acc, best_tail_acc = 0, 0, 0

    for epoch in range(config["gen_training"]["epochs"]):
        
        epoch_train_loss, epoch_train_stage_loss, epoch_train_len_loss, epoch_train_tail_penalty = 0, 0, 0, 0
        epoch_train_stage_acc, epoch_train_len_acc, epoch_train_tail_acc = 0, 0, 0
        model.train()
        for batch in tqdm(train_loader, desc="Epoch %d Training GEN" % epoch):
            # torch.Size([12, 1369, 3584]) torch.Size([12, 1369]) torch.Size([12, 13, 3584]) torch.Size([12, 13]) torch.Size([12, 7]) torch.Size([12, 7, 8]) torch.Size([12, 7])
            optimizer.zero_grad()
            logits, length_logits = model(
                batch['image_emb'].to(model.device),
                batch['image_mask'].to(model.device), 
                batch['question_emb'].to(model.device), 
                batch['question_mask'].to(model.device),
                batch['cot_initial'].to(model.device), 
                batch['A'].to(model.device)
            )
            cot_final = batch['cot_final'].to(model.device).long()
            # cot_final[cot_final == -1] = criterion.ignore_index
            loss, stage_loss, len_loss, tail_penalty = get_gen_loss(logits, length_logits, cot_final, model, criterion)
            # loss, stage_loss, len_loss = get_gen_loss(logits, length_logits, cot_final, model, criterion)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_stage_loss += stage_loss.item()
            epoch_train_len_loss += len_loss.item()
            epoch_train_tail_penalty += tail_penalty.item()
            
            stage_acc, len_acc, tail_acc = get_gen_acc(logits, length_logits, cot_final)

            epoch_train_stage_acc += stage_acc.item()
            epoch_train_len_acc += len_acc.item()
            epoch_train_tail_acc += tail_acc.item()
        acc = (epoch_train_stage_acc, epoch_train_len_acc, epoch_train_tail_acc)
        loss = (epoch_train_loss, epoch_train_stage_loss, epoch_train_len_loss, epoch_train_tail_penalty)
        log_info(acc, loss, len(train_loader), epoch, "Train", gen_logger, writer)

        if epoch % 5 == 0:
            epoch_val_loss, epoch_val_stage_loss, epoch_val_len_loss, epoch_val_tail_penalty = 0, 0, 0, 0
            epoch_val_stage_acc, epoch_val_len_acc, epoch_val_tail_acc = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Epoch %d Validation GEN" % epoch):
                    
                    logits, length_logits = model(
                        batch['image_emb'].to(model.device),
                        batch['image_mask'].to(model.device), 
                        batch['question_emb'].to(model.device), 
                        batch['question_mask'].to(model.device),
                        batch['cot_initial'].to(model.device), 
                        batch['A'].to(model.device)
                    )
                    cot_final = batch['cot_final'].to(model.device).long()
                    # cot_final[cot_final == -1] = criterion.ignore_index
                    loss, stage_loss, len_loss, tail_penalty = get_gen_loss(logits, length_logits, cot_final, model, criterion)
                    # loss, stage_loss, len_loss = get_gen_loss(logits, length_logits, cot_final, model, criterion)
                    epoch_val_loss += loss.item()
                    epoch_val_stage_loss += stage_loss.item()
                    epoch_val_len_loss += len_loss.item()
                    epoch_val_tail_penalty += tail_penalty.item()
                    
                    stage_acc, len_acc, tail_acc = get_gen_acc(logits, length_logits, cot_final)
                    epoch_val_stage_acc += stage_acc.item()
                    epoch_val_len_acc += len_acc.item()
                    epoch_val_tail_acc += tail_acc.item()

                    torch.save(model.state_dict(), config["paths"]["logs_dir_gen"] + "/epoch_%d_model.pt" % epoch)
            acc = (epoch_val_stage_acc, epoch_val_len_acc, epoch_val_tail_acc)
            loss = (epoch_val_loss, epoch_val_stage_loss, epoch_val_len_loss, epoch_val_tail_penalty)
            log_info(acc, loss, len(val_loader), epoch, "Val", gen_logger, writer)

            if best_stage_acc < epoch_val_stage_acc / len(val_loader) and best_len_acc < epoch_val_len_acc / len(val_loader) and best_tail_acc < epoch_val_tail_acc / len(val_loader):
                best_epoch = epoch
                best_stage_acc = epoch_val_stage_acc / len(val_loader)
                best_len_acc = epoch_val_len_acc / len(val_loader)
                best_tail_acc = epoch_val_tail_acc / len(val_loader)
        
    
    # gen_logger.info(f"Best Epoch: {best_epoch}, Best Stage Accuracy: {best_stage_acc}, Best Length Accuracy: {best_len_acc}, Best Tail Accuracy: {best_tail_acc}")
    
    
def get_gen_input(out, L_max, stages, stage_pool):
    A = torch.tensor(out["A"])
    a_to_final = torch.tensor(out["a_to_final"])
    if A.shape[0] < L_max:
        # Extend A matrix to match final CoT length
        padding_size = L_max - A.shape[0]
        A = torch.cat([A, torch.zeros(padding_size, A.shape[1])], dim=0)
        A = torch.cat([A, torch.zeros(A.shape[0], padding_size)], dim=1)

        a_to_final = torch.cat([a_to_final, torch.zeros(padding_size)])
            
    A = torch.cat((A, a_to_final.unsqueeze(1)), dim=1).unsqueeze(0)  
    
    
    cot_initial, _ = encode_stages(stages, stage_pool)
    # cot_initial = torch.tensor(cot_initial).unsqueeze(0)
    cot_initial = torch.as_tensor(cot_initial).unsqueeze(0)
    
    image_emb = torch.tensor(out["image_emb"][0])
    
    image_mask = torch.ones_like(image_emb[:, :, 0]).float()
    question_emb = torch.tensor(out["question_emb"])
    question_mask = torch.ones_like(question_emb[:, :, 0]).float()
    
    return image_emb, image_mask, question_emb, question_mask, cot_initial, A


def test_time_metric(sample_stages, method="Majority Voting"):
    if method == "Majority Voting":
        # gen_count_preds, gen_score = {}, {}
        # cot_count_preds, cot_score = {}, {}
        report_dict = {}
        
        count_preds = {}
        pred_score = {}
        
        record_scores = {}
        
        
        for init_cot_str, init_cot_dict in sample_stages.items():
            report_dict[init_cot_str] = {}
            for iter in range(len(init_cot_dict)-1):
                if iter not in count_preds:
                    count_preds[iter] = {}
                    record_scores[iter] = {}
                
                iter_pred = init_cot_dict[iter]['pred']
                if iter_pred not in pred_score:
                    pred_score[iter_pred] = init_cot_dict[iter]['score']
                    
                count_preds[iter][iter_pred] = count_preds[iter].get(iter_pred, 0) + 1
                    
                report_dict[init_cot_str][iter] = {
                    "stages": init_cot_dict[iter]['stages'],
                    "num_student_calls": len(init_cot_dict[iter]['stages']),
                    "score": init_cot_dict[iter]['score'],
                    "pred": init_cot_dict[iter]['pred'],
                }
                
        for iter in record_scores.keys():
            majority_pred = max(count_preds[iter], key=count_preds[iter].get)
            record_scores[iter] = pred_score[majority_pred]
            
        max_score_list = [record_scores[0]]
        max_score = record_scores[0]
        for iter in record_scores.keys():
            if iter != 0:
                max_score = max(max_score, record_scores[iter])
                max_score_list.append(max_score)
            
        return max_score_list, max_score, max_score_list[0], report_dict
                # if init_cot_dict[iter]['pred'] not in gen_count_preds:
                #     gen_count_preds[init_cot_dict[iter]['pred']] = 0
                #     gen_score[init_cot_dict[iter]['pred']] = 0
                # gen_count_preds[init_cot_dict[iter]['pred']] += 1
                # gen_score[init_cot_dict[iter]['pred']] = init_cot_dict[iter]['score']
                
        # for init_cot_str, init_cot_dict in sample_stages.items():
            
        #     if init_cot_dict[0]['pred'] not in cot_count_preds:
        #         cot_count_preds[init_cot_dict[0]['pred']] = 0
        #         cot_score[init_cot_dict[0]['pred']] = 0
        #     cot_count_preds[init_cot_dict[0]['pred']] += 1
        #     cot_score[init_cot_dict[0]['pred']] = init_cot_dict[0]['score']
            
            
        #     report_dict[init_cot_str] = {
        #         "stages": init_cot_dict[1]['stages'],
        #         "score": init_cot_dict[1]['score'],
        #         "pred": init_cot_dict[1]['pred'],
        #     }
  
        #     if report_dict[init_cot_str]['pred'] not in gen_count_preds:
        #         gen_count_preds[report_dict[init_cot_str]['pred']] = 0
        #         gen_score[report_dict[init_cot_str]['pred']] = 0
        #     gen_count_preds[report_dict[init_cot_str]['pred']] += 1
        #     gen_score[report_dict[init_cot_str]['pred']] = report_dict[init_cot_str]['score']
            
        
        # return gen_score[max(gen_count_preds, key=gen_count_preds.get)], cot_score[max(cot_count_preds, key=cot_count_preds.get)], report_dict


def test_gen_iteration(model, test_loader, student, test_ds, evo_ds, test_ids, criterion, gen_logger, config, ref_key=None):
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]
    model.eval()

    _, stage_pool = encode_stages(config["inference"]["stages_pool"])
    
    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")
    processor, student = student
    
    total_iterations = config["gen_training"]["test_iterations"]
    score_list = [0 for _ in range(total_iterations)]
    num_student_calls_list = [0 for _ in range(total_iterations)]
    token_cost_list = [0 for _ in range(total_iterations)]
    L_max = len(stage_pool)
    
    scores, cot_scores, direct_answer_scores = [], [], []
    with torch.no_grad():
        epoch_test_loss, epoch_test_stage_loss, epoch_test_len_loss, epoch_test_tail_penalty = 0, 0, 0, 0
        epoch_test_stage_acc, epoch_test_len_acc, epoch_test_tail_acc = 0, 0, 0
        
        with tqdm(enumerate(test_loader), desc="Testing GEN", total=len(test_loader)) as outer_tqdm:
            for i, batch in outer_tqdm:
                try:
                    idx = batch['idx'][0]
                    cot_initial = batch['cot_initial'][0]
                    cot_final = batch['cot_final'][0]
                    list_cot_gt_decoded = []
                    for cot_decoded in cot_final:
                        list_cot_gt_decoded.append(decode_stages(cot_decoded, stage_pool))
                        
                    sample = evo_ds[test_ids[idx]].copy()
                    if 'options' in config['dataset']:
                        if "vision" in config['dataset']['options'][0] or "4" in config['dataset']['options'][0]:
                            dataset_name = config['dataset']['data_id'].split('/')[-1]+ "_" + config['dataset']['options'][0] 
                        elif "10" in config['dataset']['options'][0]:
                            dataset_name = config['dataset']['data_id'].split('/')[-1]
                    elif config["dataset"]["vison_only"] and config["dataset"]["data_id"] == "AI4Math/MathVerse":
                        dataset_name = config['dataset']['data_id'].split('/')[-1] + "_vision"
                    else:
                        dataset_name = config['dataset']['data_id'].split('/')[-1]
                        
                    search_space_path = config["paths"]["search_space_dir"] + f"/{dataset_name}/search_space_{sample['id']}.pkl"
                    if os.path.exists(search_space_path):
                        with open(search_space_path, "rb") as f:
                            gen_logger.info(f"Loading search space from {search_space_path}")
                            search_space = pickle.load(f)
                    else:
                        search_space = SearchSpace()
                        
                    stage_output_path = config["paths"]["stage_outputs_dir"] + f"/{dataset_name}/stage_outputs_{sample['id']}.pkl"
                    os.makedirs(config["paths"]["stage_outputs_dir"] + f"/{dataset_name}", exist_ok=True)
                    if os.path.exists(stage_output_path):
                        with open(stage_output_path, "rb") as f:
                            gen_logger.info(f"Loading stage outputs from {stage_output_path}")
                            stage_outputs = pickle.load(f)
                    else:
                        stage_outputs = {}
                    
                    answer = sample.get('answer')
                    
                    sample_stages = {}
                    for cot in cot_initial:
                        stages = decode_stages(cot, stage_pool)
                        init_cot_str = ",".join(stages)
                        sample_stages[init_cot_str] = {}
                        sample_stages[init_cot_str][0] = {
                            "stages": stages,
                            "score": 0,
                            "pred": "",
                        }
                    
                        with tqdm(total=total_iterations, desc="Iteration", leave=False) as inner_tqdm:
                            for iter in range(total_iterations):
                                stages = sample_stages[init_cot_str][iter]['stages']
                                chain = Chain_1(stages, gen_logger, student, processor, config, search_space=search_space, base_system_prompts=base_system_prompts)

                                
                                if ",".join(stages) in stage_outputs:
                                    out = stage_outputs[",".join(stages)]
                                    # out = chain.run(idx, sample, return_debug=True)
                                    
                                    # stage_outputs[",".join(stages)] = out
                                    
                                else:
                                    # out = stage_outputs[",".join(stages)]
                                    out = chain.run(idx, sample, return_debug=True)
                                    
                                    stage_outputs[",".join(stages)] = out
                                    
                                search_space = chain.search_space
                                _, score, direct_score = composite_reward(out, answer, stages=stages)
                
                                if config["gen_training"]["token_cost"]:
                                    token_cost = compute_token_cost(out, processor)
                                    sample_stages[init_cot_str][iter]['token_cost'] = token_cost
                                sample_stages[init_cot_str][iter]['score'] = score
                                sample_stages[init_cot_str][iter]['pred'] = out["rendered_answer"]
                                
        
                                if len(stages) > 0:
                                    image_emb, image_mask, question_emb, question_mask, cot_initial, A = get_gen_input(out, L_max, stages, stage_pool)

                                    logits, length_logits = model(
                                        image_emb.to(model.device),
                                        image_mask.to(model.device), 
                                        question_emb.to(model.device), 
                                        question_mask.to(model.device),
                                        cot_initial.to(model.device), 
                                        A.to(model.device)
                                    )
                                    
                                    cot_pred = decode_stages(logits.argmax(dim=-1)[0], stage_pool, length_logits.argmax(dim=-1)[0].item()+1)
                                    sample_stages[init_cot_str][iter + 1] = {
                                        "stages": cot_pred,
                                        "score": 0,
                                        "pred": "",
                                    }
                                else:
                                    sample_stages[init_cot_str][iter + 1] = {
                                        "stages": stages,
                                        "score": 0,
                                        "pred": "",
                                    }
                                

                                inner_tqdm.update(1)
                                
                    max_score_list, max_gen_score, cot_score, report_dict = test_time_metric(sample_stages)
                    

                    
                    for iter in range(total_iterations):
                        score_list[iter] += max_score_list[iter]

                        for init_cot_str in sample_stages.keys():
                            num_student_calls_list[iter] += len(sample_stages[init_cot_str][iter]['stages']) + 1

                            if config["gen_training"]["token_cost"]:
                                token_cost_list[iter] += sample_stages[init_cot_str][iter]['token_cost']

                        
                    
                    
                    record = {
                        "idx": idx,
                        "question": sample.get("question"),
                        "score": max_gen_score,
                        "cot_score": cot_score,
                        "direct_answer_score": float(direct_score),
                        "direct_answer_pred": out["direct_answer_rendered"],
                        "iterations": sample_stages,
                    }
                    gen_logger.info(json.dumps(record, ensure_ascii=False, indent=4))
                    
                    scores.append(max_gen_score)
                    cot_scores.append(cot_score)
                    direct_answer_scores.append(direct_score)
                    # num_student_calls += sum([len(report_dict[init_cot_str][iter]['stages']) for init_cot_str in report_dict.keys() for iter in report_dict[init_cot_str].keys()])
    
                    # Update search space:
                    with open(search_space_path, "wb") as f:
                        pickle.dump(search_space, f)
                        
                    
                    with open(stage_output_path, "wb") as f:
                        pickle.dump(stage_outputs, f)


                except torch.cuda.OutOfMemoryError as e:
                    gen_logger.info(f"CUDA OOM error on sample {i} (idx: {idx}): {e}")
                    gen_logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        gen_logger.info(f"Memory error on sample {i} (idx: {idx}): {e}")
                        gen_logger.info("Clearing CUDA cache and skipping this sample...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        # acc = (epoch_test_stage_acc/len(test_loader), epoch_test_len_acc/len(test_loader), epoch_test_tail_acc/len(test_loader))
        # loss = (epoch_test_loss/len(test_loader), epoch_test_stage_loss/len(test_loader), epoch_test_len_loss/len(test_loader), epoch_test_tail_penalty/len(test_loader))
        # log_info(acc, loss, len(test_loader), 0, "Test", gen_logger)
        
        mean_gen_acc = sum(scores) / max(len(scores), 1)
        mean_cot_acc = sum(cot_scores) / max(len(cot_scores), 1)
        mean_direct_answer_acc = sum(direct_answer_scores) / max(len(direct_answer_scores), 1)
        mean_score_list = [0 for _ in range(total_iterations)]
        mean_num_student_calls_list = [0 for _ in range(total_iterations)]
        mean_token_cost_list = [0 for _ in range(total_iterations)]
        for iter in range(total_iterations):
            mean_score_list[iter] = score_list[iter] / max(len(cot_scores), 1)
            mean_num_student_calls_list[iter] = num_student_calls_list[iter] / max(len(cot_scores), 1)

            if config["gen_training"]["token_cost"]:
                mean_token_cost_list[iter] = token_cost_list[iter] / max(len(cot_scores), 1)


        gen_logger.info(json.dumps({
            "split": "Test", "summary": True,
            "mean_gen_acc": float(mean_gen_acc), "n": len(scores),
            "mean_cot_acc": float(mean_cot_acc), "n": len(cot_scores),
            "mean_direct_answer_acc": float(mean_direct_answer_acc),
            "mean_num_student_calls": mean_num_student_calls_list,
            "mean_score_list": mean_score_list, "iterations": total_iterations,
            "mean_token_cost_list": mean_token_cost_list,
        }, ensure_ascii=False))

        for iter in range(total_iterations-1, -1, -1):
            mean_num_student_calls_list[iter] = sum(mean_num_student_calls_list[:iter+1])
            mean_token_cost_list[iter] = sum(mean_token_cost_list[:iter+1])


        gen_logger.info(mean_num_student_calls_list)
        gen_logger.info(mean_token_cost_list)


def test_gen_iteration_multi_model(model, test_loader, student, test_model, test_ds, evo_ds, test_ids, criterion, gen_logger, config, ref_key=None):
    if ref_key is None:
        ref_key = config["evaluation"]["ref_key"]
    model.eval()

    _, stage_pool = encode_stages(config["inference"]["stages_pool"])
    
    base_system_prompts = {stage: compose_prompt(stage) for stage in config["inference"]["stages_pool"]}
    base_system_prompts["ANSWER.CONSOLIDATION"] = compose_prompt("ANSWER.CONSOLIDATION")
    base_system_prompts["DIRECT_ANSWER"] = compose_prompt("DIRECT_ANSWER")
    processor, student = student

    processor, student = test_model
    
    
    total_iterations = config["gen_training"]["test_iterations"]
    score_list = [0 for _ in range(total_iterations)]
    num_student_calls_list = [0 for _ in range(total_iterations)]
    token_cost_list = [0 for _ in range(total_iterations)]
    L_max = len(stage_pool)
    
    scores, cot_scores, direct_answer_scores = [], [], []
    with torch.no_grad():
        epoch_test_loss, epoch_test_stage_loss, epoch_test_len_loss, epoch_test_tail_penalty = 0, 0, 0, 0
        epoch_test_stage_acc, epoch_test_len_acc, epoch_test_tail_acc = 0, 0, 0
        
        with tqdm(enumerate(test_loader), desc="Testing GEN", total=len(test_loader)) as outer_tqdm:
            for i, batch in outer_tqdm:
                try:
                    idx = batch['idx'][0]
                    cot_initial = batch['cot_initial'][0]
                    cot_final = batch['cot_final'][0]
                    list_cot_gt_decoded = []
                    for cot_decoded in cot_final:
                        list_cot_gt_decoded.append(decode_stages(cot_decoded, stage_pool))
                        
                    sample = evo_ds[test_ids[idx]].copy()
                    if 'options' in config['dataset']:
                        if "vision" in config['dataset']['options'][0] or "4" in config['dataset']['options'][0]:
                            dataset_name = config['dataset']['data_id'].split('/')[-1]+ "_" + config['dataset']['options'][0] 
                        elif "10" in config['dataset']['options'][0]:
                            dataset_name = config['dataset']['data_id'].split('/')[-1]
                    elif config["dataset"]["vison_only"] and config["dataset"]["data_id"] == "AI4Math/MathVerse":
                        dataset_name = config['dataset']['data_id'].split('/')[-1] + "_vision"
                    else:
                        dataset_name = config['dataset']['data_id'].split('/')[-1]
                        
                    search_space_path = config["paths"]["search_space_dir"] + f"/{dataset_name}/search_space_{sample['id']}.pkl"
                    if os.path.exists(search_space_path):
                        with open(search_space_path, "rb") as f:
                            gen_logger.info(f"Loading search space from {search_space_path}")
                            search_space = pickle.load(f)
                    else:
                        search_space = SearchSpace()
                        
                    stage_output_path = config["paths"]["stage_outputs_dir"] + f"/{dataset_name}/stage_outputs_{sample['id']}.pkl"
                    os.makedirs(config["paths"]["stage_outputs_dir"] + f"/{dataset_name}", exist_ok=True)
                    if os.path.exists(stage_output_path):
                        with open(stage_output_path, "rb") as f:
                            gen_logger.info(f"Loading stage outputs from {stage_output_path}")
                            stage_outputs = pickle.load(f)
                    else:
                        stage_outputs = {}
                    
                    answer = sample.get('answer')
                    
                    sample_stages = {}
                    
                    test_search_space = SearchSpace()
                    test_stage_outputs = {}
                    # test_search_space = search_space
                    # test_stage_outputs = stage_outputs
                    for cot in cot_initial:
                        stages = decode_stages(cot, stage_pool)
                        init_cot_str = ",".join(stages)
                        sample_stages[init_cot_str] = {}
                        sample_stages[init_cot_str][0] = {
                            "stages": stages,
                            "score": 0,
                            "pred": "",
                        }
                        
                        with tqdm(total=total_iterations, desc="Iteration", leave=False) as inner_tqdm:
                            for iter in range(total_iterations):
                                stages = sample_stages[init_cot_str][iter]['stages']
                                chain = Chain_1(stages, gen_logger, student, processor, config, search_space=search_space, base_system_prompts=base_system_prompts)

                                
                                if ",".join(stages) in stage_outputs:
                                    out = stage_outputs[",".join(stages)]
                                    # out = chain.run(idx, sample, return_debug=True)
                                    
                                    # stage_outputs[",".join(stages)] = out
                                    
                                else:
                                    # out = stage_outputs[",".join(stages)]
                                    out = chain.run(idx, sample, return_debug=True)
                                    
                                    stage_outputs[",".join(stages)] = out
                                    
                                search_space = chain.search_space
                                # _, score, direct_score = composite_reward(out, answer, stages=stages)
                                
                                
                                if ",".join(stages) in test_stage_outputs:
                                    test_out = test_stage_outputs[",".join(stages)]
                                else:
                                    test_chain = Chain_1(stages, gen_logger, student, processor, config, search_space=test_search_space, base_system_prompts=base_system_prompts)
                                    test_out = test_chain.run(idx, sample, return_debug=True)
                                    test_stage_outputs[",".join(stages)] = test_out
                                    
                                _, score, direct_score = composite_reward(test_out, answer, stages=stages)
                
                                if config["gen_training"]["token_cost"]:
                                    token_cost = compute_token_cost(out, processor)
                                    sample_stages[init_cot_str][iter]['token_cost'] = token_cost
                                sample_stages[init_cot_str][iter]['score'] = score
                                sample_stages[init_cot_str][iter]['pred'] = test_out["rendered_answer"]
                                
        
                                if len(stages) > 0:
                                    image_emb, image_mask, question_emb, question_mask, cot_initial, A = get_gen_input(out, L_max, stages, stage_pool)

                                    logits, length_logits = model(
                                        image_emb.to(model.device),
                                        image_mask.to(model.device), 
                                        question_emb.to(model.device), 
                                        question_mask.to(model.device),
                                        cot_initial.to(model.device), 
                                        A.to(model.device)
                                    )
                                    
                                    cot_pred = decode_stages(logits.argmax(dim=-1)[0], stage_pool, length_logits.argmax(dim=-1)[0].item()+1)
                                    sample_stages[init_cot_str][iter + 1] = {
                                        "stages": cot_pred,
                                        "score": 0,
                                        "pred": "",
                                    }
                                else:
                                    sample_stages[init_cot_str][iter + 1] = {
                                        "stages": stages,
                                        "score": 0,
                                        "pred": "",
                                    }
                                

                                inner_tqdm.update(1)
                                
                    max_score_list, max_gen_score, cot_score, report_dict = test_time_metric(sample_stages)
                    

                    
                    for iter in range(total_iterations):
                        score_list[iter] += max_score_list[iter]

                        for init_cot_str in sample_stages.keys():
                            num_student_calls_list[iter] += len(sample_stages[init_cot_str][iter]['stages']) + 1

                            if config["gen_training"]["token_cost"]:
                                token_cost_list[iter] += sample_stages[init_cot_str][iter]['token_cost']

                        
                    
                    
                    record = {
                        "idx": idx,
                        "question": sample.get("question"),
                        "score": max_gen_score,
                        "cot_score": cot_score,
                        "direct_answer_score": float(direct_score),
                        "direct_answer_pred": out["direct_answer_rendered"],
                        "iterations": sample_stages,
                    }
                    gen_logger.info(json.dumps(record, ensure_ascii=False, indent=4))
                    
                    scores.append(max_gen_score)
                    cot_scores.append(cot_score)
                    direct_answer_scores.append(direct_score)
                    # num_student_calls += sum([len(report_dict[init_cot_str][iter]['stages']) for init_cot_str in report_dict.keys() for iter in report_dict[init_cot_str].keys()])
    
                    # Update search space:
                    with open(search_space_path, "wb") as f:
                        pickle.dump(search_space, f)
                        
                    
                    with open(stage_output_path, "wb") as f:
                        pickle.dump(stage_outputs, f)


                except torch.cuda.OutOfMemoryError as e:
                    gen_logger.info(f"CUDA OOM error on sample {i} (idx: {idx}): {e}")
                    gen_logger.info("Clearing CUDA cache and skipping this sample...")
                    torch.cuda.empty_cache()
                    continue

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        gen_logger.info(f"Memory error on sample {i} (idx: {idx}): {e}")
                        gen_logger.info("Clearing CUDA cache and skipping this sample...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        # acc = (epoch_test_stage_acc/len(test_loader), epoch_test_len_acc/len(test_loader), epoch_test_tail_acc/len(test_loader))
        # loss = (epoch_test_loss/len(test_loader), epoch_test_stage_loss/len(test_loader), epoch_test_len_loss/len(test_loader), epoch_test_tail_penalty/len(test_loader))
        # log_info(acc, loss, len(test_loader), 0, "Test", gen_logger)
        
        mean_gen_acc = sum(scores) / max(len(scores), 1)
        mean_cot_acc = sum(cot_scores) / max(len(cot_scores), 1)
        mean_direct_answer_acc = sum(direct_answer_scores) / max(len(direct_answer_scores), 1)
        mean_score_list = [0 for _ in range(total_iterations)]
        mean_num_student_calls_list = [0 for _ in range(total_iterations)]
        mean_token_cost_list = [0 for _ in range(total_iterations)]
        for iter in range(total_iterations):
            mean_score_list[iter] = score_list[iter] / max(len(cot_scores), 1)
            mean_num_student_calls_list[iter] = num_student_calls_list[iter] / max(len(cot_scores), 1)

            if config["gen_training"]["token_cost"]:
                mean_token_cost_list[iter] = token_cost_list[iter] / max(len(cot_scores), 1)


        gen_logger.info(json.dumps({
            "split": "Test", "summary": True,
            "mean_gen_acc": float(mean_gen_acc), "n": len(scores),
            "mean_cot_acc": float(mean_cot_acc), "n": len(cot_scores),
            "mean_direct_answer_acc": float(mean_direct_answer_acc),
            "mean_num_student_calls": mean_num_student_calls_list,
            "mean_score_list": mean_score_list, "iterations": total_iterations,
            "mean_token_cost_list": mean_token_cost_list,
        }, ensure_ascii=False))

        for iter in range(total_iterations-1, -1, -1):
            mean_num_student_calls_list[iter] = sum(mean_num_student_calls_list[:iter+1])
            mean_token_cost_list[iter] = sum(mean_token_cost_list[:iter+1])


        gen_logger.info(mean_num_student_calls_list)
        gen_logger.info(mean_token_cost_list)


def main(args):
    set_random_seed(args.seed)
    if args.test:
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_full_test" if config["gen_training"]["full_data"] else config["paths"]["logs_dir_gen"] + "_test"
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_setup" + str(config["gen_training"]["setup"])
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_iterations" + str(config["gen_training"]["test_iterations"])
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_attn" if config["gen_training"]["attn"] else config["paths"]["logs_dir_gen"]
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_test"
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_attn" if config["gen_training"]["attn"] else config["paths"]["logs_dir_gen"]
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_i_q" if config["gen_training"]["i_q"] else config["paths"]["logs_dir_gen"]
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_2"
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_iterations" + str(config["gen_training"]["test_iterations"])

        if config["gen_training"]["token_cost"]:
            config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_token_cost"
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_3"
        evo_ds = load_data(config["dataset"]["data_id"], config["dataset"]["local_data_dir"], config, args=args)
    else:
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_full_setup" + str(config["gen_training"]["setup"]) if config["gen_training"]["full_data"] else config["paths"]["logs_dir_gen"] + "_setup" + str(config["gen_training"]["setup"])
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_attn" if config["gen_training"]["attn"] else config["paths"]["logs_dir_gen"]
        for data_id in config["training_data"]["data_id"]:
            config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_" + data_id.split("/")[-1]
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_" + config["tradataset"]["data_id"].split("/")[-1]
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_attn" if config["gen_training"]["attn"] else config["paths"]["logs_dir_gen"]
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_i_q" if config["gen_training"]["i_q"] else config["paths"]["logs_dir_gen"]
        config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_2"
        if args.multi_model:
            config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_multi_model"
        # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_3"
    
    # config["paths"]["logs_dir_gen"] = config["paths"]["logs_dir_gen"] + "_try" 


        evo_ds = []
        for data_id, local_data_dir in zip(config["training_data"]["data_id"], config["training_data"]["local_data_dir"]):
            ds = load_data(data_id, local_data_dir, config, args=args)
            evo_ds.append(ds)
        
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    # if config["gen_training"]["setup"] == 2 or config["gen_training"]["setup"] == 3:
    #     model = GENModel_naive(config=config, device=device).to(device)
    # elif config["gen_training"]["setup"] == 4:
    #     model = GENModel_1(config=config, device=device).to(device)
    # elif config["gen_training"]["setup"] == 5:
    #     model = GENModel_2(config=config, device=device).to(device)
    # elif config["gen_training"]["setup"] == 6:
    #     model = GENModel_3(config=config, device=device).to(device)

        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=7)

    # gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_MMStar")
    # gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_Mathvision")
    os.makedirs(config["paths"]["logs_dir_gen"] + "/initial_test", exist_ok=True)

    if not args.test:
        gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}")
        train_loader, val_loader, stage_pool = get_gen_dataloader(evo_ds, config, logger=gen_logger)
        if config["gen_training"]["setup"] == 1:
            # 3
            model = GENModel_1(emb_dim=train_loader.dataset[0][2].shape[1], 
                                num_stages=len(stage_pool), 
                                max_cot_len=8, device=device, config=config).to(device)
            # 2
            # model = GENModel_1(emb_dim=train_loader.dataset[0][2].shape[1], 
            #                     num_stages=len(stage_pool), 
            #                     max_cot_len=14, device=device, config=config).to(device)
                                
        # model.load_state_dict(torch.load(config["paths"]["logs_dir_gen"] + "/epoch_5_best_model.pt"))
        optimizer = AdamW(model.parameters(), lr=float(config["gen_training"]["lr"]), weight_decay=float(config["gen_training"]["weight_decay"]))

        writer = SummaryWriter(config["paths"]["logs_dir_gen"] + "/initial_test")
        train_gen(model, train_loader, val_loader, optimizer, criterion, gen_logger, config, writer)
        
        
    # test_gen(model, test_loader, optimizer, criterion, gen_logger, config, writer)
    else:
        if not args.multi_model:

            if config["dataset"]["data_id"] == "MMMU/MMMU_Pro":
                dataset_name = config["dataset"]["data_id"].split("/")[-1] + "_" + config["dataset"]["options"][0]
            else:
                dataset_name = config["dataset"]["data_id"].split("/")[-1]
                
            if config["gen_training"]["test_model"] == "MathVision":
                gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_Mathvision_branch{config['gen_training']['branch']}_{dataset_name}")
            elif config["gen_training"]["test_model"] == "MMStar":
                gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_MMStar_branch{config['gen_training']['branch']}_{dataset_name}")
            elif config["gen_training"]["test_model"] == "composite":
                gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_composite_branch{config['gen_training']['branch']}_{dataset_name}")
            else:
                raise ValueError(f"Invalid test model: {config['gen_training']['test_model']}")
            

            test_loader, test_ds, test_ids, stage_pool = get_gen_dataloader(evo_ds, config, gen_logger, test=True)
            if config["gen_training"]["setup"] == 1:
                model = GENModel_1(2560, num_stages=len(stage_pool), 
                                    max_cot_len=8, device=device, config=config).to(device)
                
            # model.load_state_dict(torch.load("logs/log_gen_MMMU_MathVerse_attn_2/epoch_100_model.pt"))
            if config["gen_training"]["test_model"] == "MathVision":
                model.load_state_dict(torch.load("logs/log_gen_MathVision_attn_2/epoch_99_model.pt", map_location=device))
            elif config["gen_training"]["test_model"] == "MMStar":
                model.load_state_dict(torch.load("logs/log_gen_MMStar_attn_2/epoch_99_model.pt", map_location=device))
            elif config["gen_training"]["test_model"] == "composite":
                # model.load_state_dict(torch.load("logs/log_gen_MMMU_MathVerse_attn_2/epoch_100_model.pt", map_location=device))
                model.load_state_dict(torch.load("logs/log_gen_MMMU_MathVerse_i_q_2/epoch_60_model.pt", map_location=device))
            else:
                raise ValueError(f"Invalid test model: {config['gen_training']['test_model']}")
            
            student = load_model(config["model"]["model_id_student"], device=args.gpu, API=False, 
                                        local_dir=config["model"]["local_model_dir_student"])
                                    

            # test_gen(model, test_loader, student, test_ds, test_ids, criterion, gen_logger, config)
            test_gen_iteration(model, test_loader, student, test_ds, evo_ds, test_ids, criterion, gen_logger, config)


        else:
            if config["dataset"]["data_id"] == "MMMU/MMMU_Pro":
                dataset_name = config["dataset"]["data_id"].split("/")[-1] + "_" + config["dataset"]["options"][0]
            else:
                dataset_name = config["dataset"]["data_id"].split("/")[-1]

            test_model_name = config["model"]["test_model_id_student"].split("/")[-1]
                
            if config["gen_training"]["test_model"] == "MathVision":
                gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_Mathvision_branch{config['gen_training']['branch']}_{dataset_name}_{test_model_name}")
            elif config["gen_training"]["test_model"] == "MMStar":
                gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_MMStar_branch{config['gen_training']['branch']}_{dataset_name}_{test_model_name}")
            elif config["gen_training"]["test_model"] == "composite":
                gen_logger = setup_split_logger(config["paths"]["logs_dir_gen"], f"gen_seed_{args.seed}_composite_branch{config['gen_training']['branch']}_{dataset_name}_{test_model_name}")
            else:
                raise ValueError(f"Invalid test model: {config['gen_training']['test_model']}")
            

            test_loader, test_ds, test_ids, stage_pool = get_gen_dataloader(evo_ds, config, gen_logger, test=True)
            if config["gen_training"]["setup"] == 1:
                model = GENModel_1(2560, num_stages=len(stage_pool), 
                                    max_cot_len=8, device=device, config=config).to(device)
                
            # model.load_state_dict(torch.load("logs/log_gen_MMMU_MathVerse_attn_2/epoch_100_model.pt"))
            if config["gen_training"]["test_model"] == "MathVision":
                model.load_state_dict(torch.load("logs/log_gen_MathVision_attn_2/epoch_99_model.pt", map_location=device))
            elif config["gen_training"]["test_model"] == "MMStar":
                model.load_state_dict(torch.load("logs/log_gen_MMStar_attn_2/epoch_99_model.pt", map_location=device))
            elif config["gen_training"]["test_model"] == "composite":
                # model.load_state_dict(torch.load("logs/log_gen_MMMU_MathVerse_attn_2/epoch_100_model.pt", map_location=device))
                model.load_state_dict(torch.load("logs/log_gen_MMMU_MathVerse_i_q_2/epoch_60_model.pt", map_location=device))
            else:
                raise ValueError(f"Invalid test model: {config['gen_training']['test_model']}")
            
            student = load_model(config["model"]["model_id_student"], device=args.gpu, API=False, 
                                        local_dir=config["model"]["local_model_dir_student"])

            test_model = load_model(config["model"]["test_model_id_student"], device=args.gpu, API=False, 
                                        local_dir=config["model"]["test_local_model_dir_student"])
                                    

            # test_gen(model, test_loader, student, test_ds, test_ids, criterion, gen_logger, config)
            test_gen_iteration_multi_model(model, test_loader, student, test_model, test_ds, evo_ds, test_ids, criterion, gen_logger, config)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training GEN model with seed control')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible generation')
    parser.add_argument('--gpu', type=int, default=1, help='device ID')
    parser.add_argument('--test', action='store_true', help='Test GEN model')
    parser.add_argument('--multi-model', type=int, default=0, help='Test on other student models')
    args = parser.parse_args()
    main(args)