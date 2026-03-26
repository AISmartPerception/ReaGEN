import os
import json
import sys
import random
from pathlib import Path

def extract_idx_score(file_path, output_file=None):
    """
    Extract idx and score from a JSONL file with multi-line JSON objects.
    
    Args:
        file_path (str): Path to the JSONL file
        output_file (str, optional): Path to save the results. If None, prints to stdout.
    
    Returns:
        list: List of tuples (idx, score)
    """
    results = []
    ids = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_idx = None
    current_score = None
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Look for idx line
        if '"idx":' in line:
            try:
                # Extract idx value
                idx_part = line.split('"idx":')[1].split(',')[0].strip()
                current_idx = int(idx_part)
            except (ValueError, IndexError):
                continue
                
        # Look for score line
        elif '"score":' in line:
            try:
                # Extract score value
                score_part = line.split('"score":')[1].split(',')[0].strip()
                current_score = float(score_part)
            except (ValueError, IndexError):
                continue
        
        # When we have both idx and score, add to results
        if current_idx is not None and current_score is not None:
            ids.append(current_idx)
            results.append((current_idx, current_score))
            current_idx = None
            current_score = None
    
    print(f"Total ids found: {len(ids)} for {file_path}")
    return results, ids


stages_config = {
    "0": [],
    "1": ["BBOX"],
    "2": ["SCENE.SUMMARY", "BBOX", "TEXT.DETECTION"],
    "3": ["BBOX", "SCENE.SUMMARY", "TEXT.DETECTION"],
    "4": ["TEXT.DETECTION", "BBOX", "SCENE.SUMMARY"],
    "5": ["SCENE.SUMMARY", "QUESTION.PARSING", "TEXT.DETECTION", "BBOX"],
    "6": ["TEXT.DETECTION", "BBOX", "QUESTION.PARSING","SCENE.SUMMARY"],
    "7": ["SCENE.SUMMARY", "QUESTION.PARSING", "BBOX", "TEXT.DETECTION", "COLOR.ATTRIBUTE", "COUNT"],
    "8": ["BBOX", "SCENE.SUMMARY", "QUESTION.PARSING", "TEXT.DETECTION", "COLOR.ATTRIBUTE", "COUNT"],
    "9": ["TEXT.DETECTION", "BBOX", "COLOR.ATTRIBUTE", "COUNT", "SCENE.SUMMARY", "QUESTION.PARSING"]
}



def find_poor_stage0_good_others(stage0_threshold=0.5, other_stages_threshold=0.7):
    """
    Find IDs that perform poorly on stage 0 but well on other stages.
    
    Args:
        stage0_threshold (float): Score threshold below which stage 0 performance is considered poor
        other_stages_threshold (float): Score threshold above which other stages performance is considered good
    
    Returns:
        list: List of IDs that meet the criteria
    """
    stage_results = {}
    stage_ids = {}
    
    # Load results for all stages
    for stage_key, stages in stages_config.items():
        stage_str = ','.join(stages) if stages else ""
        file_path = f"logs/eval_intuition_set_stages[{stage_str}].jsonl"
        
        try:
            result, ids = extract_idx_score(file_path)
            stage_results[stage_key] = {idx: score for idx, score in result}
            stage_ids[stage_key] = set(ids)
            print(f"Stage {stage_key}: {len(ids)} IDs")
        except Exception as e:
            print(f"Error processing stage {stage_key}: {e}")
            stage_results[stage_key] = {}
            stage_ids[stage_key] = set()
    
    # Find common IDs across all stages
    if stage_ids:
        common_ids = set.intersection(*stage_ids.values())
        print(f"Common IDs across all stages: {len(common_ids)}")
    else:
        return []
    
    # Find IDs that perform poorly on stage 0 but well on other stages
    selected_ids = []
    
    for idx in common_ids:
        # Check stage 0 performance (poor)
        stage0_score = stage_results["0"].get(idx, None)
        if stage0_score is None or stage0_score >= stage0_threshold:
            continue
        
        # Check other stages performance (good)
        other_stages_good = False
        
        for stage_key in ["1", "2", "3", "4", "5"]:
            if stage_key in stage_results:
                score = stage_results[stage_key].get(idx, None)
                if score is not None:
                    if score > stage0_threshold:
                        other_stages_good = True
                        break

        if other_stages_good:
            selected_ids.append((idx))
    
    print(f"\nFound {len(selected_ids)} IDs that perform poorly on stage 0 but well on other stages")

    new_ids = [id for id in common_ids if id not in selected_ids]
    new_ids = random.sample(new_ids, 500)
    new_ids = selected_ids + new_ids


    return stage_results, new_ids


if __name__ == "__main__":
    # get_acc()
    # mean_acc = find_common_ids_across_stages()
    # print(f"Mean acc across all stages: {mean_acc}")
    
    # Find IDs that perform poorly on stage 0 but well on other stages
    seed = 1
    random.seed(seed)
    results, new_ids = find_poor_stage0_good_others(stage0_threshold=0.5, other_stages_threshold=0.7)
    print(f"New IDs: {len(new_ids)}")

    with open("data/testing_ids.json", "w") as f:
        json.dump({"idx_list": new_ids}, f)

    print(f"Store id length: {len(new_ids)}")

    mean_acc = {}
    for stage_key, result in results.items():
        acc = []
        for idx in result:
            score = result[idx]
            if idx in new_ids:
                acc.append(score)
        mean_acc[stage_key] = sum(acc) / len(acc)
    print(f"Mean acc: {mean_acc}")

