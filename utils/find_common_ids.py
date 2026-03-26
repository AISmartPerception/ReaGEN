#!/usr/bin/env python3
"""
Script to find the intersection of sample IDs across all log files.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def extract_ids_from_file(file_path):
    """
    Extract all idx values from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
    
    Returns:
        set: Set of unique idx values found in the file
    """
    ids = set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split by lines and reconstruct JSON objects
        lines = content.split('\n')
        current_json = ""
        brace_count = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines when not inside a JSON object
            if not line and brace_count == 0:
                continue
                
            # Count braces to track JSON object boundaries
            if line.startswith('{'):
                current_json = line
                brace_count = line.count('{') - line.count('}')
            elif brace_count > 0:
                current_json += " " + line
                brace_count += line.count('{') - line.count('}')
            
            # When we have a complete JSON object
            if brace_count == 0 and current_json:
                try:
                    data = json.loads(current_json)
                    if 'idx' in data:
                        ids.add(data['idx'])
                except json.JSONDecodeError:
                    pass
                finally:
                    current_json = ""
                    brace_count = 0
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()
    
    return ids

def find_common_ids_across_files(file_patterns, logs_dir="logs"):
    """
    Find IDs that appear in ALL specified files.
    
    Args:
        file_patterns (list): List of file patterns or full filenames
        logs_dir (str): Directory containing the log files
    
    Returns:
        tuple: (common_ids_set, file_stats_dict)
    """
    logs_path = Path(logs_dir)
    file_ids = {}
    
    # Get all matching files
    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]
    
    matching_files = []
    for pattern in file_patterns:
        if pattern.startswith("eval_intuition_set_stages"):
            # Direct filename
            file_path = logs_path / pattern
            if file_path.exists():
                matching_files.append(str(file_path))
        else:
            # Pattern matching
            for file_path in logs_path.glob(f"*{pattern}*"):
                if file_path.is_file() and file_path.suffix == '.jsonl':
                    matching_files.append(str(file_path))
    
    if not matching_files:
        print("No matching files found!")
        return set(), {}
    
    print(f"Processing {len(matching_files)} files...")
    
    # Extract IDs from each file
    for file_path in matching_files:
        filename = Path(file_path).name
        print(f"Extracting IDs from: {filename}")
        ids = extract_ids_from_file(file_path)
        file_ids[filename] = ids
        print(f"  Found {len(ids)} unique IDs")
    
    # Find intersection (IDs that appear in ALL files)
    if file_ids:
        common_ids = set.intersection(*file_ids.values())
        print(f"\nIDs common to ALL files: {len(common_ids)}")
    else:
        common_ids = set()
    
    return common_ids, file_ids

def analyze_id_coverage(file_ids):
    """
    Analyze which IDs appear in how many files.
    """
    id_count = defaultdict(int)
    all_ids = set()
    
    for filename, ids in file_ids.items():
        all_ids.update(ids)
        for id_val in ids:
            id_count[id_val] += 1
    
    print(f"\nID Coverage Analysis:")
    print(f"Total unique IDs across all files: {len(all_ids)}")
    
    # Count how many IDs appear in each number of files
    coverage_stats = defaultdict(int)
    for id_val, count in id_count.items():
        coverage_stats[count] += 1
    
    print(f"ID coverage distribution:")
    for num_files in sorted(coverage_stats.keys(), reverse=True):
        print(f"  IDs in {num_files} file(s): {coverage_stats[num_files]}")
    
    return id_count, coverage_stats

def main():
    # Define the evaluation log files you want to analyze
    eval_files = [
        "eval_intuition_set_stages[].jsonl",
        "eval_intuition_set_stages[BBOX].jsonl",
        "eval_intuition_set_stages[SCENE.SUMMARY,BBOX,TEXT.DETECTION].jsonl",
        "eval_intuition_set_stages[BBOX,SCENE.SUMMARY,TEXT.DETECTION].jsonl",
        "eval_intuition_set_stages[TEXT.DETECTION,BBOX,SCENE.SUMMARY].jsonl",
        "eval_intuition_set_stages[SCENE.SUMMARY,QUESTION.PARSING,BBOX,TEXT.DETECTION,COLOR.ATTRIBUTE,COUNT].jsonl"
    ]
    
    # Find common IDs across all files
    common_ids, file_ids = find_common_ids_across_files(eval_files)
    
    if common_ids:
        print(f"\nCommon IDs found: {sorted(list(common_ids))[:10]}..." if len(common_ids) > 10 else f"\nCommon IDs found: {sorted(list(common_ids))}")
        
        # Save common IDs to file
        output_file = "common_ids_across_all_files.txt"
        with open(output_file, 'w') as f:
            for id_val in sorted(common_ids):
                f.write(f"{id_val}\n")
        print(f"Common IDs saved to: {output_file}")
    else:
        print("\nNo IDs are common to ALL files.")
    
    # Detailed analysis
    analyze_id_coverage(file_ids)
    
    return common_ids, file_ids

if __name__ == "__main__":
    main()
