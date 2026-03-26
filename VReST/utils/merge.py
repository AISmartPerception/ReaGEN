import json
import pandas as pd

def merge_json_to_excel(res_paths, output_file, data_name):
    datas = {}
    category = []

    # 遍历每个方法和文件路径
    for model, methods in res_paths.items():
        for method, file_path in methods.items():
            # 读取 JSON 文件
            with open(file_path, 'r') as f:
                data = json.load(f)
            datas[f"{model}_{method}"] = data
            category.append(f"{model}_{method}")

    full_ids = list(datas[f"{model}_{method}"].keys())

    if data_name == "charxiv":
        merged_dict = {}
        for full_id in full_ids:
            merged_dict[full_id] = {}
            merged_dict[full_id]["figure_id"] = datas[category[-1]][full_id]["figure_id"]
            merged_dict[full_id]["question"] = datas[category[-1]][full_id]["question"]
            merged_dict[full_id]["ground_truth"] = datas[category[-1]][full_id]["ground_truth"]
            for method in category:
                merged_dict[full_id][f"{method}_response"] = datas[method][full_id]["response"]
            for method in category:
                merged_dict[full_id][f"{method}_extracted_answer"] = datas[method][full_id]["extracted_answer"]
            for method in category:
                merged_dict[full_id][f"{method}_score"] = bool(datas[method][full_id]["llm_score"])
    elif data_name == "mathvista":
        merged_dict = {}
        for full_id in full_ids:
            merged_dict[full_id] = {}
            merged_dict[full_id]["figure_id"] = datas[category[-1]][full_id]["pid"]
            merged_dict[full_id]["question"] = datas[category[-1]][full_id]["query"]
            merged_dict[full_id]["ground_truth"] = datas[category[-1]][full_id]["answer"]
            for method in category:
                merged_dict[full_id][f"{method}_response"] = datas[method][full_id]["response"]
            for method in category:
                merged_dict[full_id][f"{method}_extracted_answer"] = datas[method][full_id]["extracted_answer"]
            for method in category:
                merged_dict[full_id][f"{method}_score"] = bool(datas[method][full_id]["true_false"])
    elif data_name == "mathvision":
        merged_dict = {}
        for full_id in full_ids:
            merged_dict[full_id] = {}
            merged_dict[full_id]["figure_id"] = datas[category[-1]][full_id]["id"]
            merged_dict[full_id]["question"] = datas[category[-1]][full_id]["question"]
            merged_dict[full_id]["ground_truth"] = datas[category[-1]][full_id]["answer"]
            for method in category:
                merged_dict[full_id][f"{method}_response"] = datas[method][full_id]["response"]
            for method in category:
                merged_dict[full_id][f"{method}_extracted_answer"] = datas[method][full_id]["extracted_answer"]
            for method in category:
                merged_dict[full_id][f"{method}_score"] = bool(datas[method][full_id]["llm_score"])
    else:
        raise ValueError("Data not supported")
        
    df = pd.DataFrame.from_dict(merged_dict, orient='index')
    df.to_excel(output_file, index=False)
    print(f"Saved to {output_file}")
