import json

def filter_vision_only(input_file, output_file):
    """
    Filter out only the samples with the "Vision only" category from MathVerse dataset.
    Assumes the file is a JSON array (list of dictionaries).
    """
    with open(input_file, "r") as f:
        try:
            # Load the entire JSON file as a list of dictionaries
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            return

    # Filter "Vision only" samples
    filtered_data = [item for item in data if item.get("problem_version") == "Vision Only"]

    # Save the filtered data to a new file
    with open(output_file, "w") as f:
        for item in filtered_data:
            json.dump(item, f)
            f.write("\n")  # Write each item on a new line to preserve JSONL format

    print(f"Filtered {len(filtered_data)} Vision only samples.")
    
# Usage:
input_file = "/home/mohan/InternVL2-main/prompt_evolution_direction/Baselines/VReST/data/MathVerse/testmini.json"  # Path to the file you uploaded
output_file = "/home/mohan/InternVL2-main/prompt_evolution_direction/Baselines/VReST/data/MathVerse/testmini_vision_only.jsonl"  # Path where you want the filtered data

filter_vision_only(input_file, output_file)
