import json

def calculate_average_stats(generation_file):
    # Initialize counters for total student calls, input/output tokens
    total_student_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_total_tokens = 0
    total_samples = 0

    try:
        # Load the generation.json file
        with open(generation_file, 'r') as f:
            data = json.load(f)

        # Loop through all the samples (problems/questions)
        for sample_id, sample in data.items():
            # Check if 'vrest_stats' exists in the sample
            if 'decision' in sample:
                vrest_stats = sample['decision']['vrest_stats']
                
                # Add to total counters
                total_student_calls += vrest_stats.get('student_calls', 0)
                total_input_tokens += vrest_stats.get('input_tokens', 0)
                total_output_tokens += vrest_stats.get('output_tokens', 0)
                total_total_tokens += vrest_stats.get('total_tokens', 0)
                total_samples += 1

                # Optional: Debugging: print first few sample data (can be removed)
                print(f"Sample ID: {sample_id}")
                print(f"Student Calls: {vrest_stats.get('student_calls', 0)}")
                print(f"Input Tokens: {vrest_stats.get('input_tokens', 0)}")
                print(f"Output Tokens: {vrest_stats.get('output_tokens', 0)}")
                print(f"Total Tokens: {vrest_stats.get('total_tokens', 0)}")

        # Calculate averages
        if total_samples > 0:
            avg_student_calls = total_student_calls / total_samples
            avg_input_tokens = total_input_tokens / total_samples
            avg_output_tokens = total_output_tokens / total_samples
            avg_total_tokens = total_total_tokens / total_samples

            # Print the results
            print(f"\nAverage number of student calls: {avg_student_calls:.2f}")
            print(f"Average number of input tokens: {avg_input_tokens:.2f}")
            print(f"Average number of output tokens: {avg_output_tokens:.2f}")
            print(f"Average number of total tokens: {avg_total_tokens:.2f}")
        else:
            print("No valid data found in the file.")

    except Exception as e:
        print(f"Error: {str(e)}")

# Provide the path to your generation.json file
generation_file = '/home/mohan/InternVL2-main/prompt_evolution_direction/Baselines/VReST/outputs/data=mmmu_pro_standard_10_100/llm=Qwen3_VL_4B_Instruct/prompt_method=mctsv8/generation.json'
calculate_average_stats(generation_file)
