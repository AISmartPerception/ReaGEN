SYSTEM_SCORING_INST = """You are a helpful assistant. You will be given a question, the correct answer (ground truth answer), and a model response. Your task is to:
1. Extract the relevant answer from the model's response, ignoring any irrelevant or extra information. 
2. Compare the extracted answer with the correct answer and determine whether the model's response is correct. If the extracted answer is the same as the correct answer, set the "score" to 1. It's acceptable to have different grammar or form. If the extracted answer is different from the correct answer, set the "score" to 0.

The input will be formatted as follows:
**Question:** <question>
**Ground Truth Answer:** <ground_truth>
**Model Response:** <model_response>

Please extract the answer based on the model's response and output the result in the following JSON format:
{
  "extracted_answer": "<extracted answer>",
  "score": <score>
}
Ensure that the response strictly follows this structure and is valid JSON. Do not generate additional information and ensure that I can parse the json directly from your output.
"""

SCORING_INST = """The input example is provided below:
**Question:** {question}
**Ground Truth Answer:** {ground_truth}
**Model Response:** {model_response}
"""

USER_SCORING_INST = """You are a helpful assistant. You will be given a **Question**, the **Ground Truth Answer**, and a **Predicted Answer**. Your task is to compare the **Ground Truth Answer** with the **Predicted Answer** and determine whether the **Predicted Answer** is correct. It's acceptable to have different grammar or form. If the **Predicted Answer** is correct, you should say "Yes". If the **Predicted Answer** is incorrect, you should say "No".

**Question:** {question}
**Ground Truth Answer:** {ground_truth}
**Predicted Answer:** {model_response}

Is the **Predicted Answer** correct?
"""

guided_choice = ["Yes", "No"]