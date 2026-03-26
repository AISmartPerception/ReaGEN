"""
multi agents system
"""

init_system_prompt = """You are a helpful assistant."""

plan_prompt = """Please plan the following problem by generating a step-by-step plan. List each step in a bullet-point format, with each step described in a short sentence.

**Problem:** {question}

**Output Format Example:**

1. **Plan 1:** Describe the main goal or action of the first step

2. **Plan 2:** Describe the main goal or action of the second step

3. **Plan 3:** Describe the main goal or action of the third step

4. ...

Please strictly follow the output format to generate replies, do not generate superfluous content.
"""

solve_first_plan_prompt = """Please solve the following plan based on the image, providing the most concise response possible.

**Plan:** {plan}
"""

solve_first_plan_with_feedback_prompt = """Please solve the following plan based on the image and evaluation feedback, providing the most concise response possible.

**Plan:** {plan}

**Incorrect Answer:** {answer}

**Evaluation Feedback:** {feedback}

Please re-solve the plan based on the incorrect answer and evaluation feedback.
"""

solve_last_plan_prompt = """Please continue to solve the Plan {step} based on the image and previous plans, providing the most concise response possible.

{plans_and_answers}
{step}. **Plan {step}:** {plan}
"""

solve_last_plan_with_feedback_prompt = """Please re-solve the Plan {step} based on the image, previous plans and evaluation feedback, providing the most concise response possible.

{plans_and_answers}

{step}. **Plan {step}:** {plan}

**Incorrect Answer:** {answer}

**Evaluation Feedback:** {feedback}

Please re-solve the Plan {step} based on the incorrect answer and evaluation feedback.
"""



"""
1. **Plan 1:** [Insert Plan 1 here]
   - **Answer 1:** [Insert Answer 1 here]

2. **Plan 2:** [Insert Plan 2 here]
   - **Answer 2:** [Insert Answer 2 here]

3. **Plan 3:** [Insert Plan 3 here]
   - **Answer 3:** [Insert Answer 3 here]

4. ...
"""
plan_answer_template = """
{step}. **Plan {step}:** {plan}
   - **Answer {step}:** {answer}
"""

validate_last_answers_prompt = """Please verify whether the answer for Plan {step} is correct, check whether it conflicts with and is inconsistent with the previous plans and answers, and check whether the answer itself is consistent with the content in the picture. Clearly indicate whether Plan {step} passes the validation or not (Pass/Fail). If it has not passed the verification, please describe in detail why. 
Don't repeat prompt, don't repeat plan and answer.

**Plans and Answers:**
{plans_and_answers}
"""


# 对验证结果进行过滤，将通过验证的plan和answer保存到memory，对于不通过的，判断是否需要重新生成答案，如果判断模型已经没有足够的能力生成答案，则忽略掉这个plan和answer

judge_last_plans_prompt = """Please extract the validation results of the Plan {step} and assign an action (Save to Memory/Regenerate Answer/Ignore). For Plan {step}, follow these steps:

1. **Save to Memory:** If a plan passes the validation, save the plan and its corresponding answer to memory.
2. **Regenerate Answer:** If a plan does not pass the validation, determine if the model has the capability to regenerate a new answer.
   - If the model can regenerate a new answer, generate a new answer.
   - If the model cannot regenerate a new answer, ignore the plan and answer.

**Plans and Answers:**
{plans_and_answers}

**Validation Results for Plan {step}:**
{validation_results}

**Output Format Example:**

**Plan {step}:** <Pass/Fail>
- **Action:** <Save to Memory/Regenerate Answer/Ignore>

Please strictly follow the output format to generate replies, do not generate superfluous content.
"""


summarize_final_answer_prompt = """Based on the overall planning process, summarize the thought process and provide the final answer to the original question. 

Example of a valid response:
The summarized thought process is: ...
The final_answer is: ...

**Original Question:** {question}

**Overall Thought Process:** 
{plans_and_answers}

Please provide the final answer based on the overall thought process.
"""

summarize_final_answer_without_plans_prompt = """Please answer the following Problem step by step.
**Problem:** {question}
"""