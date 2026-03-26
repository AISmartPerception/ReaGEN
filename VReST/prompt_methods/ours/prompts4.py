"""
multi agents system
"""

init_system_prompt = """You are a helpful assistant."""

plan_prompt = """Please plan the following problem by generating a step-by-step plan. Ensure that each step incorporates relevant visual information where applicable. List each step in a bullet-point format, with each step described in a short sentence. The output format should be easy to parse and split, with clear delimiters (such as numbered points or special symbols) between steps.

**Problem:** {question}

**Output Format Example:**

1. **Plan 1:** Describe the main goal or action of the first step

2. **Plan 2:** Describe the main goal or action of the second step

3. **Plan 3:** Describe the main goal or action of the third step

4. ...

"""

solve_plan_prompt = """Please solve the following plan based on the image, providing the most concise response possible.

**Plan:** {plan}
"""

solve_plan_with_feedback_prompt = """Please solve the following plan based on the image and evaluation feedback, providing the most concise response possible.

**Plan:** {plan}

**Incorrect Answer:** {answer}

**Evaluation Feedback:** {feedback}

Please re-solve the plan based on the incorrect answer and evaluation feedback.
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

"Indicate whether it passes validation: Pass/Fail"

validate_answers_prompt = """Please evaluate the following plans and their corresponding answers for errors, inconsistencies, or logical fallacies. For each plan, provide a separate evaluation point. Clearly indicate whether each plan passes the validation or not. The output format should be easy to parse and split, with clear delimiters (such as numbered points or special symbols) between evaluations. Don't repeat prompt, don't repeat plan and answer. Please directly generate an evaluation for each answer.

**Plans and Answers:**
{plans_and_answers}

**Output Format Example:**

1. **Plan 1:** <Evaluation result of Plan 1. Include an analysis of the answers.>

2. **Plan 2:** <Evaluation result of Plan 2. Include an analysis of the answers.>

3. **Plan 3:** <Evaluation result of Plan 3. Include an analysis of the answers.>

4. ...

"""

"""
1. **Plan 1:** [Insert Plan 1 here]
   - **Answer 1:** [Insert Answer 1 here]
   - **Validation Result 1:** [Insert Validation Result 1 here (Pass/Fail)]

2. **Plan 2:** [Insert Plan 2 here]
   - **Answer 2:** [Insert Answer 2 here]
   - **Validation Result 2:** [Insert Validation Result 2 here (Pass/Fail)]

3. **Plan 3:** [Insert Plan 3 here]
   - **Answer 3:** [Insert Answer 3 here]
   - **Validation Result 3:** [Insert Validation Result 3 here (Pass/Fail)]

4. ...
"""
validation_results_template = """
{step}. **Plan {step}:** {plan}
   - **Answer {step}:** {answer}
   - **Validation Result {step}:** {validation_result}
"""

# 对验证结果进行过滤，将通过验证的plan和answer保存到memory，对于不通过的，判断是否需要重新生成答案，如果判断模型已经没有足够的能力生成答案，则忽略掉这个plan和answer
judge_plans_prompt = """Please filter the validation results of the following plans and their corresponding answers. For each plan, follow these steps:

1. **Save to Memory:** If a plan passes the validation, save the plan and its corresponding answer to memory.
2. **Regenerate Answer:** If a plan does not pass the validation, determine if the model has the capability to regenerate a new answer.
   - If the model can regenerate a new answer, generate and save the new answer.
   - If the model cannot regenerate a new answer, ignore the plan and answer.

**Plans and Answers:**
{plans_and_answers}

**Validation Results:**
{validation_results}

**Output Format Example:**

1. **Plan 1:** <Validation Result 1>
   - **Action:** <Save to Memory/Regenerate Answer/Ignore>

2. **Plan 2:** <Validation Result 2>
   - **Action:** <Save to Memory/Regenerate Answer/Ignore>

3. **Plan 3:** <Validation Result 3>
   - **Action:** <Save to Memory/Regenerate Answer/Ignore>

4. ...

---

**Output Example:**

1. **Plan 1:** Pass
   - **Action:** Save to Memory

2. **Plan 2:** Fail
   - **Action:** Regenerate Answer

3. **Plan 3:** Fail
   - **Action:** Ignore

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