"""
设计一个多模态agent框架，能够对复杂的图标问答问题进行多步推理，假设你可以调用一个多模态大模型的API，请你帮我设计符合要求的prompt。要求如下：
1、首先对问题进行精细化，澄清其中不清楚的实体、定义和概念。
2、从这个复杂的问题中动态分解出一个子问题。
3、回答这个子问题。
4、验证子问题是否回答正确，如果不正确，重新执行第3步。
5、判断以上推理是否能够得出最终答案，如果不可以则返回第2步，继续动态分解子问题。
6、综合上述推理，总结成一个思维链，得出最终答案。
"""

init_system_prompt = """You are a helpful assistant.
"""

refine_question = """Please analyze the following question in detail, identifying any unclear entities, definitions, and concepts, and clarify them one by one. Then paraphrase the question in a more complete and understandable way.
The original question is: {question}
"""

decompose_question_first = """
To better answer complex questions based on images, please generate a step-by-step plan to solve the problem. The plan should be structured in a way that each step logically follows the previous one. Provide the title of the first step in the plan, ensuring it is a clear and specific action that logically precedes others in solving the complete question. Only generate the title of the first step, without repeating the original question or generating any other content.

**Example Demonstration:**

- **Complete Question:** Among all bars, what is the least time percentage difference between any two neighboring bars?
- **First Step Title:** Identify the number of bars and their respective labels or categories.

**Your Task:**

- **Complete Question:** {question}
- **First Step Title:** Generate the title of the first step here
"""

# decompose_question_first = """To better answer complex questions based on image, please write down the first sub-question that needs to be solved, based on the complete question. Only generate the question, without generating any other content.

# There is a demonstration of how to decompose the question:
# The complete question is: Among all bars, what is the least time percentage difference between any two neighboring bars?
# The first sub-question is: Determine the number of bars and their respective labels or categories.

# The complete question is: {question}
# So, please generate the first sub-question.
# """

decompose_question = """
To better answer complex questions based on images, please generate a step-by-step plan to solve the problem. The plan should be structured in a way that each step logically follows the previous one. Provide the title of the next step in the plan, ensuring it is a clear and specific action that logically follows the already resolved steps in solving the complete question. Only generate the title of the next step, without repeating the original question or generating any other content.

**Example Demonstration:**

- **Complete Question:** Among all bars, what is the least time percentage difference between any two neighboring bars?
- **Steps Solved:**
  - Step 1: Identify the number of bars and their respective labels or categories.
  - Step 2: Obtain the time percentage values associated with each bar.
  - Step 3: Compute the percentage difference between each pair of neighboring bars.
- **Next Step Title:** Identify the smallest percentage difference among all calculated differences.

**Your Task:**

- **Complete Question:** {question}
- **Steps Solved:** 
{sub_questions}
- **Next Step Title:** Generate the title of the next step here
"""

# decompose_question = """To better answer complex questions based on image. Based on the complete question and the already resolved sub-questions, write down the next sub-question that needs to be solved, ensuring it does not duplicate any existing sub-questions. Only generate the question, without generating any other content.

# There is a demonstration of how to decompose the question:
# The complete question is: Among all bars, what is the least time percentage difference between any two neighboring bars?
# The sub-questions that has been solved are:
# Sub-question 1: Determine the number of bars and their respective labels or categories.
# Sub-question 2: Obtain the time percentage values associated with each bar.
# Sub-question 3: Compute the percentage difference between each pair of neighboring bars.
# The next sub-question is: Identify the smallest percentage difference among all calculated differences.

# The complete question is: {question}
# The sub-questions that has been solved are: 
# {sub_questions}
# So, please generate the next sub-question.
# """

decompose_question_with_feedback = """
To better answer complex questions based on images, please generate a step-by-step plan to solve the problem. The plan should be structured in a way that each step logically follows the previous one and addresses the feedback provided. Provide the title of the next step in the plan, ensuring it is a clear and specific action that logically follows the already resolved steps and addresses the feedback. Only generate the title of the next step, without repeating the original question or generating any other content.

**Example Demonstration:**

- **Complete Question:** Among all bars, what is the least time percentage difference between any two neighboring bars?
- **Steps Solved:**
  - Step 1: Identify the number of bars and their respective labels or categories.
  - Step 2: Obtain the time percentage values associated with each bar.
  - Step 3: Compute the percentage difference between each pair of neighboring bars.
- **Feedback:** We have not identified the smallest percentage difference among all calculated differences.
- **Next Step Title:** Identify the smallest percentage difference among all calculated differences.

**Your Task:**

- **Complete Question:** {question}
- **Steps Solved:** 
{sub_questions}
- **Feedback:** {feedback}
- **Next Step Title:** Generate the title of the next step here
"""

# decompose_question_with_feedback = """To better answer complex questions based on image. Based on the complete question and the already resolved sub-questions, write down the next sub-question that needs to be solved, ensuring it does not duplicate any existing sub-questions. Only generate the question, without generating any other content.

# There is a demonstration of how to decompose the question:
# The complete question is: Among all bars, what is the least time percentage difference between any two neighboring bars?
# The sub-questions that has been solved are:
# Sub-question 1: Determine the number of bars and their respective labels or categories.
# Sub-question 2: Obtain the time percentage values associated with each bar.
# Sub-question 3: Compute the percentage difference between each pair of neighboring bars.
# Now we have not sufficient reasoning to derive the final answer beacuse: We have not identified the smallest percentage difference among all calculated differences.
# The next sub-question is: Identify the smallest percentage difference among all calculated differences.

# The complete question is: {question}
# The sub-questions that has been solved are:
# {sub_questions}
# Now we have not sufficient reasoning to derive the final answer beacuse: 
# {feedback}
# Please generate the next sub-question.
# """

answer_sub_question = """Please answer the following questions based on the image, providing the most concise response possible. 
The question is: {question}
"""

reanswer_sub_question = """Please reanswer the following question based on the image and evaluation feedback, providing the most concise response possible.
The question is: {question}
The incorrect answer is: {answer}
Please reanswer the question.
"""

reanswer_sub_question_with_feedback = """Please reanswer the following question based on the image and evaluation feedback, providing the most concise response possible.

The question is: {question}
The incorrect answer is: {answer}
The answer is incorrect because: {feedback}
Please reanswer the question.
"""

validate_answer = """Please verify the answers to the following question based on the content of the image. Evaluate and identify any errors, inconsistencies, or logical fallacies in the answer. 
If the answer is incorrect, write the corresponding evaluation feedback and set the next action to 'answer_sub_question'. Otherwise, set the evaluation feedback to 'The answer is correct.' and set the next action to 'check_final_answer'.

The question is: {question}
The answer is: {answer}

Example of a valid response:
- feedback: The answer is incorrect because...
- next_action: answer_sub_question
"""

check_final_answer = """Evaluate whether the current reasoning and answers to sub-questions are sufficient to derive the final answer. 
If the current reasoning and answers are sufficient, set the evaluation feedback to 'The reasoning is sufficient' and set the next action to 'generate_final_answer'. Otherwise, write the corresponding feedback and set the next action to 'decompose_question'.

The original question is: {question}
The current reasoning process is: 
{reasoning}

Example of a valid response:
- feedback: The reasoning is insufficient because...
- next_action: decompose_question
"""

generate_final_answer = """Based on the overall reasoning process, summarize the thought process and provide the final answer to the original question. 

Example of a valid response:
The summarized thought process is: ...
The final_answer is: ...

The original question is: {question}
The overall thought process is: 
{reasoning}
"""