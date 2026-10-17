import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

test_case = {
    "id": "test_001",
    "input": "Write a Python function to check if a number is prime.",
    "model_output": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True",
    "criteria": "correctness and efficiency"
}
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0  # Keeps the grading strict and consistent
)
grading_template = PromptTemplate(
    template="""
    You are an expert AI teacher. Grade the model output based on the input and criteria.
    Input: {input}
    Model Output: {model_output}
    Criteria: {criteria}
    Respond in strict JSON format using these exact keys:
    {{
        "score": <integer from 1 to 5>,
        "verdict": "PASS or FAIL",
        "rationale": "explain your grade here"
    }}
    """,
    input_variables=["input", "model_output", "criteria"]
)
def run_evaluation():
    start_time = time.time()
    prompt = grading_template.invoke({
        "input": test_case["input"],
        "model_output": test_case["model_output"],
        "criteria": test_case["criteria"]
    })
    response = judge_llm.invoke(prompt)

    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 2)

    try:
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        grading_result = json.loads(clean_text)
    except Exception as e:
        grading_result = {
            "score": 0,
            "verdict": "ERROR",
            "rationale": "Failed to read JSON: " + str(e)
        }
    print("Test ID:", test_case["id"])
    print("Score:", grading_result.get("score"), "/ 5")
    print("Verdict:", grading_result.get("verdict"))
    print("Rationale:", grading_result.get("rationale"))
    print("Latency:", latency_ms, "ms")

run_evaluation()