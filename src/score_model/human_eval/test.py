import itertools
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Union
import re
from src.score_model.human_eval.human_eval.data import read_problems, write_jsonl
import subprocess

# New Evaluation Function
def test(model, config):
    model.eval()
    samples_first = []
    samples_second = []
    problems = read_problems()
    tokenizer = AutoTokenizer.from_pretrained(config.model_variant)
    tokenizer.pad_token = tokenizer.eos_token
    all_keys = list(problems.keys())
    last_64_keys = all_keys[-64:]
    selected_problems = {k: problems[k] for k in last_64_keys}

    write_jsonl("selected_problems.jsonl",
                [{"task_id": k, **v} for k, v in selected_problems.items()])
    for problem_id in tqdm(last_64_keys):
        #  Prompt 1
        #print(problem_id)
        coding_prompt = problems[problem_id]["prompt"]
        #print("Prompt\n")
        #print(coding_prompt)
        prompt1_header = "You are an expert Python programmer, and here is your task: Complete the following python function: \n"
        prompt1 = prompt1_header + coding_prompt
        input1 = tokenizer(prompt1, return_tensors="pt").to(model.device)

        output1 = model.model.generate(
            input1.input_ids,
            max_length=500,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_code1 = tokenizer.decode(output1[0], skip_special_tokens=True)

        completion1 = generated_code1[len(prompt1):] # remove header and coding prompt

        # Prompt 2
        prompt2_header = "There might be an error in the code below because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program! \n"
        prompt2 = prompt2_header + generated_code1[len(prompt1_header):] # adds just the coding portion of prompt 1's response

        input2 = tokenizer(prompt2, return_tensors="pt").to(model.device)

        output2 = model.model.generate(
            input2.input_ids,
            max_length=1000,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_code2 = tokenizer.decode(output2[0], skip_special_tokens=True)

        completion2 = generated_code2[len(prompt2_header) + len(coding_prompt):] # remove header and coding prompt

        samples_first.append({
            "task_id": problem_id,
            "completion": completion1
        })

        samples_second.append({
            "task_id": problem_id,
            "completion": completion2
        })
    
    write_jsonl("humaneval_samples1.jsonl", samples_first)
    write_jsonl("humaneval_samples2.jsonl", samples_second)
    from src.score_model.human_eval.human_eval.evaluation import evaluate_functional_correctness
    
    # Call the function directly
    results1 = evaluate_functional_correctness(
        sample_file="humaneval_samples1.jsonl",
        problem_file="selected_problems.jsonl"
    )
    print(results1)
    
    results2 = evaluate_functional_correctness(
        sample_file="humaneval_samples2.jsonl",
        problem_file="selected_problems.jsonl"
    )
    print(results2)

    file_path = "humaneval_samples1.jsonl_results.jsonl"
    data = []

    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    data_dict1 = {i: item for i, item in enumerate(data)}

    file_path = "humaneval_samples2.jsonl_results.jsonl"
    data = []

    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    data_dict2 = {i: item for i, item in enumerate(data)}
    # Get Results
    count_i_c = 0
    count_c_i = 0
    correct_1 = 0
    correct_2 = 0
    num_problems = len(selected_problems)
    for i in range(len(data_dict1)):
        if data_dict1[i]['passed']:
            correct_1 += 1
        if data_dict2[i]['passed']:
            correct_2 += 1
        if data_dict1[i]['passed'] and not data_dict2[i]['passed']:
            count_c_i += 1
        elif not data_dict1[i]['passed'] and data_dict2[i]['passed']:
            count_i_c += 1

    print("Accuracy@t1: " + str(correct_1 / num_problems))
    print("Accuracy@t2: " + str(correct_2 / num_problems))
    print("delta(t1,t2): " + str((correct_2 - correct_1) / num_problems))
    print("delta(t1,t2) i to c: " + str(count_i_c / num_problems))
    print("delta(t2,t1) c to i: " + str(count_c_i / num_problems))