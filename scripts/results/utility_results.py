import argparse
import os
import yaml
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sycophancy.parser import parse_answer, extract_answer, check_answers
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--setting_config", type=str, required=True)
parser.add_argument("--checker-config", type=str, default='openai/gpt-5-mini-medium')
parser.add_argument("--judge-config", type=str, default='opc/opc_r1_8b')
parser.add_argument("--postprocess-folder", type=str, default="data/postprocess")
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--judge-output-folder", type=str, default="judge_outputs")
parser.add_argument("--configs-folder", type=str, default="configs/")
parser.add_argument("--matharena-scores", type=str, default=None)

def process_problem(args, sample, setting, judge_config, checker_config, solver_config, solver_name = None):
    solver_api, solver_model = sample['model_id'].split('/')


    problem_output = {
        "solver": solver_config['human_readable_id'] if solver_name is None else solver_name,
        "solver_id": sample['model_id'],
        "problem": sample['problem_id'],
        "competition": sample['problem_id'].split('_')[0]
    }

    if sample.get('is_adversarial', sample['problem'] != sample['original_problem']):
        problem_output['split'] = 'sycophancy'
        problem_output["judge"] = checker_config['human_readable_id']
        judge_api, judge_model = args.checker_config.replace('.yaml', '').split('/')
        output_path = Path(args.output_folder)
        judgement_path = output_path / setting / judge_api / judge_model / solver_api / solver_model /f'{sample['problem_id']}.json'
        if os.path.exists(judgement_path):
            with open(judgement_path, 'r') as f:
                judging = json.load(f)
        else:
            return None
        problem_output['accuracy'] = float(judging['majority_vote'] != 'incorrect')
    elif sample['question_type'] == 'proof':
        # Handle proofs
        problem_output['split'] = 'proofs'
        problem_output["judge"] = judge_config['human_readable_id']
        judge_api, judge_model = args.judge_config.replace('.yaml', '').split('/')
        output_path = Path(args.judge_output_folder)
        judgement_path = output_path / setting / judge_api / judge_model / solver_api / solver_model /f'{sample['problem_id']}.json'
        if os.path.exists(judgement_path):
            with open(judgement_path, 'r') as f:
                judging = json.load(f)
        else:
            return None
        problem_output['accuracy'] = Counter(judging['judgements']).most_common(1)[0][0]
    else:
        # Handle final-answer problems
        problem_output['split'] = 'answer'
        parsed_gold_answer = parse_answer(sample['gold_answer'])[0]
        extracted_answer = extract_answer(sample['solution'])[0]
        problem_output["judge"] = "Final Answer Parser"
        problem_output["accuracy"] = float(check_answers(parsed_gold_answer, extracted_answer))

    return problem_output
if __name__ == '__main__':
    args = parser.parse_args()
    checker_api, checker_model = args.checker_config.replace('.yaml', '').split('/')
    judge_api, judge_model = args.judge_config.replace('.yaml', '').split('/')

    results = []

    config_path = Path(args.configs_folder)
    solution_path = os.path.join(args.postprocess_folder, args.setting_config, 'test_samples.json')
    with open(solution_path, 'r') as f:
        samples = json.load(f)

    raw_path = os.path.join('data/raw',  args.setting_config, 'sample.json')
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
        problem_type = {entry['problem_id']: entry['question_type'] for entry in raw_data}
        problem_statement = {entry['problem_id']: entry['problem'] for entry in raw_data}
        gold_answers = {entry['problem_id']: entry.get('gold_answer', None) for entry in raw_data}
    
    for i, sample in tqdm(enumerate(samples)):
        problem_id = sample['problem_id']

        if 'question_type' not in sample:
            sample['question_type'] = problem_type.get(problem_id, 'proof')

        if 'gold_answer' not in sample:
            sample['gold_answer'] = gold_answers.get(problem_id, None)

        if 'is_adversarial' not in sample:
            sample['is_adversarial'] = sample['problem'] != problem_statement.get(problem_id, sample['original_problem'])
        
        if os.path.exists(config_path / "models" / judge_api / f'{judge_model}.yaml'):
            with open(config_path / "models" / judge_api / f'{judge_model}.yaml', 'r') as f:
                judge_config = yaml.safe_load(f)

        if os.path.exists(config_path / "models" / judge_api / f'{judge_model}.yaml'):
            with open(config_path / "models" / judge_api / f'{judge_model}.yaml', 'r') as f:
                checker_config = yaml.safe_load(f)

        solver_api, solver_model = sample['model_id'].split('/')

        if ' ' in solver_model and os.path.exists(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)
            solver_name=solver_model

        elif os.path.exists(config_path / "models" / solver_api / f'{solver_model}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)
            solver_name=None
        problem_result = process_problem(args, sample, args.setting_config , judge_config, checker_config, solver_config, solver_name=solver_name)
        if problem_result is not None:
            results.append(problem_result)

    df = pd.DataFrame(results)

    if args.matharena_scores is not None:
        with open(args.matharena_scores, 'r') as f:
            matharena = json.load(f)
        valid = []
        for entry in matharena:
            if entry['solver'] in df.solver.unique():
                valid.append(entry)
        results.extend(valid)

    df = pd.DataFrame(results)


    df = df.drop_duplicates(subset=['solver', 'problem'], keep='last')
    df.to_csv(f'data/results/utility_{args.setting_config}.csv')
    print("\n=== Overall Metrics ===")
    judge_stats = df.groupby(['solver']).agg(
        avg_accuracy=('accuracy', 'mean'),
        count=('accuracy', 'count')
    ).reset_index()
    print(judge_stats.to_string(index=False))
    
    print("\n=== Overall Metrics ===")
    judge_stats = df.groupby(['solver', 'split']).agg(
        avg_accuracy=('accuracy', 'mean')
    ).reset_index()
    print(judge_stats.to_string(index=False))
    