import argparse
import os
import yaml
import sys
import json
from pathlib import Path
from tqdm import tqdm
import random
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, required=True)
parser.add_argument("--model_id", type=str, default="qwen3_4b")
parser.add_argument("--output-folder", type=str, default="outputs/")
parser.add_argument("--judge-output-folder", type=str, default="judge_outputs/")
parser.add_argument("--verified-folder", type=str, default="data/verified/")
parser.add_argument("--tmp-folder", type=str, default="data/tmp/")
parser.add_argument("--configs-folder", type=str, default="configs/")
parser.add_argument("--val_len", type=int, default=200)
if __name__ == '__main__':
    args = parser.parse_args()

    # Parse and verify perturbed outputs
    output_dir = os.path.join(args.output_folder, args.project)
    adv_results = []
        
    output_path = Path(output_dir)
    config_path = Path(args.configs_folder)

    for problem_path in tqdm(output_path.glob('*/*/*/*/*.json')):
        judge_api, judge_model, solver_api, solver_model, _ = problem_path.parts[-5:]
        if solver_model != args.model_id:
            continue
        if os.path.exists(config_path / "models" / judge_api / f'{judge_model}.yaml'):
            with open(config_path / "models" / judge_api / f'{judge_model}.yaml', 'r') as f:
                judge_config = yaml.safe_load(f)

        if ' ' in solver_model and os.path.exists(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)

        elif os.path.exists(config_path / "models" / solver_api / f'{solver_model}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)

        with open(problem_path, 'r') as f:
            problem_data = json.load(f)

        if problem_data['majority_vote'] in ['correct', 'detected']:
            problem_data['is_adversarial'] = True
            problem_data['question_type'] = 'proof'
            adv_results.append(problem_data)

    # Parse and verify proofs
    output_dir = os.path.join(args.judge_output_folder, args.project)
    proof_results = []
        
    output_path = Path(output_dir)
    config_path = Path(args.configs_folder)

    for problem_path in tqdm(output_path.glob('*/*/*/*/*.json')):
        judge_api, judge_model, solver_api, solver_model, _ = problem_path.parts[-5:]
        if solver_model != args.model_id:
            continue
        if os.path.exists(config_path / "models" / judge_api / f'{judge_model}.yaml'):
            with open(config_path / "models" / judge_api / f'{judge_model}.yaml', 'r') as f:
                judge_config = yaml.safe_load(f)

        if ' ' in solver_model and os.path.exists(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)

        elif os.path.exists(config_path / "models" / solver_api / f'{solver_model}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)

        with open(problem_path, 'r') as f:
            problem_data = json.load(f)

        if problem_data['majority_vote'] == 'correct' or problem_data['majority_vote'] == 1:
            problem_data['is_adversarial'] = False
            problem_data['question_type'] = 'proof'
            proof_results.append(problem_data)

    # Combine and save all
    with open(os.path.join(args.verified_folder, args.project, 'final_answer.json'), 'r') as f:
        final_answer_results = json.load(f)
    
    all_results = final_answer_results + proof_results + adv_results
    random.shuffle(all_results)

    print(f'Saved a total of {len(all_results)} samples')
    print(f'Final Answer: {len(final_answer_results)}')
    print(f'Proofs: {len(proof_results)}')
    print(f'Abstention/correction: {len(adv_results)}')
    with open(os.path.join(args.verified_folder, args.project, 'full.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    all_results = [
        {
            "problem_idx": entry['problem_id'],
            "problem": entry['problem'],
            "gold_answer": entry['full_solution'],
            "question_type": entry.get('question_type', 'answer'),
            "is_adversarial": entry.get('is_adversarial', False)
        }    
     for entry in all_results]

    with open('data/train/train.json', 'w') as f:
        json.dump(all_results[args.val_len:], f, indent=4)
    with open('data/train/validation.json', 'w') as f:
        json.dump(all_results[:args.val_len], f, indent=4)
    # Delete unnecessary stuff
    # tmp_folder = os.path.join(args.tmp_folder, args.project.replace('.yaml', ''))
    # os.system(f"rm -r {tmp_folder}")
    # os.system(f"rm -r {os.path.join(args.verified_folder, args.project, 'final_answer.json')}")
