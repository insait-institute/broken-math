import argparse
import os
import yaml
import sys
import json
from pathlib import Path
from tqdm import tqdm
from sycophancy.parser import parse_answer, extract_answer, check_answers

sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, required=True)
parser.add_argument("--model_ids", nargs='+', default=['all'])
parser.add_argument("--output-folder", type=str, default="data/verified/")
parser.add_argument("--tmp-folder", type=str, default="data/tmp/")
parser.add_argument("--configs-folder", type=str, default="configs/")
parser.add_argument("--raw-folder", type=str, default='data/raw/')
parser.add_argument("--solved-folder", type=str, default='data/solved/')

if __name__ == '__main__':
    args = parser.parse_args()
    raw_folder = os.path.join(args.raw_folder, args.project.replace('yaml', ''))
    solved_folder = os.path.join(args.solved_folder, args.project.replace('yaml', ''))
    os.makedirs(os.path.join(args.output_folder, args.project.replace('yaml', '')), exist_ok=True)
    
    adv_examples = []
    final_answer_examples = []
    proof_examples = []

    with open(os.path.join(raw_folder, 'sample.json'), 'r') as f:
        data = json.load(f)
        problem_type = {entry['problem_id']: entry['question_type'] for entry in data}
        problem_statement = {entry['problem_id']: entry['problem'] for entry in data}
        gold_answers = {entry['problem_id']: entry.get('gold_answer', None) for entry in data}

    solved_path = Path(solved_folder)
    config_path = Path(args.configs_folder)

    with open(os.path.join('data/postprocess', args.project.replace('yaml', '') ,'test_samples.json'), 'r') as f:
        data = json.load(f)
    for i, entry in tqdm(enumerate(data)):
        problem_id = entry['original_problem_id']

        if (entry['model_id'] not in args.model_ids and not args.model_ids[0]=='all') or (problem_id not in problem_statement and 'original_problem' not in entry and 'question_type' not in entry):
            continue
        if 'gold_answer' not in entry:
            entry['gold_answer'] = gold_answers.get(problem_id, None)
        original_statement = problem_statement[problem_id] if problem_id in problem_statement else entry['original_problem']
        entry_type = problem_type[problem_id] if problem_id in problem_type else entry['question_type']
        if original_statement != entry['problem'] or entry.get('is_adversarial', False):
            adv_examples.append(entry)
        elif entry_type == 'proof' or entry['gold_answer'] is None:
            proof_examples.append(entry)
        else:
            parsed_gold_answer = parse_answer(entry['gold_answer'])[0]
            extracted_answer = extract_answer(entry['solution'])[0]
            if not check_answers(parsed_gold_answer, extracted_answer):
                final_answer_examples.append(entry)
    with open(os.path.join(args.output_folder, args.project.replace('yaml', ''), 'final_answer.json'), 'w') as f:
        json.dump(final_answer_examples, f)

    tmp_path = os.path.join(args.tmp_folder, args.project.replace('yaml', ''))
    os.makedirs(tmp_path, exist_ok=True)

    with open(os.path.join(tmp_path, 'proofs.json'), 'w') as f:
        json.dump(proof_examples, f, indent=4)

    with open(os.path.join(tmp_path, 'adversarial.json'), 'w') as f:
        json.dump(adv_examples, f, indent=4)
