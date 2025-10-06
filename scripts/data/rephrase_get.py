import argparse
import json
import time
import re
import os
import yaml
import numpy as np
import pandas as pd
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets, Dataset

parser = argparse.ArgumentParser()

parser.add_argument('--batches', type=str, nargs='+') # Batch ID for mode = id, and file name for mode file
parser.add_argument('--dataset', type=str)
parser.add_argument('--output', '-o', type=str, default='./data/raw/sycophancy')
parser.add_argument('--json-out', action='store_true')
parser.add_argument('--training', action='store_true')
args = parser.parse_args()

def parse_line(line):
    data = json.loads(line)
    raw_out = data['response']['body']['choices'][0]['message']['content']
    return raw_out

def format_to_json(data, out_path, additional_cols = None):
    problems = []
    for i, row in data.iterrows():
        entry = {
            "original_problem": row['problem'],
            "problem": row["rephrased_statement"],
            "problem_id": row["problem_id"] if "problem_id" in row else f"{row['competition']}_{i}",
            "solutions": [
                {
                    "author": "Human",
                    "solution": row.get("solution", row.get("ground_truth_solution", ""))
                }
            ]
        }
        if additional_cols is not None:
            for col in additional_cols:
                entry[col] = row[col]
        problems.append(entry)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(problems, f)

if __name__ == '__main__':
    client = OpenAI(
        api_key=os.environ['OPEN_AI_KEY'],
        organization='org-G6dhx70fTdN7hVOnf2SGkYaE',
        project='proj_HokLep8NLZOKpNesn38DqBuy',
    )
    

    results = []
    for batch in args.batches:
        batch_job = client.batches.retrieve(batch)
        if batch_job.status == 'completed':
            result_file_id = batch_job.output_file_id
            file_response = client.files.content(result_file_id).text
            for line in file_response.splitlines():
                body = parse_line(line)
                results.append(body)
    if os.path.exists(args.dataset) and args.training:
        data = pd.read_json(args.dataset)
        rephrased_statements = []
        counter = 0
        for i, row in data.iterrows():
            if row['needs_rephrasing']:
                rephrased_statements.append(results[counter])
                counter += 1
            else:
                rephrased_statements.append(row['problem'])
        data['rephrased_statement'] = rephrased_statements
        data = data[~(data.rephrased_statement.str.len() < 50)]
        format_to_json(data, args.dataset.replace('.json','')+'_rephrased.json', additional_cols=['question_type', 'gold_answer', 'is_adversarial',])
    elif os.path.exists(args.dataset) and args.dataset.endswith('.csv'):
        data = pd.read_csv(args.dataset)
        data['rephrased_statement'] = results
        data = data[~(data.rephrased_statement.str.len() < 50)]
        if args.json_out:
            format_to_json(data, args.dataset.replace('.csv','')+'_rephrased.json')
        else:
            data.to_csv(args.dataset.replace('.csv','')+'_rephrased.csv', index=False)
            print(f'Successfully stored new dataset in local file {args.dataset.replace('.csv','')+"_rephrased.csv"}')
    elif os.path.exists(args.dataset) and args.dataset.endswith('.json'):
        data = pd.read_json(args.dataset)
        data = data[data['solutions'].apply(lambda x: len(x) > 0)]
        data['solution'] = data['solutions'].apply(lambda x: x[0]['solution'])
        data['rephrased_statement'] = results
        data = data[~(data.rephrased_statement.str.len() < 50)]
        format_to_json(data, args.dataset.replace('.json','')+'_rephrased.json')
    elif os.path.exists(args.dataset) and args.dataset.endswith('.yaml'):
        with open(args.dataset, 'r') as f:
            config = yaml.safe_load(f)
        data = load_dataset(config['dataset'], token=os.environ['HF_API_KEY'])
        splits = config.get('splits', [k for k in data])
        data = concatenate_datasets([data[k] for k in splits]).to_pandas()
        data['rephrased_statement'] = results
        data = data[~(data.rephrased_statement.str.len() < 50)]
        if args.json_out:
            format_to_json(data, os.path.join(args.output, f'{config['dataset'].replace('/','_')}_rephrased.json'))
        else:
            Dataset.from_pandas(data).push_to_hub(f'{args.dataset}_rephrased', token=os.environ['HF_API_KEY'])
            print(f'Successfully stored new dataset in local file {args.dataset.replace('.csv','')+"_rephrased.csv"}')
    else:
        data = load_dataset(args.dataset, token=os.environ['HF_API_KEY'])
        data = concatenate_datasets([data[k] for k in data]).to_pandas()
        data['rephrased_statement'] = results
        data = data[~(data.rephrased_statement.str.len() < 50)]
        if args.json_out:
            format_to_json(data, args.dataset.replace('/','')+'_rephrased.json')
        else:
            Dataset.from_pandas(data).push_to_hub(f'{args.dataset}_rephrased', token=os.environ['HF_API_KEY'])
            print(f'Successfully stored new dataset in HuggingFace repo {args.dataset}_rephrased')