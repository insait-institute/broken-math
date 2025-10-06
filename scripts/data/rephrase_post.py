import argparse
import json
import os
import pandas as pd
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets
import yaml
from prompts import REPHRASE_PROMPT
parser = argparse.ArgumentParser()

parser.add_argument('input', type=str)
parser.add_argument('--output', '-o', type=str, default='./')
parser.add_argument('--training', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    client = OpenAI(
        api_key=os.environ['OPEN_AI_KEY'],
        organization='org-G6dhx70fTdN7hVOnf2SGkYaE',
        project='proj_HokLep8NLZOKpNesn38DqBuy',
    )
    tasks = []
    sol_column='solution'
    if os.path.exists(args.input) and args.training:
        data = pd.read_json(args.input)
        data = data[data.needs_rephrasing]
    elif os.path.exists(args.input) and args.input.endswith('.csv'):
        data = pd.read_csv(args.input)
    elif os.path.exists(args.input) and args.input.endswith('.json'):
        data = pd.read_json(args.input)
        data = data[data['solutions'].apply(lambda x: len(x) > 0)]
        data['solution'] = data['solutions'].apply(lambda x: x[0]['solution'])
    elif os.path.exists(args.input) and args.input.endswith('.yaml'):
        with open(args.input, 'r') as f:
            config = yaml.safe_load(f)
        data = load_dataset(config['dataset'], token=os.environ['HF_API_KEY'])
        splits = config.get('splits', [k for k in data])
        data = concatenate_datasets([data[k] for k in splits]).to_pandas()
        sol_column=config.get('solution_column', 'solution')
    else:
        data = load_dataset(args.input, token=os.environ['HF_API_KEY'])
        data = concatenate_datasets([data[k] for k in data]).to_pandas()

    for i, row in data.iterrows():
        if 'statement' in row:
            problem = row['statement']
        elif 'problem_statement' in row:
            problem = row['problem_statement']
        else:
            problem = row['problem']

        body = {
            "model":"gpt-5-mini",
            "messages":[
                {
                "role": "user",
                "content": REPHRASE_PROMPT.format(problem=problem, solution=row[sol_column])
                }
            ],
            "reasoning_effort": "high",
            "verbosity": "low"
        }

        task = {
            "custom_id": f"rephrase_sycophancy-{args.input.split('/')[-1]}-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        tasks.append(task)

    file_name = f"./rephrase_sycophancy-{args.input.split('/')[-1]}"
    files = [file_name+'_1.jsonl']
    open(files[0], 'w').close()
    counter = 1
    for obj in tasks:
        with open(files[-1], 'a') as file:
            file.write(json.dumps(obj) + '\n')
        if os.path.getsize(files[-1])/1024/1024 > 95 :
            counter += 1
            files.append(file_name+f'_{counter}.jsonl')
            open(files[-1], 'w').close()

    print('Batch prepared')
    batch_jobs = []
    for file in files:
            
        batch_file = client.files.create(
            file=open(file, "rb"),
            purpose="batch"
        )

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_jobs.append(batch_job)
    print('Batch sent')
        
