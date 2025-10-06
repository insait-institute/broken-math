from sycophancy.data_collection.gather_data import process_deeptheorem, process_numina, process_ours
import pandas as pd
import json
import os
import numpy as np
from argparse import ArgumentParser
DEEPTHEOREM_PATH = 'Jiahao004/DeepTheorem'
NUMINA_PATH = 'AI-MO/NuminaMath-1.5'
OURS_PATH = 'our_dataset.csv'



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--n_samples', type=int, default=15000)
    
    args = parser.parse_args()
    N_SAMPLES = args.n_samples

    data_deeptheorem = process_deeptheorem(DEEPTHEOREM_PATH, n_samples=N_SAMPLES//3, sycophancy_frac=0 if args.validation else 0.5)
    data_numina = process_numina(NUMINA_PATH, n_samples=N_SAMPLES//3, sycophancy_frac=0 if args.validation else 0.5)
    data_pm = process_ours(OURS_PATH, n_samples=N_SAMPLES//3, sycophancy_frac=0 if args.validation else 0.5)

    full_dataset = pd.concat([data_deeptheorem, data_numina, data_pm])
    shuffled_df = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    json_data = shuffled_df.to_dict(orient='records')

    for entry in json_data:
        entry['original_problem'] = entry['problem']
        try:
            if np.isnan(entry['gold_answer']):
                entry['gold_answer'] = None
        except:
            pass

    if args.validation:
        with open('./data/train/matharena_old.json', 'r') as f:
            matharena_old = json.load(f)

        for entry in matharena_old:
            entry['problem_id'] = entry['problem_idx']
            entry['original_problem'] = entry['problem']
            entry['solution'] = entry['gold_answer']
            entry['needs_rephrasing'] = False
            del entry['problem_idx']

        os.makedirs('./data/raw/validation', exist_ok=True)
        with open('./data/raw/validation/sample.json', 'w') as f:
            json.dump(json_data + matharena_old, f, indent=4) 
    else:
        os.makedirs('./data/raw/training_data', exist_ok=True)
        with open('./data/raw/training_data/sample_full.json', 'w') as f:
            json.dump(json_data, f, indent=4) 