import os
import re
import random
import pandas as pd
import numpy as np
from datasets import load_dataset, concatenate_datasets
from sycophancy.data_collection.sampling import stratified_sample_with_redistribution
from sycophancy.data_collection.processing import is_valid_numerical_latex
from typing import Tuple

COLUMN_LIST = ['problem_id', 'problem', 'solution', 'gold_answer', 'question_type', 'is_adversarial', 'needs_rephrasing']

def replace_prove_disprove(text: str, replacements: str) -> str:

    def replacer(match):
        # Get the original matched phrase (e.g., "Prove or Disprove")
        original_phrase = match.group(0)
        # Randomly select the new base word
        new_word = random.choice(replacements)

        # Apply capitalization based on the first word of the original match
        if original_phrase.split()[0].isupper():
            return new_word.upper()  # e.g., PROVE -> SHOW
        elif original_phrase.split()[0].istitle():
            return new_word.capitalize()  # e.g., Prove -> Show
        else:
            return new_word  # e.g., prove -> show

    return re.sub(r'prove or disprove', replacer, text, flags=re.IGNORECASE)

def replace_prove(text: str) -> str: return replace_prove_disprove(text, ['prove', 'show'])

def process_deeptheorem(data_path:str, n_samples:int=-1, sycophancy_frac:float=0.5) -> pd.DataFrame:
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        data = load_dataset(data_path)['train'].to_pandas()
    
    sample = stratified_sample_with_redistribution(data, n_samples, source_col='source')
    sample['problem_id'] = sample.apply(lambda x: f'deeptheorem_{x['source']}_{x['id']}',axis = 1)
    sample['question_type'] = 'proof'
    sample['gold_answer'] = None
    sample['needs_rephrasing'] = False

    adversarial = sample.sample(frac=sycophancy_frac)
    adversarial['problem'] = adversarial['neg'].apply(lambda x: replace_prove(x['question']))
    adversarial['solution'] = adversarial['neg'].apply(lambda x: x['response'])
    adversarial['is_adversarial'] = True

    
    standard = sample[~sample.index.isin(adversarial.index)]
    standard['problem'] = standard['pos'].apply(lambda x: replace_prove(x['question']))
    standard['solution'] = standard['pos'].apply(lambda x: x['response'])
    standard['is_adversarial'] = False

    standard = standard[COLUMN_LIST]
    adversarial = adversarial[COLUMN_LIST]
    return pd.concat([standard, adversarial])

def process_ours(data_path, n_samples=-1, sycophancy_frac=0.5):
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        ds = load_dataset(data_path)
        data = concatenate_datasets([ds[k] for k in ds]).to_pandas()
    data = data[['id', 'statement', 'solution', 'answer', 'competition', 'answer_type']]
    data['problem_id'] = data['id'].apply(lambda x: 'ours_'+str(x))
    data = data[data.answer.apply(lambda x: pd.isna(x) or is_valid_numerical_latex(x))]
    data['question_type'] = data.answer.apply(lambda x: 'proof' if pd.isna(x) else 'answer')
    sample = stratified_sample_with_redistribution(data, n_samples=n_samples, source_col='question_type')
    sample['gold_answer'] = sample['answer']
    sample['problem'] = sample['statement']

    adversarial = sample.sample(frac=sycophancy_frac)
    adversarial['needs_rephrasing'] = True
    adversarial['is_adversarial'] = True

    
    standard = sample[~sample.index.isin(adversarial.index)]
    standard['is_adversarial'] = False
    standard['needs_rephrasing'] = False

    standard = standard[COLUMN_LIST]
    adversarial = adversarial[COLUMN_LIST]
    return pd.concat([standard, adversarial])

def process_numina(data_path, n_samples=-1, sycophancy_frac=0.5):
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        data = load_dataset(data_path)['train'].to_pandas()
    data['problem_id'] = [f'numina_{i}' for i in range(len(data))]
    data = data[data.source.isin(['amc_aime', 'olympiads', 'aops_forum'])]
    data = data[~data.synthetic & (data.problem_is_valid == 'Yes') & (data.solution_is_valid == 'Yes')]
    data = data[data.answer.apply(lambda x: x=='proof' or is_valid_numerical_latex(x))]
    
    sample = stratified_sample_with_redistribution(data, n_samples, source_col='source')
    sample['question_type'] = sample['answer'].apply(lambda x: 'proof' if x=='proof' else 'answer')
    sample['gold_answer'] = sample['answer'].apply(lambda x: None if x=='proof' else x)

    adversarial = sample.sample(frac=sycophancy_frac)
    adversarial['needs_rephrasing'] = True
    adversarial['is_adversarial'] = True

    
    standard = sample[~sample.index.isin(adversarial.index)]
    standard['is_adversarial'] = False
    standard['needs_rephrasing'] = False

    standard = standard[COLUMN_LIST]
    adversarial = adversarial[COLUMN_LIST]
    return pd.concat([standard, adversarial])