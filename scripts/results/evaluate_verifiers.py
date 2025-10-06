import argparse
import os
import yaml
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--setting_config", type=str, required=True)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--configs-folder", type=str, default="configs/")
parser.add_argument("--judged-folder", type=str, default="data/judged/sycophancy_recent/")

def map_judgments(judgments):
    return [1 if j in ['correct', 'detected'] else 0 for j in judgments]

def process_problem(problem_path, judge_config, solver_config, solver_name = None, judged_gt = None):
    with open(problem_path, 'r') as f:
        problem_data = json.load(f)
    judgement = problem_data['judgements']
    
    judged_gt = None if judged_gt is None else 1 if judged_gt in [1,3] else 0

    return {
        "judge": judge_config['human_readable_id'],
        "solver": solver_config['human_readable_id'] if solver_name is None else solver_name,
        "problem": problem_data['problem_id'],
        "competition": problem_data['problem_id'].split('_')[0],
        "judgement": judgement,
        "majority_vote": problem_data['majority_vote'],  
        "true_grade": judged_gt,
        "split": "matharena" if "matharena" in problem_path else "putnam" if "putnam" in problem_path else "generic",
        "pass_at_1": np.mean([judged_gt == judgement for judgement in map_judgments(problem_data['judgements'])]),
        "maj_at_n": map_judgments([problem_data['majority_vote']])[0] == judged_gt,
        "cost": problem_data['cost']['cost']
    }

if __name__ == '__main__':
    args = parser.parse_args()
    output_dir = os.path.join(args.output_folder, args.setting_config.replace('.yaml', ''))
    results = []
        
    output_path = Path(output_dir)
    config_path = Path(args.configs_folder)
    judged_folder = Path(args.judged_folder)
    judged_gt = {}
    for problem_path in tqdm(judged_folder.glob('*/*/*.json')):
        with open(problem_path, 'r') as f:
            data = json.load(f)
        judge_id, comp, problem = problem_path.parts[-3:]
        problem_id = '_'.join([comp, problem])
        for attempt in data['attempts']:
            if problem_id not in judged_gt:
                judged_gt[problem_id] = {}
            judged_gt[problem_id][attempt['model_id']] = attempt['grading']['score']
    for problem_path in tqdm(output_path.glob('*/*/*/*/*.json')):
        judge_api, judge_model, solver_api, solver_model, problem_id = problem_path.parts[-5:]
        if os.path.exists(config_path / "models" / judge_api / f'{judge_model}.yaml'):
            with open(config_path / "models" / judge_api / f'{judge_model}.yaml', 'r') as f:
                judge_config = yaml.safe_load(f)

        if ' ' in solver_model and os.path.exists(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)
            problem = process_problem(str(problem_path), judge_config, solver_config, solver_name=solver_model, judged_gt=judged_gt.get(problem_id, {}).get(f'{solver_api}/{solver_model}', None))

            if problem['true_grade'] is not None:
                results.append(problem)

        elif os.path.exists(config_path / "models" / solver_api / f'{solver_model}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)
            problem = process_problem(str(problem_path), judge_config, solver_config, judged_gt=judged_gt.get(problem_id, {}).get(f'{solver_api}/{solver_model}', None))

            if problem['true_grade'] is not None:
                results.append(problem)

    df = pd.DataFrame(results)

    print("\n=== Overall Judge Statistics ===")
    judge_stats = df.groupby(['judge']).agg(
        avg_accuracy=('pass_at_1', 'mean'),
        maj_voting_accuracy=('maj_at_n', 'mean'),
        cost=('cost', 'sum')
    ).reset_index()
    print(judge_stats.to_string(index=False))

    model_map = {
        "Claude-4-Sonnet (Think)": '\\textsc{Claude 4 Sonnet}',
        "DeepSeek-R1 (05/28)": '\\textsc{DeepSeek R1 (05/28)}',
        "DeepSeek-R1-05-28": '\\textsc{DeepSeek R1 (05/28)}',
        "DeepSeek-R1-Distill-14B": '\\textsc{DeepSeek R1 14B}',
        "DeepSeek-R1-Qwen3-8B": '\\ronesmall{}',
        "OPC R1 8B": '\\opcrone{}',
        "discrete_long": '\\textsc{OPC R1 8B (long)}',
        "Naive Baseline": '\\textsc{Baseline}',
        "Qwen3-235B-A22B": '\\textsc{Qwen3 235B-A22B}',
        "Qwen3-32B": '\\textsc{Qwen3 32B}',
        "Qwen3-8B": '\\textsc{Qwen3 8B}',
        'gemini-2.5-flash (think)': '\\textsc{Gemini-Flash}',
        "gemini-2.5-pro": '\\textsc{Gemini-Pro}',
        'gpt-4.1': '\\textsc{GPT 4.1}',
        'o3 (high)': '\\textsc{o3}',
        'o4-mini (high)': '\\textsc{o4-mini}',
        'GPT-5-mini (low)': '\\gptfivemini{}~\\textsc{(low)}',
        'GPT-5-mini (medium)': '\\gptfivemini{}~\\textsc{(medium)}',
        'GPT-5-mini (minimal)': '\\gptfivemini{}~\\textsc{(minimal)}',
        'gpt-4.1-mini': '\\gptfourmini{}',
    }

    latex_df = judge_stats[["judge", "avg_accuracy", "maj_voting_accuracy", "cost"]].copy()
    latex_df.columns = ["Judge", "pass@1", "maj@3", "Cost"]

    # Format float values
    latex_df['Judge'] = latex_df['Judge'].apply(lambda x: model_map[x])
    latex_df["pass@1"] = (latex_df["pass@1"]*100).map("{:.1f}".format)
    latex_df["maj@3"] = (latex_df["maj@3"]*100).map("{:.1f}".format)
    latex_df["Cost"] = latex_df["Cost"].map("{:,.2f}".format)
    latex_df = latex_df.sort_values(by='pass@1', ascending=False)
    # Generate LaTeX
    latex_table = latex_df.to_latex(index=False, escape=False, column_format='lccc', caption="Judge Performance Summary", label="tab:judge_summary")

    print(latex_table)
