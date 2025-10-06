import argparse
import os
import yaml
import json
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from sycophancy.parser import parse_confidence
from sycophancy.confidence import compute_confidence

parser = argparse.ArgumentParser()
parser.add_argument("--setting_config", type=str, required=True)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--configs-folder", type=str, default="configs/")
parser.add_argument("--parse-confidence", action='store_true')
parser.add_argument("--logit-stats", action='store_true')


def process_problem(problem_path, judge_config, solver_config, solver_name = None, calculate_confidence = False, logit_stats = None):
    with open(problem_path, 'r') as f:
        problem_data = json.load(f)

    logit_metrics = {}
    if logit_stats:
        try:
            logits = np.load(os.path.join('data/logits', f"{'/'.join(problem_path.split('/')[-3:-1])}-{problem_data['problem_id']}.npy"))
        except Exception as e:
            logger.error(f'Found error {e} when accessing {os.path.join('data/logits', '/'.join(problem_path.split('/')[-3:-1]), f"{problem_data['problem_id']}.npy")}')
        logit_metrics = compute_confidence(logits)
    
    return {
        "judge": judge_config['human_readable_id'],
        "solver": solver_config['human_readable_id'] if solver_name is None else solver_name,
        "solver_id": '/'.join(problem_path.split('/')[-3:-1]),
        "problem": problem_data['problem_id'],
        "competition": problem_data['problem_id'].split('_')[0],
        "true_grade": problem_data['majority_vote'],
        "split": "matharena" if "matharena" in problem_path else "putnam" if "putnam" in problem_path else "generic",
        "cost": problem_data.get('solution_cost', 0),
        "confidence": parse_confidence(problem_data.get('full_solution', None)) if calculate_confidence else None,
        **logit_metrics
    }

if __name__ == '__main__':
    args = parser.parse_args()
    output_dir = os.path.join(args.output_folder, args.setting_config.replace('.yaml', ''))
    results = []
        
    output_path = Path(output_dir)
    config_path = Path(args.configs_folder)

    for problem_path in tqdm(output_path.glob('*/*/*/*/*.json')):
        judge_api, judge_model, solver_api, solver_model, problem = problem_path.parts[-5:]
        if os.path.exists(config_path / "models" / judge_api / f'{judge_model}.yaml'):
            with open(config_path / "models" / judge_api / f'{judge_model}.yaml', 'r') as f:
                judge_config = yaml.safe_load(f)
            
            
        if ' ' in solver_model and os.path.exists(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model.split(' ')[0]}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)
            results.append(process_problem(str(problem_path), judge_config, solver_config, solver_name=solver_model, calculate_confidence=args.parse_confidence, logit_stats=args.logit_stats))

        elif os.path.exists(config_path / "models" / solver_api / f'{solver_model}.yaml'):
            with open(config_path / "models" / solver_api / f'{solver_model}.yaml', 'r') as f:
                solver_config = yaml.safe_load(f)

            results.append(process_problem(str(problem_path), judge_config, solver_config, calculate_confidence=args.parse_confidence, logit_stats=args.logit_stats))

    df = pd.DataFrame(results)
    df['incorrect'] = df['true_grade'].apply(lambda x: x == 'incorrect')
    df.to_csv(f'{args.setting_config}.csv')

    print("\n=== Overall Metrics ===")
    judge_stats = df.groupby(['solver', 'judge']).agg(
        hallucination_rate=('incorrect', 'mean'),
        cost=('cost', 'sum')
    ).reset_index()
    print(judge_stats.to_string(index=False))

    print("\n=== Overall Judge Statistics ===")
    judge_stats_category = df.groupby(['solver', 'true_grade', 'judge']).agg(
        avg_accuracy=('problem', 'count'),
        cost=('cost', 'sum')
    ).reset_index()
    print(judge_stats_category.to_string(index=False))

    model_map = {
        "Claude-4-Sonnet (Think)": '\\textsc{Claude 4 Sonnet}',
        "DeepSeek-R1 (05/28)": '\\textsc{DeepSeek R1 (05/28)}',
        "DeepSeek-R1-05-28": '\\textsc{DeepSeek R1 (05/28)}',
        "DeepSeek-R1-Distill-14B": '\\textsc{DeepSeek R1 14B}',
        "DeepSeek-R1-Qwen3-8B": '\\ronesmall{}',
        "OPC R1 8B": '\\opcrone{}',
        "discrete_long": '\\textsc{OPC R1 8B (long)}',
        "Naive Baseline": '\\textsc{Baseline}',
        "Qwen3-235B-A22B": '\\qwenbig',
        "Qwen3-32B": '\\textsc{Qwen3 32B}',
        "Qwen3-8B": '\\textsc{Qwen3 8B}',
        "Qwen3-4B (25/07)": '\\qwensmallfour{}',
        'gemini-2.5-flash (think)': '\\textsc{Gemini-Flash}',
        "gemini-2.5-pro": '\\textsc{Gemini-Pro}',
        'gpt-4.1': '\\textsc{GPT 4.1}',
        'o3 (high)': '\\textsc{o3}',
        'o4-mini (high)': '\\textsc{o4-mini}',
        'GPT-5-mini (low)': '\\gptfivemini{}~\\textsc{(low)}',
        'GPT-5-mini (medium)': '\\gptfivemini{}~\\textsc{(medium)}',
        'GPT-5-mini (minimal)': '\\gptfivemini{}~\\textsc{(minimal)}',
        'gpt-4.1-mini': '\\gptfourmini{}',
        'DeepSeek-v3.1 (Think)': '\\dsvthreeone{}',
        'GPT OSS 120B (high)': '\\gptoss{}',
        'GPT-5 (high)': '\\gptfive{}',
        'Grok 4': '\\grokfour{}',
        'Grok 4 Fast': '\\grokfourfast{}'
    }

    latex_df = judge_stats[judge_stats.judge == 'GPT-5-mini (medium)'][["solver", "hallucination_rate", "cost"]].copy()
    latex_df.columns = ["Model", "Hallucination rate", "Cost"]

    # Format float values
    latex_df['Model'] = latex_df['Model'].apply(lambda x: model_map[x])
    latex_df["Hallucination rate"] = (latex_df["Hallucination rate"]*100).map("{:.1f}".format)
    latex_df["Cost"] = latex_df["Cost"].map("{:,.2f}".format)
    latex_df = latex_df.sort_values(by='Hallucination rate', ascending=True)
    # Generate LaTeX
    latex_table = latex_df.to_latex(index=False, escape=False, column_format='lccc', caption="Judge Performance Summary", label="tab:judge_summary")

    print(latex_table)

    