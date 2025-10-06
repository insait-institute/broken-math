import argparse
from sycophancy.configs import load_config
from sycophancy.postprocess import process_sample, exclude_basic, join, exclude_discrepancy, train_test_split, random_train_test_split
import os
import json
import glob
from collections import defaultdict
import re
def reindex_multiple_solutions(samples):
    id_counter = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        sample['original_problem_id'] = sample['problem_id']
        if id_counter[sample['model_id']][sample['original_problem_id']] > 0:
            sample['problem_id'] += f'-part{id_counter[sample['model_id']][sample['original_problem_id']]}'
        id_counter[sample['model_id']][sample['original_problem_id']] += 1

def main():
    parser = argparse.ArgumentParser(description="Postprocess the grading results.")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--config-base", type=str, default="configs/projects")
    parser.add_argument("--exclude-regexes", type=str, nargs='*', default=[], help="Regexes of problem ids to exclude from postprocessing")
    parser.add_argument("--test-regexes", type=str, nargs='*', default=[], help="Regexes of problem ids to include in the test set")
    parser.add_argument("--save-folder", type=str, default="data/postprocess", help="Folder to save the postprocessed results")
    parser.add_argument("--random-sample", action='store_true')
    parser.add_argument("--hold-out-model", type=str, default='deepseek/r1_0528')
    parser.add_argument("--attempts-field", type=str, default='attempts')

    args = parser.parse_args()
    
    project_config = load_config(os.path.join(args.config_base, f"{args.project}.yaml"))

    project_name = args.project if args.attempts_field == 'attempts' else f'{args.project}-{args.attempts_field}'
    save_folder = os.path.join(args.save_folder, project_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_files = glob.glob(f"{project_config.solved_base_folder}/**/*.json", recursive=True)

    all_samples = []

    for file_name in all_files:
        samples = process_sample(file_name, attempts_field=args.attempts_field)
        all_samples.extend(samples)

    reindex_multiple_solutions(all_samples)

    print(f"Total samples: {len(all_samples)}")
    test_samples = all_samples
    print(f"Test samples: {len(test_samples)}")
    # save as jsonl files
    with open(os.path.join(save_folder, "test_samples.json"), "w") as f:
        json.dump(test_samples, f, indent=4)

if __name__ == "__main__":
    main()


