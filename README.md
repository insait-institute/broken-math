<div align="center">
  <h1><img height="150px" src="./images/sycophancy_icon.png" alt="BrokenMath"><br>BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs</h1>
  <a href="https://www.python.org/">
    <img alt="Build" src="https://img.shields.io/badge/Python-3.12-1f425f.svg?color=blue">
  </a>
  <a href="https://opensource.org/license/apache-2-0">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache%20License%202.0-yellow.svg">
  </a>
  <a href="https://huggingface.co/datasets/INSAIT-Institute/BrokenMath">
    <img alt="BrokenMath Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BrokenMath-ffc107?color=ffc107&logoColor=white">
  </a>
</div>

---

## 👋 Overview

This repository contains the code for the paper *BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs*. This README explains how to:

* Execute our benchmark and training data pipeline, 
* Reproduce our experiments,
* Train the model as described in the paper.

---

## 🚀 Installation

You can use either **UV** or **Conda** to install and manage dependencies.

### Option 1: Install Using UV

UV is a fast Python package manager.

* **macOS & Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

* **Windows:**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

> If using UV, prepend all commands with `uv run`.

### Option 2: Using Conda

To use Conda:

```bash
conda create -n sycophancy python=3.12
conda activate sycophancy
python -m pip install -e .
```

This installs the project in editable mode.

---

## 🛢️ Creating your own Sycophantic Data

If you want to apply our perturbation framework on you problem set, begin by preparing a file with your problems. This can be either a `.csv` or `.json` file with the only requirement being that each entry/row must contain a `problem` and `solution` item. You can the run the script:

```bash
python ./scripts/data/rephrase_post.py <file_location>
```
For this step, we currently support the OpenAI Batch API, with the model and other parameters being adaptable in the script itself. The script will give you a set of batch ID(s), which you will need to retrieve the results after the batch completion. At that point, you can run.

```bash
python ./scripts/data/rephrase_get.py --batches <batch_ids> --dataset <file_location>
```
If you require more than 1 batches the **BATCH_ID** parameter supports a space-separated list of ids that should be given in the order of submission. **FILE_LOCATION** refers to the original file's location. The rephrased problem file will be found in the same directory as the original, with all relevant metadata retained.

## 🚧 Managing and Running Your Own Project

### Setting Up a Project

You need to define several config files. This repository already includes three fully-working example configurations that you can reference. See `src/sycophancy/config.py` for all possible config options, we only mention the most important ones here.

#### `configs/projects/{project_name}.yaml`

```yaml
name: [[Name of the Project, can be anything]]
admins:
  - [[Judge ID of an admin. How to define a Judge ID is explained later]]
  - ...

model_configs:
  - [[Path to model config to run for this project, excluding yaml and relative to configs/models, e.g., openai/o3]]
  - ...
```

#### `configs/solvers/{project_name}.yaml`

```yaml
prompt: [[Prompts for questions. Will be formatted using problem={problem_statement}, so make sure that any loose brackets {,} appear doubled, and that "{problem}" appears somewhere in the prompt]]
prompt_gold: [[Prompt for questions that contain a final answer that can be extracted. Similar instructions as above.]]
n_solutions: [[Number of solutions to sample per model]]
problem_regexes:
  - [[Regex that has to match the problem path, defaults to everything. Multiple regexes will match the union of all them]]
  - ...
allow_shuffle: [[Whether to shuffle the model answers in each problem (good for anonymity), but can break if you run best-of-n selection and do not run the solver all at once ]]
overwrite: [[Whether to overwrite existing results]]
```

To setup a solver config for best-of-n selection, you additionally have to specify the following keys:

```yaml
type: "best_of_n"
attempt_key: "attempts_singular"

n_solutions: [[Number of solutions to select from ]]

judges:
  - name: [[Name of the judge]]
    judge: [[Config of the judge model, relative to configs/models]]
    mode: [[Either "random", "discrete", "continuous", or "dual"]]
    prompt: [[Prompt of the judge. Will be formatted with problem={problem_statement}, solution={model_solution} if the mode is not dual. If the mode is dual, then it will be formatted with problem={problem_statement}, solution_a={model_solution_a}, solution_b={model_solution_b}. Your prompt should state that the model should format its final answer in \boxed{{}}.]]
    correct_word: [[If mode=discrete, this will be used to compare the models judgment with to see if it thinks the proof is correct]]
    tournament_type: [[If mode=dual, the type of tournament that is run between model answers. Either "swiss" or "bracket"]]
    tournament_config: [[If mode=dual, the config of the tournament]]
      - max_losses: [[The maximum number of losses in a bracket tournament before a solution is finally removed.]]
      - n_rounds: [[Number of rounds in the swiss tournament]]
      - first_win_word: [[The word the model is instructed to output if the first output wins.]]
      - second_win_word: [[The word the model is instructed to output if the second output wins.]]
  - ...
```
For the iterative agent, the relevant configurations will be explained in the `model` configurations.

#### `configs/settings/{project_name}.yaml`

```yaml
name: [[The project name]]
prompt_path: [[Path to the raw prompt you want to use for sycophantic verification]] 
data_path: [[Path to the posprocessed data (refer to the relevant section below)]] 
n_solutions: [[The number of solutions you want to verify. Set to -1 to include all.]]
```

#### `configs/judge_settings/{project_name}.yaml`

These configurations are identical to the ones in `configs/settings`, but are used for proof correctness verification, instead of sycophantic classification.

### 🤖 Setting up new models

#### `configs/models/{model_provider}/{model_name}.yaml`

```yaml
model: [[Model name to be forwarded to the relevant API]]
api: [[The model provider API name]]
max_tokens: [[The maximum output token limit]]
temperature: [[The generation temperature]]
top_p: [[The top-p sampling parameter]]
concurrent_requests: [[How many requests should be run in parallel]]
read_cost: [[The cost per million input tokens]]
write_cost: [[The cost per million output tokens]]
human_readable_id: [[With what name the results should be displayed for readability]]
```


You are free to include other relevant parameters that will be passed to the API directly. For more complex examples, such as the OpenAI API, you can look at the examples we provide in the relevant folders.

#### Iterative agent configuration
To set up the iterative agent, create a configuration for the base model as above, and then create one for the agent as below:

```yaml
type: agent
original_model_config: models/{model_provider}/{model_name}
agent_config: agent_scaffolds/standard
human_readable_id: [[With what name the results should be displayed for readability]]
```


#### Running models locally with vLLM

We support all 3 methods of running vLLM, which can be configured via the `api` field in the model configuration. The 3 different way to pass this parameter are as follows:

- `vllm`: An asynchronous vLLM engine that enables saving at the point of generating the solution. **NOTE**: for multi-sample runs, if left without any running queries, the engine is natively forced to shut down, which may cause the script to terminate prematurely.

- `vllm_sync`: A synchronous vLLM engine that will generate responses for all queries before returning them. This engine is the fastest of the 3, but no progress will be made in the event of a crash during generation.

- `vllm_server`: An asynchronous vLLM engine, with dynamic request processing, similar to sending queries to a proprietary API. Requires a running background vllm server, which can be launched with:

```bash
vllm serve <model_name> --dtype auto --api-key token-abc123 --tensor-parallel-size <n_gpus>
```
For models in the Qwen series, which require longer context to what is enabled in the configuration, serve the model with:
```bash
vllm serve <model_name> --dtype auto --api-key token-abc123 --tensor-parallel-size <n_gpus> --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072
```
---

## ➕ Adding New Problems

1. Problems must be formatted as a list of dictionaries in `data/raw/{filename}.json`:

```json
{
  "problem_id": "[[Problem ID. Anything before the first '_' will be used as competition name]]",
  "problem": "[[Problem Statement]]",
  "solution": "[[Ground Truth Solution]]",
  "gold_answer": "[[Ground truth answer, if one exists]]",
  "is_adversarial": "[[Whether the problem is perturbed from the original with a faulty premise]]",
  "question_type": "[['proof' or 'answer']]",
  "original_problem": "[[The original problem statement if rephrased, otherwise the same as 'problem']]:"
}
```

You can access the problems in BrokenMath from the `data/raw/sycophancy_recent` folder, and our model training data in `data/raw/training_data`

2. Process the problems

## ▶️ Processing the Unsolved Problems

After the dataset has been finalized, prepare it for the solution generation stage:

```bash
python scripts/process.py --project {project_name}
```

## ▶️ Running New Problems

Once configs and API keys are in place, generate model outputs:

```bash
python scripts/solve.py --project {project_name}
```
For running models locally (with the exception of the vLLM server setting), we recommend disabling the default parallel execution.

## ➡️ Postprocessing Results

Prepare final results for testing:

```bash
python scripts/postprocess.py --project {project_name}
```

---

## Solution Evaluation

### Split Solutions

We evaluate each type of problem depending on the task - sycophancy, final-answer, or proofs. If you are **only** running sycophantic verification, you can proceed to the next step. Otherwise, you need to run:

```bash
python ./scripts/verify_solutions.py --project <project_name>
```

### Sycophancy Verification

To evaluate the sycophantic behaviour, set up a configuration in `configs/setting`, and run:

#### (Optional) Removing log-probabilities from postprocessed data

Data containing the log-probabilities can consume a significant amount of space, and will take much longer to process. To alleviate this effect, you can run the following command to separate the log-probability data from the solutions, storing them in `data/logits`.

```bash
python ./scripts/data/reformat_logits.py --project <project_name>
```

#### Obtaining judgments

```bash
python ./scripts/check_solutions.py --checker_configs openai/gpt-5-mini-medium --setting_config <project_name> --n 3 --skip-existing
```

#### Calculating results
To obtain the results, run:
```bash
python ./scripts/results/sycophancy_results.py --setting_config <project_name>
```

To parse the self-confidence results, enable the `--parse-confidence` flag. To compute logit statistics, if available, enable the `--logit-stats` flag.


### Proof Verification

To evaluate the proof correctness of a solution set, set up a configuration in `configs/judge_setting`, and run:

#### Obtaining judgments

```bash
python ./scripts/judge_inference.py --judge_configs opc/opc_r1_8b --setting_config <project_name> --n 5 --skip-existing
```

#### Calculating results (including final-answer)
To obtain the results, run:
```bash
python ./scripts/results/utility_results.py --setting_config <project_name>
```

## 📊 Reproducing Results

We have included a summary of our results, available in the `data/results` folder. Using them, you can reproduce our results and figures using the Jupyter notebook:

```bash
notebooks/extraction.ipynb
```

---


## 🏋️‍♀️ Training a Model 
To finetune a model, we adapt the `trl` framework, for which you should run the following steps:

### Preparing the training data

1. To prepare the training data, first create a dataset using 

```bash
python scripts/data/prepare_training_data.py
```
where you can add your own dataset by modifying the **OURS_PATH** variable. We have not included our collected dataset due to size restrictions.

2. Rephrase the problems

Around half of the problems will be flagged for sycophantic rephrasing. Follow the procedure we describe in the relevant section, running the scripts with the `--training` tag enabled.

3. Create a project, generate and verify solutions

Follow the instructions for verification for the corresponding problem types to verify each problem type with its respectful grader.

4. Aggregate the data

For finalizing the dataset, run:

```bash
python scripts/finalize_verification.py --project <project_name>
```

### Editing the training configuration

You are free to setup your configuration or modify any hyperparameters in `configs/training`, according to the [transformers documentation](https://huggingface.co/docs/transformers/v4.56.2/en/index).

### Preparing an "Accelerate" configuration

For more effective multi-GPU training (or to enable other training methodologies), create an `accelerate` configuration by following the steps after running:

```bash
accelerate config --config_file <file_path>
```

A default configuration is provided in `configs/accelerate/`

### Running the training script

To run the training script, use the following command:

```bash
accelerate launch --config_file <accelerate_config_path> ./scripts/train/train_sft.py --config <config_path> --adv_frac <adv_frac> --proof_frac <proof_frac>
```

**ADV_FRAC** denotes what fraction of the dataset should comprise of abstention/correction examples.  **PROOF_FRAC** denotes what fraction of the *non-adversarial* samples should be proof-based. The dataset size will be adjusted with these parameters in mind.
