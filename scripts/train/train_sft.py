import os
import argparse
import yaml
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from functools import partial

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return sorted(list(lora_module_names - {'lm_head'}))

def setup_wandb(config):
    report_to = config['training_args'].get('report_to', 'none')
    if isinstance(report_to, str):
        report_to = [report_to]

    if "wandb" in report_to:
        if 'WANDB_API_KEY' not in os.environ:
            print("WARNING: 'wandb' is in `report_to` but WANDB_API_KEY environment variable is not set.")
            print("Disabling W&B logging.")
            report_to.remove("wandb")
        else:
            project_name = os.environ.get('WANDB_PROJECT', 'sycophancy-bench')
            os.environ['WANDB_PROJECT'] = project_name
            print(f"W&B integration is active. Logging to project: '{project_name}'")
    
    config['training_args']['report_to'] = report_to

def prepare_dataset(dataset, template, n_samples=-1, adv_frac=0.5, proof_frac=0.5):

    adversarial_problems = dataset.filter(lambda x: x.get("is_adversarial", False) is True, num_proc=os.cpu_count())
    non_adversarial_proof_problems = dataset.filter(
        lambda x: x.get("is_adversarial", True) is False and x.get("question_type") == "proof",
        num_proc=16
    )
    non_adversarial_answer_problems = dataset.filter(
        lambda x: x.get("is_adversarial", True) is False and x.get("question_type") == "answer",
        num_proc=16
    )

    desired_num_adv = int(len(dataset) * adv_frac)
    desired_num_proof = int(len(dataset) * (1 - adv_frac) * proof_frac)
    desired_num_answer = int(len(dataset) * (1 - adv_frac) * (1 - proof_frac))

    num_adv_to_take = min(desired_num_adv, len(adversarial_problems))
    num_proof_to_take = min(desired_num_proof, len(non_adversarial_proof_problems))
    num_answer_to_take = min(desired_num_answer, len(non_adversarial_answer_problems))

    sampled_datasets = []

    if num_adv_to_take > 0:
        sampled_datasets.append(adversarial_problems.shuffle(seed=42).select(range(num_adv_to_take)))
    if num_proof_to_take > 0:
        sampled_datasets.append(non_adversarial_proof_problems.shuffle(seed=42).select(range(num_proof_to_take)))
    if num_answer_to_take > 0:
        sampled_datasets.append(non_adversarial_answer_problems.shuffle(seed=42).select(range(num_answer_to_take)))

    final_dataset = concatenate_datasets(sampled_datasets)

    final_dataset = final_dataset.shuffle(seed=42)

    print(f"Total samples in final dataset: {len(final_dataset)}")

    final_dataset = final_dataset.map(lambda ex: formatting_prompts_func(ex, template))
    if n_samples != -1:
        final_dataset = final_dataset[:n_samples]

    return final_dataset

def formatting_prompts_func(example, template):
    messages = {
        "messages": [
            {"role": "user", "content": template.format(problem=example['problem'])},
            {"role": "assistant", "content": example['gold_answer']}
        ]
    }

    return messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with SFT and LoRA.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--adv_frac", type=float, default=0.5)
    parser.add_argument("--proof_frac", type=float, default=0.5)

    args = parser.parse_args()

    config = load_config(os.path.join('configs/training/', args.config))
    setup_wandb(config)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)

    generation_config = config.get('generation_config', {})


    dataset = load_dataset("json", data_files={
        "train": config['train_file'],
        "validation": config['validation_file'],
    })
    
    print(f"Training dataset size: {len(dataset['train'])}")
    print(f"Validation dataset size: {len(dataset['validation'])}")

    torch_dtype = getattr(torch, config.get('torch_dtype', 'bfloat16'))
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    
    if 'lora_config' in config:
        if not config['lora_config'].get('target_modules'):
            print("`target_modules` not specified in config. Finding all linear layers...")
            target_modules = find_all_linear_names(model)
            config['lora_config']['target_modules'] = target_modules
            print(f"Found target modules: {target_modules}")
        peft_config = LoraConfig(**config['lora_config'])
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    config['training_args']['output_dir'] = os.path.join(config['training_args']['output_dir'], f'adv_{args.adv_frac}-proofs_{args.proof_frac}/')

    training_arguments = SFTConfig(
        **config['training_args'],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=prepare_dataset(dataset['train'], template=config['prompt_template'], adv_frac=args.adv_frac, proof_frac=args.proof_frac),
        eval_dataset=prepare_dataset(dataset['validation'], template=config['prompt_template']),
        processing_class=tokenizer,
        callbacks=[eval_callback] if config.get('additional_eval', {}).get('do_final_answer', False) else [],
    )
    if config.get('additional_eval', {}).get('do_final_answer', False):
        eval_callback.set_trainer(trainer)

    print("Starting SFT training...")
    trainer.train()

    final_model_path = os.path.join(config['training_args']['output_dir'], "final_model")
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    
    print("Finished training.")