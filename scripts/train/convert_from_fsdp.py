import os
import shutil
import argparse
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

def is_fsdp_checkpoint(path: str) -> bool:
    fsdp_bin_file = os.path.join(path, "pytorch_model_fsdp.bin")
    
    return os.path.exists(fsdp_bin_file)

def consolidate_checkpoint(checkpoint_path: str, base_model_path: str):
    accelerator = Accelerator()
    accelerator.print("-" * 80)
    accelerator.print(f"Processing checkpoint: {checkpoint_path}")

    if not is_fsdp_checkpoint(checkpoint_path):
        accelerator.print("Skipping: Does not appear to be an FSDP checkpoint.")
        return

    tmp_output_path = f"{checkpoint_path}_consolidated_tmp"
    backup_path = f"{checkpoint_path}_old"

    if accelerator.is_main_process:
        if os.path.exists(tmp_output_path):
            shutil.rmtree(tmp_output_path)
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)

    try:
        if accelerator.is_main_process:
            accelerator.print(f"Preparing clean directory from base model '{base_model_path}'...")
            
            clean_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            clean_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            
            clean_model.save_pretrained(tmp_output_path, safe_serialization=True)
            clean_tokenizer.save_pretrained(tmp_output_path)
            
            del clean_model
            del clean_tokenizer
            torch.cuda.empty_cache()
            accelerator.print("Clean directory prepared successfully.")

        accelerator.wait_for_everyone()

        model_for_loading = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        dummy_optimizer = torch.optim.AdamW(model_for_loading.parameters(), lr=1e-5)
        
        model_for_loading, dummy_optimizer = accelerator.prepare(model_for_loading, dummy_optimizer)

        accelerator.print("-> Loading FSDP sharded weights...")
        accelerator.state.step = 0
        accelerator.load_state(checkpoint_path)
        accelerator.print("-> Checkpoint weights loaded successfully.")
        
        accelerator.print(f"-> Saving consolidated bfloat16 weights to: {tmp_output_path}")
        accelerator.save_model(
            model=model_for_loading,
            save_directory=tmp_output_path,
            safe_serialization=True
        )
        accelerator.print("-> Consolidated weights saved.")
        
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.print("-> Performing safe replacement of original checkpoint...")
            os.rename(checkpoint_path, backup_path)
            os.rename(tmp_output_path, checkpoint_path)
            shutil.rmtree(backup_path)
            accelerator.print(f"-> Successfully consolidated and replaced checkpoint: {checkpoint_path}")

    except Exception as e:
        accelerator.print("\n" + "=" * 30 + " ERROR " + "=" * 30)
        accelerator.print(f"Failed to process checkpoint: {checkpoint_path}")
        accelerator.print("Original files have been left intact.")
        accelerator.print(f"Error: {e}")
        accelerator.print(traceback.format_exc())
        accelerator.print("=" * 67 + "\n")
        
        if accelerator.is_main_process:
            if os.path.exists(tmp_output_path):
                shutil.rmtree(tmp_output_path)
            if os.path.exists(backup_path) and not os.path.exists(checkpoint_path):
                os.rename(backup_path, checkpoint_path)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and consolidate all Trainer FSDP checkpoints in a directory tree."
    )
    parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
    )
    parser.add_argument(
        '--n_checkpoints',
        type=int,
        default=-1
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Starting consolidation process in root folder: {args.root_folder}")

    checkpoints_to_process = []
    for root, dirs, _ in os.walk(args.root_folder):
        for dir_name in sorted(dirs):
            if dir_name.startswith("checkpoint-"):
                checkpoints_to_process.append(os.path.join(root, dir_name))
    
    if not checkpoints_to_process:
        accelerator.print("No 'checkpoint-*' directories found.")
        exit(0)

    if args.n_checkpoints != -1:
        checkpoints_to_process = checkpoints_to_process[:args.n_checkpoints]

    accelerator.print(f"\nFound {len(checkpoints_to_process)} checkpoint(s) to process:")
    for cp in checkpoints_to_process:
        accelerator.print(f"- {cp}")

    for checkpoint_path in checkpoints_to_process:
        consolidate_checkpoint(checkpoint_path, args.base_model_path)
        torch.cuda.empty_cache()

    accelerator.print("\nConsolidation process finished.")