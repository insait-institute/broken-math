import argparse
import json
import os
from pathlib import Path
import shutil
from decimal import Decimal
import ijson
import numpy as np

def json_serializer(obj):
    """
    A custom serializer for json.dump to handle types it doesn't know,
    like Decimal objects produced by ijson.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def process_samples(project: str):

    base_data_dir = Path("data")
    input_file = base_data_dir / "postprocess" / project / "test_samples.json"
    temp_output_file = input_file.with_suffix(".json.tmp")
    logits_dir = base_data_dir / "logits"

    print(f"Starting processing for project: '{project}'")
    print(f"Input file: {input_file}")
    
    logits_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logits will be saved in: {logits_dir}")

    try:
        with open(input_file, 'rb') as fin, open(temp_output_file, 'w') as fout:
            fout.write('[')
            is_first_entry = True
            json_stream = ijson.items(fin, 'item')
            for entry in json_stream:
                if not is_first_entry:
                    fout.write(',')
                else:
                    is_first_entry = False

                model_id = entry["model_id"]
                problem_id = '_'.join(entry["problem_id"].split('/'))


                logprobs_data = entry["cost"]["logprobs"]
                
                logits_data = [[item[2] for item in matrix] for matrix in logprobs_data]
                min_len = min([len(e) for e in logits_data])
                logits_data = [e[:min_len] for e in logits_data]

                try:
                    logits_to_save = np.array(logits_data, dtype=np.float32)
                except:
                    breakpoint()
                logit_filename = logits_dir / f"{model_id}-{problem_id}.npy"
                os.makedirs(logits_dir / model_id.split('/')[0], exist_ok=True)
                np.save(logit_filename, logits_to_save)

                del entry["cost"]["logprobs"]

                json.dump(entry, fout, default=json_serializer)

            fout.write(']')

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        if temp_output_file.exists():
            os.remove(temp_output_file)
        raise

    shutil.move(str(temp_output_file), str(input_file))
    print("\nProcessing complete.")
    print(f"Successfully cleaned and replaced {input_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",type=str)

    args = parser.parse_args()
    process_samples(args.project)