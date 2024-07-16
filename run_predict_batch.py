import os
import sys
import subprocess
from pathlib import Path

from utils import get_dataset_name

test_data_path = {
    "drone_tweets": "data/drone/masked_all_tweets.csv",
    "drone_reddit": "data/drone/responses/drone5_all_data.csv",
    "energy_reddit": "data/energy/responses/full_energy.csv"
}

# for each trained model
for item in Path("./checkpoint").iterdir():
    if item.is_dir():
        model_id = item.name
        if model_id != "Meta-Llama-3-8B-Instruct":
            continue

        # dataset = model_id.split("-")[0]
        
        # if ("4bit" in model_id) or ("lora" in model_id):        # split exp on gpus
            # continue
        # else:
            # continue
        print(model_id)
        # test_path = test_data_path[dataset]
        # dataset_name = get_dataset_name(test_path)
        dataset_name = "drone_tweets"
        test_path = test_data_path[dataset_name]
        # if "meld-plain" in model_id:
        #     test_path = "data/plain/meld/test.json"
        # elif "iemocap-plain" in model_id:
        #     test_path = "data/plain/iemocap/test.json"
        few_shots = True
        include_roles = True
        for combine_prompt in [True, False]:
            for rep_pen in [1.05, 1.1, 1.15, 1.2]:
                for max_new_tokens in [30, 50, 100]:
                    for temperature in [0.01, 0.1, 0.3, 0.5]:
                        out_dir = (f"output/{dataset_name}/{model_id}/max_new_toks={max_new_tokens}-temp={temperature}"
                                   f"-rep_pen={rep_pen}-combine_prompts={combine_prompt}-few_shots={few_shots}"
                                   f"-include_roles={include_roles}")
                        out_dir = Path(out_dir)
                        
                        pred_file = Path(out_dir / "predictions.csv")
                        if pred_file.exists(): 
                            continue

                        cmd = (f"python predict.py --model meta-llama/Meta-Llama-3-8B-Instruct --test_data {test_path} " 
                            f"--max_new_tokens {max_new_tokens} --temperature {temperature} "
                            f"--repetition_penalty {rep_pen} --combine_prompts={combine_prompt} "
                            f"--few_shots={few_shots} --output={str(out_dir)} --include_roles={include_roles} "
                            f"| tee {out_dir}/predict.log.txt")

                        print("cmd:", cmd)

                        out_dir.mkdir(exist_ok=True, parents=True)
                        subprocess.run(['bash', '-c', cmd])

        print("\n")

        # CUDA_VISIBLE_DEVICES=5 python run_predict_batch.py --model meta-llama/Meta-Llama-3-8B-Instruct --test_data data/drone/masked_all_tweets.csv