import os
import subprocess
import tempfile

if __name__ == "__main__":
    dir_path = tempfile.mkdtemp("evaluator_trainer_parity_test")

    print("dir_path", dir_path)

    transformers_version = "4.20.1"
    branch = ""
    if not transformers_version.endswith(".dev0"):
        branch = f"--branch v{transformers_version}"
    subprocess.run(
        f"git clone --depth 3 --filter=blob:none --sparse {branch} https://github.com/huggingface/transformers",
        shell=True,
        cwd=dir_path,
    )

    print("after git clone", os.listdir(dir_path))



    subprocess.run(
        "git sparse-checkout set examples/pytorch/text-classification",
        shell=True,
        cwd=os.path.join(dir_path, "transformers"),
    )

    print("after sparse-checkout", os.listdir(os.path.join(dir_path, "transformers")))

    model_name = "howey/bert-base-uncased-sst2"
    subprocess.run(
        f"python3 examples/pytorch/text-classification/run_glue.py"
        f" --model_name_or_path {model_name}"
        f" --task_name sst2"
        f" --do_eval"
        f" --max_seq_length 9999999999"  # rely on tokenizer.model_max_length for max_length
        f" --output_dir {os.path.join(dir_path, 'textclassification_sst2_transformers')}"
        f" --max_eval_samples 200",
        shell=True,
        cwd=os.path.join(dir_path, "transformers"),
    )

    print("after run_glue.py", os.listdir(dir_path))
    print("after run_glue.py", os.path.join(dir_path, 'textclassification_sst2_transformers'))

    with open(
        f"{os.path.join(dir_path, 'textclassification_sst2_transformers', 'eval_results.json')}", "r"
    ) as f:
        transformers_results = json.load(f)

    print(transformers_results)
