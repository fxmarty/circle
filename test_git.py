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