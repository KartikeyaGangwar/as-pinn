import zipfile
import os

zip_path = "as_pinn_kaggle_bundle.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

include_dirs = ["models", "physics", "benchmarks", "results", "manuscript", "tests", "data"]
include_files = ["train_two_stage_as_pinn.py", "kaggle_runner.ipynb", "requirements.txt", "README.md", "LICENSE"]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in include_files:
        if os.path.exists(f):
            zf.write(f, arcname=f)
    for d in include_dirs:
        for root, dirs, files in os.walk(d):
            if "__pycache__" in root or ".git" in root or ".pytest_cache" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".aux", ".log", ".out")):
                    continue
                full_path = os.path.join(root, file)
                zf.write(full_path, arcname=full_path)

print(f"[+] Successfully generated {zip_path} ({os.path.getsize(zip_path)/1e6:.2f} MB)")
