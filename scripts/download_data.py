import os
import shutil
import git
import zipfile


def clone_repository(url, branch="main", clone_to="./data"):
    # Ensure the target directory exists
    if os.path.exists(clone_to):
        shutil.rmtree(clone_to)
    os.makedirs(clone_to)

    # Clone the repository
    print(f"Cloning the repository from branch '{branch}'...")
    git.Repo.clone_from(url, clone_to, branch=branch)
    print("Done!")

    # Unzip zip files directly to clone_to folder
    for root, _, files in os.walk(clone_to):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(clone_to)
                os.remove(zip_path)
                print(f"Unzipped and deleted: {zip_path}")

    print(f"Finished extracting files to: {clone_to}")


url = "https://huggingface.co/datasets/rayeeli/EMAP.git"
branch = "main"
clone_to = "./data"

clone_repository(url, branch, clone_to)
