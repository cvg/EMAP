import os
import zipfile
import gdown


def download_and_unzip_google_drive_files(paths, download_to="./data"):
    if not os.path.exists(download_to):
        os.makedirs(download_to)

    for path in paths:
        # Convert Google Drive link to direct download link
        file_id = path.split("/d/")[1].split("/")[0]
        direct_link = f"https://drive.google.com/uc?id={file_id}"

        print(f"Downloading from {direct_link}...")
        output_path = os.path.join(download_to, f"{file_id}.zip")
        gdown.download(direct_link, output_path, quiet=False)
        print(f"Downloaded {output_path}")

        # Unzip the downloaded file
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(download_to)
        os.remove(output_path)
        print(f"Unzipped and deleted: {output_path}")

    print(f"Finished extracting files to: {download_to}")


# Google Drive file paths
paths = [
    "https://drive.google.com/file/d/17aUcCJCP5vgARs237H0BtlRoms5-CR6e/view?usp=sharing",
    "https://drive.google.com/file/d/1eZZiMcTfoiYfIxtv4Wy3lQYAudZpKlE0/view?usp=sharing",
    "https://drive.google.com/file/d/1pum-25MEFhXQu1fZLy_f9lRMBxvF1ssm/view?usp=sharing",
]

download_to = "./data"
download_and_unzip_google_drive_files(paths, download_to)
