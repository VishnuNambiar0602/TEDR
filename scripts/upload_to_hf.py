import os
from huggingface_hub import HfApi

# Configuration
REPO_ID = "Toosterpan/Tranformers"  # Replace with your HF repo ID
DATA_DIR = "temp_data_backup/data"
TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")  # Replace with your HF write token

def upload_dataset():
    api = HfApi()
    
    print(f"Starting upload of {DATA_DIR} to https://huggingface.co/datasets/{REPO_ID}")
    
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=TOKEN)
        
        # Upload the entire folder
        # This method supports resuming and is efficient for many small files
        api.upload_folder(
            folder_path=DATA_DIR,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=TOKEN
        )
        print("Upload successful!")
        
    except Exception as e:
        print(f"Error during upload: {e}")

if __name__ == "__main__":
    if "YOUR_USERNAME" in REPO_ID or "YOUR_HF_TOKEN" in TOKEN:
        print("Please edit upload_to_hf.py and set your REPO_ID and TOKEN.")
    else:
        upload_dataset()
