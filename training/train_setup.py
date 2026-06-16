import kagglehub
import os

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("asfakali2/iruvd-dataset-for-automatic-vehicle-detection")

print(f"Dataset downloaded to: {path}")

# List contents to understand structure
print("\nDataset structure:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(root)))
    subindent = ' ' * 4 * (level + 1)
    for f in files[:5]: # Only print first 5 files to avoid clutter
        print('{}{}'.format(subindent, f))
    if len(files) > 5:
        print(f'{subindent}... ({len(files)-5} more files)')




















