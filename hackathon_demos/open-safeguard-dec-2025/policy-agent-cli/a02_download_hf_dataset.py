"""Download the OpenAI Moderation API Evaluation dataset from Hugging Face."""

from datasets import load_dataset
import os

# Define the dataset and output directory
dataset_name = "mmathys/openai-moderation-api-evaluation"
output_dir = "./policy_pak/data"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading dataset: {dataset_name}")
print(f"Output directory: {output_dir}")

# Load the dataset
dataset = load_dataset(dataset_name)

print(f"\nDataset loaded successfully!")
print(f"Dataset splits: {list(dataset.keys())}")

# Print dataset information
for split_name, split_data in dataset.items():
    print(f"\n{split_name} split:")
    print(f"  Number of examples: {len(split_data)}")
    print(f"  Features: {split_data.features}")

    # Show a sample
    if len(split_data) > 0:
        print(f"\n  Sample from {split_name}:")
        print(f"  {split_data[0]}")

# Save the dataset to disk
print(f"\nSaving dataset to: {output_dir}")
dataset.save_to_disk(output_dir)

print("\nDataset downloaded and saved successfully!")
print(f"You can load it later with: dataset = load_from_disk('{output_dir}')")
