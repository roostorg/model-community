# Copyright 2025 Roblox Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fetch and preprocess Lex Fridman podcast transcripts for hate speech detection.

This script downloads the Lex Fridman podcast dataset from Hugging Face and processes it into a format
suitable for training a Sentinel hate speech detection model.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import argparse


def download_dataset(url, cache_dir):
    """
    Download the dataset if not already cached.

    Args:
        url (str): URL to download the dataset from.
        cache_dir (str): Directory to cache the downloaded file.

    Returns:
        str: Path to the downloaded file, or None if download failed.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(
            cache_dir, "lex-fridman-podcastUsing cached file.parquet"
        )

        if not os.path.exists(local_path):
            print(f"Downloading parquet file ...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete!")
        else:
            print(f"Using cached file at {local_path}")

        return local_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def process_segments(df):
    """
    Process segments from the dataset DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing podcast segments.

    Returns:
        list: List of episodes with processed segments.
    """
    podcast_data = []
    for _, row in df.iterrows():
        if "segments" in row:
            episode = {"segments": []}
            for segment in row["segments"]:
                if isinstance(segment, dict) and "text" in segment and segment["text"]:
                    episode["segments"].append({"text": segment["text"]})
            podcast_data.append(episode)
    return podcast_data


def load_lex_podcast_dataset():
    """
    Download and load the neutral podcast dataset from Hugging Face.

    Returns:
        list: A list of episodes, where each episode contains segments with text.
    """
    try:
        print("Loading neutral podcast dataset directly from Hugging Face...")
        url = (
            "https://huggingface.co/datasets/Whispering-GPT/lex-fridman-podcast/"
            "resolve/main/data/train-00000-of-00001-25f40520d4548308.parquet"
        )
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "neutral"
        )

        local_path = download_dataset(url, cache_dir)
        if not local_path:
            return None

        # Read the parquet file with pandas
        df = pd.read_parquet(local_path)
        print(f"Successfully loaded data with {len(df)} rows")

        # Process the segments
        podcast_data = process_segments(df)
        print(f"Processed {len(podcast_data)} episodes with segments")
        return podcast_data

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def collect_segment_texts(episode):
    """
    Extract text segments from a single episode.

    Args:
        episode (dict): Dictionary containing episode segments.

    Returns:
        list: List of text segments from the episode.
    """
    segment_texts = []
    if "segments" in episode and episode["segments"]:
        for segment in episode["segments"]:
            if isinstance(segment, dict) and "text" in segment and segment["text"]:
                segment_texts.append(segment["text"])
    return segment_texts


def extract_random_segments(dataset, num_segments):
    """
    Extract random segments from the dataset.

    Args:
        dataset (list): List of podcast episodes.
        num_segments (int): Number of segments to extract.

    Returns:
        list: List of text segments.
    """
    all_segments = []

    # Collect all segments from episodes
    for episode in tqdm(dataset, desc="Extracting segments"):
        all_segments.extend(collect_segment_texts(episode))

    print(f"Collected {len(all_segments)} segments before sampling")

    # Sample segments if necessary
    if len(all_segments) > num_segments:
        all_segments = random.sample(all_segments, num_segments)
        print(f"Randomly sampled {num_segments} segments")
    elif len(all_segments) < num_segments:
        print(
            f"Warning: Could only collect {len(all_segments)} segments, fewer than the requested {num_segments}"
        )

    return all_segments


def save_segments_to_csv(
    segments, output_path, filename="neutral-segments-training.csv"
):
    """
    Save training segments to a CSV file.

    Args:
        segments (list): List of text segments.
        output_path (str): Directory to save the CSV file.
        filename (str): Name of the output CSV file.
    """
    os.makedirs(output_path, exist_ok=True)
    df = pd.DataFrame(
        {
            "filename": ["lex_fridman_podcast"] * len(segments),
            "paragraph_segment": segments,
            "segment_id": range(len(segments)),
            "label": [0] * len(segments),  # 0 for negative/neutral examples
        }
    )
    output_file = os.path.join(output_path, filename)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(segments)} training segments to {filename}")


def save_episodes_for_eval(
    episodes, output_path, filename="neutral-episodes-eval.parquet"
):
    """
    Save entire episodes for evaluation in parquet format.

    Args:
        episodes (list): List of episode dictionaries.
        output_path (str): Directory to save the parquet file.
        filename (str): Name of the output parquet file.
    """
    os.makedirs(output_path, exist_ok=True)

    # Format episodes for saving
    eval_data = []
    for i, episode in enumerate(episodes):
        if "segments" in episode and episode["segments"]:
            segments = [
                seg["text"]
                for seg in episode["segments"]
                if "text" in seg and seg["text"]
            ]
            if segments:
                eval_data.append(
                    {
                        "episode_id": f"lex_fridman_episode_{i}",
                        "segments": segments,
                        "full_text": " ".join(segments),
                    }
                )

    df = pd.DataFrame(eval_data)
    output_file = os.path.join(output_path, filename)
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(eval_data)} evaluation episodes to {filename}")


def parse_args():
    """Parse the command line arguments and return parsed values."""
    parser = argparse.ArgumentParser(
        description="Download and process Lex Fridman podcast segments as neutral examples."
    )
    parser.add_argument(
        "-n",
        "--num-segments",
        type=int,
        default=15000,
        help="Number of segments to extract for training (default: 15000, typically 10x the number of positive examples)",
    )
    parser.add_argument(
        "-e",
        "--num-eval-episodes",
        type=int,
        default=30,
        help="Number of full episodes to extract for evaluation (default: 30)",
    )
    return parser.parse_args()


def main():
    """Main function to execute the data preparation pipeline."""
    args = parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Load the dataset
    dataset = load_lex_podcast_dataset()
    if not dataset:
        print("Failed to load dataset. Exiting.")
        return

    # Split dataset into training and evaluation
    if len(dataset) < args.num_eval_episodes:
        print(
            f"Warning: Only {len(dataset)} episodes available, using all for evaluation"
        )
        eval_episodes = dataset
        train_episodes = []
    else:
        # Randomly select episodes for evaluation
        eval_episodes = random.sample(dataset, args.num_eval_episodes)
        # Use remaining episodes for training
        train_episodes = [ep for ep in dataset if ep not in eval_episodes]

    # Extract training segments from remaining episodes
    if train_episodes:
        segments = extract_random_segments(train_episodes, args.num_segments)
        save_segments_to_csv(segments, str(data_dir))

    # Save evaluation episodes
    save_episodes_for_eval(eval_episodes, str(data_dir))


if __name__ == "__main__":
    main()
