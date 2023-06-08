import os
import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set the directory path where the WAV files are stored
# dataset_dirpath = "data/raw/large/wavs/noisy/"
# csv_filepath = "reports/durations_large_noisy.csv"

dataset_dirpath = "data/raw/large/wavs/clean/"
csv_filepath = "reports/durations_large_clean.csv"

# Get all the file names with .wav extension from the directory
wav_files = [file for file in os.listdir(dataset_dirpath) if file.endswith(".wav")]

# Define a function to get duration and filename from a WAV file
def get_duration_and_filename(file):
    try:
        with wave.open(dataset_dirpath + file, 'r') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
            return (file, duration)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return (file, 0.0)

# Use tqdm to show progress while getting durations and filenames
durations_and_filenames = []
for file in tqdm(wav_files):
    durations_and_filenames.append(get_duration_and_filename(file))

# Create a pandas DataFrame with the durations and filenames
df = pd.DataFrame(durations_and_filenames, columns=["filename", "duration"])

# Write the DataFrame to a CSV file
df.to_csv(csv_filepath, index=False)
