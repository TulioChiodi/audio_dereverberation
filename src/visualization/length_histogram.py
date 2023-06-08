import os
import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set the directory path where the WAV files are stored
# csv_filepath = "reports/durations_large_noisy.csv"
csv_filepath = "reports/durations_large_clean.csv"

df = pd.read_csv(csv_filepath)

# Plot a histogram of the durations
durations = df["duration"].tolist()
plt.hist(durations, bins=50)
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.title('Histogram of WAV file durations')
plt.show()
