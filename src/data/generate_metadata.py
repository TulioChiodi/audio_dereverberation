import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(df, test_size, val_size, stratify_col):
    # Split the dataset into train and test sets, stratified by the target label
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[stratify_col], random_state=42)

    # Split the train set into train and validation sets, stratified by the target label
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df[stratify_col], random_state=42)

    # Return the split datasets
    return train_df, val_df, test_df

dataset_size = "large"
test_size = 500
val_size = 0.1

noisy_data_dirpath = Path(f"data/raw/{dataset_size}/wavs/noisy/")
clean_data_dirpath = Path(f"data/raw/{dataset_size}/wavs/clean/")

noisy_filepaths = noisy_data_dirpath.glob("*.wav")
noisy_filenames = [filepath.name for filepath in noisy_filepaths]
clean_filenames = [f"{'_'.join(filename.split('_')[:2])}.wav" for filename in noisy_filenames]
anechoic_event = [filename.split("_")[0] for filename in noisy_filenames]

df = pd.DataFrame({"noisy_filename": noisy_filenames, "clean_filename": clean_filenames, "anechoic_event": anechoic_event})
stratify_col = "anechoic_event"

train_df, val_df, test_df = split_dataset(df, test_size, val_size, stratify_col)

# Print the sizes of the split datasets
print("Train set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))

train_df.to_csv(f"data/raw/{dataset_size}/train_metadata.csv", index=False)
val_df.to_csv(f"data/raw/{dataset_size}/val_metadata.csv", index=False)
test_df.to_csv(f"data/raw/{dataset_size}/test_metadata.csv", index=False)