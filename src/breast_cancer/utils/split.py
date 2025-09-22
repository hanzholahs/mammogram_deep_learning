import pandas as pd

def extract_split_file(filepath: str):
    df = pd.read_csv(filepath)
    image_paths = df["filepath"].values
    image_labels = df["label"].values
    return image_paths, image_labels

