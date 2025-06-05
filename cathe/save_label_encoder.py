import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import argparse

def save_label_encoder_from_training_data(labels_path: str, output_path: str):
    """Extract and save the label encoder from training labels.
    
    Args:
        labels_path: Path to training labels CSV file
        output_path: Path to save the label encoder pickle file
    """
    # Load training labels
    df = pd.read_csv(labels_path)
    
    # Fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['SF'])
    
    # Save label encoder
    with open(output_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Label encoder saved to {output_path}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)[:10]}...")  # Show first 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to training labels CSV")
    parser.add_argument("--output", default="label_encoder.pkl", help="Output pickle file")
    
    args = parser.parse_args()
    save_label_encoder_from_training_data(args.labels, args.output)
