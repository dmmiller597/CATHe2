import numpy as np
import pandas as pd
from pathlib import Path

def filter_and_save_data():
    # Paths
    data_dir = Path("data/TED")
    
    # Load training labels to get valid label set
    train_labels = pd.read_csv(data_dir / "Y_Train_SF.csv")['SF'].values
    valid_labels = np.unique(train_labels)
    print(f"Number of unique training labels: {len(valid_labels)}")

    # Process validation and test sets
    for split in ["val", "test"]:
        # Load current data
        labels = pd.read_csv(data_dir / f"Y_{split.capitalize()}_SF.csv")['SF'].values
        embeddings = np.load(data_dir / f"prot_t5_embeddings_{split}.npz")['embeddings']
        
        # Filter data
        keep_idx = np.isin(labels, valid_labels)
        filtered_embeddings = embeddings[keep_idx]
        filtered_labels = labels[keep_idx]
        
        print(f"\n{split.upper()} set:")
        print(f"Original samples: {len(labels)}")
        print(f"Filtered samples: {len(filtered_labels)}")
        
        # Save filtered data
        np.savez(data_dir / f"prot_t5_embeddings_{split}_filtered.npz", embeddings=filtered_embeddings)
        # Create DataFrame with 'SF' as column name to maintain the original structure
        pd.DataFrame({'SF': filtered_labels}).to_csv(data_dir / f"Y_{split.capitalize()}_SF_filtered.csv", index=False)

if __name__ == "__main__":
    filter_and_save_data()