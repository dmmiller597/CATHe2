import torch
import numpy as np
from pathlib import Path
from src.data.data_module_ted import CATHeTEDDataModule

def test_data_loading():
    # Initialize the data module
    data_module = CATHeTEDDataModule(
        data_dir="data/TED",
        train_embeddings="prot_t5_embeddings_train.npz",
        val_embeddings="prot_t5_embeddings_val.npz",
        test_embeddings="prot_t5_embeddings_test.npz",
        batch_size=32
    )

    # Setup the data module
    data_module.setup()

    # Test dataset sizes
    assert len(data_module.datasets['train']) > 0, "Training dataset is empty"
    assert len(data_module.datasets['val']) > 0, "Validation dataset is empty"
    assert len(data_module.datasets['test']) > 0, "Test dataset is empty"
    
    # Test number of classes
    assert data_module.num_classes > 1, "Dataset should have multiple classes"
    assert isinstance(data_module.num_classes, int), "num_classes should be an integer"

    # Test embedding properties
    sample_embedding, sample_label = data_module.datasets['train'][0]
    assert isinstance(sample_embedding, torch.Tensor), "Embedding should be a torch.Tensor"
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"
    assert sample_embedding.dim() == 2, "Embedding should be 2-dimensional"
    assert not torch.isnan(sample_embedding).any(), "Embeddings contain NaN values"
    assert not torch.isinf(sample_embedding).any(), "Embeddings contain infinite values"

    # Test dataloader functionality
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Test batch properties
    batch = next(iter(train_loader))
    embeddings, labels = batch
    assert len(batch) == 2, "Batch should contain embeddings and labels"
    assert embeddings.shape[0] == labels.shape[0], "Batch size mismatch between embeddings and labels"
    assert embeddings.shape[0] <= data_module.batch_size, f"Batch size exceeds specified size of {data_module.batch_size}"

    # Test label properties
    unique_labels = torch.unique(data_module.datasets['train'].labels)
    assert len(unique_labels) > 1, "Dataset should have multiple unique labels"
    assert unique_labels.min() >= 0, "Labels should be non-negative"
    assert torch.all(labels < data_module.num_classes), "Labels should be less than num_classes"

    # Test label encoder
    label_encoder = data_module.datasets['train'].label_encoder
    assert hasattr(label_encoder, 'categories_'), "Label encoder should have categories"
    assert len(label_encoder.categories_[0]) == data_module.num_classes, "Label encoder categories should match num_classes"

    # Test dataset splits
    train_ids = set(data_module.datasets['train'].indices)
    val_ids = set(data_module.datasets['val'].indices)
    test_ids = set(data_module.datasets['test'].indices)
    
    assert len(train_ids.intersection(val_ids)) == 0, "Training and validation sets overlap"
    assert len(train_ids.intersection(test_ids)) == 0, "Training and test sets overlap"
    assert len(val_ids.intersection(test_ids)) == 0, "Validation and test sets overlap"

    # Print summary information
    print("\n=== Test Summary ===")
    print(f"✓ All {len(data_module.datasets['train'])} training samples validated")
    print(f"✓ All {len(data_module.datasets['val'])} validation samples validated")
    print(f"✓ All {len(data_module.datasets['test'])} test samples validated")
    print(f"✓ {data_module.num_classes} classes validated")
    print(f"✓ Embedding dimension: {sample_embedding.shape[1]}")
    print(f"✓ Batch size: {data_module.batch_size}")
    print("✓ All tests passed successfully!")

if __name__ == "__main__":
    test_data_loading()