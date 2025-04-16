# Contrastive Learning for CATH Superfamily Classification

## Conceptual Overview
Contrastive learning is a self-supervised learning technique that learns representations by contrasting positive pairs against negative pairs. In our CATH superfamily classification project, we use triplet-based contrastive learning to create an embedding space where proteins from the same superfamily are close together, while proteins from different superfamilies are far apart.

## Core Components and Workflow

### 1. Data Representation
- **Input Data**: Protein embeddings (1024-dimensional vectors) from ProtT5
- **Labels**: CATH superfamily classifications 
- **Objective**: Learn projection from 1024D to 128D space where distance represents superfamily similarity

### 2. Triplet Formation
For each training iteration, we form triplets:
- **Anchor**: Reference protein embedding
- **Positive**: Different protein from same superfamily
- **Negative**: Protein from different superfamily

The TripletDataset class handles triplet formation by:
1. Identifying superfamily label for each anchor
2. Finding proteins with same label for positives
3. Selecting proteins with different labels for negatives

### 3. Mining Strategies
Three strategies for negative sample selection:
- **Random Mining**: Random negative selection (simplest)
- **Semi-Hard Mining**: Negatives farther than positives but producing non-zero loss
- **Hard Mining**: Closest negative samples (most challenging)

### 4. Neural Network Architecture
ContrastiveCATHeModel implements:
- Input layer (1024 neurons)
- Hidden layers [1024, 512] with LayerNorm, ReLU, Dropout
- Output layer (128 neurons) with L2 normalization

### 5. Loss Function: Triplet Loss
Maximizes anchor-negative distance while minimizing anchor-positive distance:
- d(a,p): Squared Euclidean distance between anchor-positive
- d(a,n): Squared Euclidean distance between anchor-negative  
- margin: Minimum separation (default 1.0)

### 6. Training Procedure
**Forward Pass**:
- Project triplets through network
- Apply L2 normalization

**Loss Calculation**:
- Compute distances
- Apply triplet loss with margin
- Backpropagate

**Validation**:
- Project validation embeddings
- Perform k-NN classification
- Monitor balanced accuracy

### 7. Inference Pipeline
For new proteins:
1. Project embedding into learned space
2. Find k nearest neighbors
3. Use weighted majority voting for prediction

## Implementation Details

### Key Classes
- **TripletDataset**: Handles data loading and triplet formation
- **ContrastiveDataModule**: Manages data splits and loading
- **TripletLoss**: Implements loss function
- **ContrastiveCATHeModel**: Neural network and training logic

### Hyperparameters
- Embedding dimension: 128
- Projection layers: [1024, 512]
- Margin: 1.0
- k neighbors: 5
- Mining strategy: Choice impacts training
- Batch size: 256

### Advantages
- Similarity-based classification
- Transfer learning capability
- Interpretable embeddings
- Few-shot learning support
- Natural confidence metric

### Challenges
- Triplet selection scaling
- Mining strategy overhead
- Class imbalance handling
- Embedding collapse prevention
- Complex evaluation needs





PROMPT:

# Updated Prompt for Curriculum Learning Implementation in CATH Classification

## Implement Curriculum Learning for CATH Hierarchy

Please create a professional implementation of curriculum learning for the contrastive learning CATH model. The implementation should:

1. Parse CATH superfamily labels (e.g., "1.2.3.4") into their hierarchical components:
   - Level 1: Class (C)
   - Level 2: Architecture (A)
   - Level 3: Topology (T)
   - Level 4: Homologous superfamily (H)

2. Develop a curriculum strategy that progressively trains on increasing levels of hierarchy:
   - Stage 1: Train on Class level only (e.g., "1.x.x.x")
   - Stage 2: Train on Class + Architecture (e.g., "1.2.x.x")
   - Stage 3: Train on Class + Architecture + Topology (e.g., "1.2.3.x")
   - Stage 4: Train on full CATH classification (e.g., "1.2.3.4")

3. Modify the ContrastiveDataModule to support curriculum learning by:
   - Adding a curriculum_stage parameter to control the current hierarchy level
   - Creating dynamic label mapping based on the current stage
   - Implementing a function to advance the curriculum when performance plateaus

4. Update the training pipeline (contrastive_train.py) to:
   - Track validation metrics at each curriculum stage
   - Implement criteria for advancing to the next curriculum stage
   - Save model checkpoints at stage transitions

5. Modify the ContrastiveCATHeModel to handle the changing label space across curriculum stages

6. Update the configuration file (contrastive.yaml) to include curriculum learning parameters

The implementation should be efficient, well-documented, and minimally invasive to the existing codebase structure.


Experimental Plan for Contrastive Learning Optimization
| Experiment ID | Focus Area | Target Level | Current Setup | Proposed Change / Experiment | Rationale / Expected Outcome | Key Files / Parameters | Evaluation Metrics |
| :------------ | :----------- | :------------------- | :------------------------------------------------- | :--------------------------------------------------------------- | :------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| C-1 | Sampling | C-Level (4 Classes) | WeightedRandomSampler (balances classes) | Implement/Use MPerClassSampler (e.g., P=4, K=256 for batch=1024) | Guarantee presence of positive pairs within batches; Improve triplet mining efficiency & stability. | contrastive_data_module.py::_get_dataloader, potentially add m_per_class to contrastive.yaml | train/active_triplets, train/loss, val/knn_*, val/dist_* metrics, convergence speed |
| C-2 | Mining | C-Level (4 Classes) | SemiHardMiner with SoftTripletLoss | Switch to BatchHardMiner with SoftTripletLoss | Selects hardest negatives overall; Potentially faster convergence, but risk of outlier sensitivity. | contrasted.py::__init__ (miner selection) | train/active_triplets, train/loss, val/knn_*, val/dist_* metrics, convergence speed |
| C-3 | Loss | C-Level (4 Classes) | soft_triplet_loss (no margin) | Switch to nn.TripletMarginLoss (requires margin param) | Enforce explicit separation margin; Potentially create better-defined clusters. | contrasted.py::__init__ (loss_fn, add margin), contrastive.yaml (add model.margin), training_step | val/knn_*, val/dist_margin, val/class_overlap, val/triplet_violation_rate |
| C-4 | Metrics | C-Level (4 Classes) | kNN Acc, Distances, t-SNE, Uniformity, Violation Rate | Add Recall@K (e.g., R@1, R@5) and mAP calculation | Directly measure retrieval performance and ability to find related items in embedding space. | contrasted.py::on_validation_epoch_end (implement retrieval metric calculation) | New metrics: val/recall_at_k, val/map |
| C-5 | Metrics | C-Level (4 Classes) | (As above) | Correlate embedding distance vs. structural similarity | Quantify if the learned space truly captures biological structural relationships (Goal c). | Requires external structural similarity data (e.g., TM-scores) and analysis script/step. | Spearman correlation coefficient between distances and structural scores |
| ------------- | ------------ | -------------------- | -------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| SF-1 | Arch. | SF-Level (~3k Classes) | output_dim: 128 | Increase output_dim significantly (e.g., 256, 512, 768) | Provide higher capacity embedding space needed to separate many classes. | contrastive.yaml::model.output_dim | val/knn_*, val/dist_*, val/recall_at_k, val/map |
| SF-2 | Arch. | SF-Level (~3k Classes) | projection_dims: [1024, 512] | Potentially increase depth/width (e.g., [1024, 1024, 512]) | Increase model capacity if performance plateaus with increased output_dim. | contrastive.yaml::model.projection_dims | val/knn_*, val/dist_*, val/recall_at_k, val/map (monitor for overfitting) |
| SF-3 | Sampling | SF-Level (~3k Classes) | (Should be MPerClassSampler from C-1) | Ensure MPerClassSampler is used; Tune P & K | Essential for ensuring diverse negatives & sufficient positives with many classes. | contrastive_data_module.py::_get_dataloader | train/active_triplets, train/loss, training stability |
| SF-4 | Training | SF-Level (~3k Classes) | batch_size: 1024, accumulate_grad_batches: 4 | Increase effective batch size further if possible | Need larger batches to sample diverse classes/negatives adequately. | contrastive.yaml::training.batch_size, contrastive.yaml::training.accumulate_grad_batches | Memory usage, train/active_triplets, convergence speed, val/knn_* |
| SF-5 | Loss | SF-Level (~3k Classes) | TripletMarginLoss (from C-3) or SoftTripletLoss | Investigate Hierarchical Loss or Multi-Similarity Loss | Explicitly model CATH hierarchy or leverage multiple pairs for potentially better separation. | Requires implementing/integrating new loss functions in contrasted.py | val/knn_*, val/dist_*, potentially hierarchy-aware metrics |
| SF-6 | Validation| SF-Level (~3k Classes) | knn_val_neighbors: 1 | Increase knn_val_neighbors (e.g., to 3 or 5) | Improve robustness of kNN validation with potentially sparse classes. | contrastive.yaml::model.n_neighbors | val/knn_* (compare k=1 vs k>1 results) |
| SF-7 | Viz. | SF-Level (~3k Classes) | t-SNE plot for all points | Adapt visualization: Plot subsets, use UMAP, label centroids | Make visualizations interpretable with a large number of classes. | contrasted.py::_generate_tsne_plot (modify sampling/plotting logic, potentially add UMAP) | Qualitative assessment of cluster separation/structure in visualizations |