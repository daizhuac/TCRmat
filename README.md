# TCRmat: Epitope-TCR Binding Prediction Model

A machine learning model for predicting Epitope-T cell receptor (TCR) binding interactions using Weight Matrix (PAWM) and Residue-Residue Interaction Potential Matrix (RRIPM).

## Overview

This repository contains code for training and predicting peptide-TCR binding interactions. The model uses an iterative optimization approach to learn both a Position-specific Affinity Weight Matrix (PAWM) and a Residue-Residue Interaction Potential Matrix (RRIPM) from sequence data.

## Repository Structure

```
.
├── data_input/           # Input data directory
│   ├── data_train.csv    # Training data
│   └── data_test.csv     # Test data
├── parameter/            # Parameter files
│   ├── rr_inter_initial.mat  # Initial residue interaction matrix
│   ├── PAWM.mat              # Trained weight matrix (output)
│   └── RRIPM.mat             # Trained residue interaction matrix (output)
├── output/               # Prediction output directory
│   └── prediction_output.csv # Prediction results (output)
├── train_TCRmat.py        # Model training script
└── predict_TCRmat.py      # Prediction script
```

## Requirements

- Python 3.8+
- pandas 1.3.0+
- numpy 1.21.0+

Install dependencies with:
```bash
pip install pandas>=1.3.0 numpy>=1.21.0
```

## Data Format

Input data files (`data_train.csv` and `data_test.csv`) should contain the following columns:
- `peptide`: Amino acid sequence of the peptide
- `cdr3`: CDR3 region sequence of the TCR
- `bind`: Binding label (1 for binding, 0 for non-binding) - required for training only

## Usage

### Training

1. Prepare your training data in `data_input/data_train.csv`
2. Ensure the initial residue interaction matrix exists at `parameter/rr_inter_initial.mat`
3. Run the training script:

```bash
python train_TCRmat.py
```

The training process will generate two output files:
- `parameter/PAWM.mat`: Trained Position-specific Affinity Weight Matrix
- `parameter/RRIPM.mat`: Trained Residue-Residue Interaction Potential Matrix

### Prediction

1. Prepare your test data in `data_input/data_test.csv`
2. Ensure trained model parameters (`PAWM.mat` and `RRIPM.mat`) exist in the `parameter` directory
3. Run the prediction script:

```bash
python predict_TCRmat.py
```

Prediction results will be saved to `output/prediction_output.csv` with an additional `predicted_score` column.

## Model Details

The model uses an iterative optimization approach:

1. **Initialization**: Both matrices are initialized with reasonable starting values
2. **Iterative Optimization**:
   - Update the weight matrix using the current residue interaction matrix
   - Update the residue interaction matrix using the current weight matrix
   - Repeat for a fixed number of iterations
3. **Smoothing**: Apply smoothing to the weight matrix to reduce noise

The prediction score is calculated using the dot product of the contact map (derived from peptide-TCR sequences and RRIPM) and the weight matrix (PAWM).

## Parameters

Key parameters (defined in the scripts):
- `MAX_LEN`: Maximum length of the weight matrix (33)
- `ITERATION_NUM`: Number of optimization iterations (3)
- `DELET_LEN`: Length to trim from both ends of CDR3 sequences (4)
- `WTM_INIT_PARAM`: Weight matrix initialization parameter (2.5)
- `WTM_SMOOTH_FACTOR`: Weight matrix smoothing factor (0.1)
- `RESIDUE_INTER_WEIGHT`: Residue interaction matrix weight (4000)

## Output

The prediction output file contains all original columns from the test data plus:
- `predicted_score`: The predicted binding score (higher values indicate stronger predicted binding)

