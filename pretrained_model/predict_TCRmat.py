"""
Predict peptide-TCR binding using trained Weight Matrix (PAWM) and Residue-Residue Interaction Potential Matrix (RRIPM)

This script loads trained model parameters, predicts peptide-TCR binding for test data, and outputs prediction results.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

# Model configuration parameters (must be consistent with training)
MAX_LEN = 33  # Maximum length of the matrix
MAX_MID = MAX_LEN // 2  # Midpoint position of the matrix
DELET_LEN = 4  # Length to delete from both ends of CDR3 sequence


def data_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data, consistent with training processing
    
    Args:
        data: Dataframe containing peptide, cdr3, and bind columns
        
    Returns:
        Preprocessed dataset
    """
    # Add specific residues to the beginning and end of CDR3 sequences
    data['cdr3'] = ['C' + item + 'F' for item in data['cdr3']]
    
    # Remove specific lengths of residues from both ends of CDR3 sequences
    data['cdr3'] = [item[DELET_LEN : len(item) - DELET_LEN] for item in data['cdr3']]
    
    return data


def calculate_cmap(pep: List[str], tcr: List[str], residue_inter: np.ndarray) -> np.ndarray:
    """
    Calculate contact map, consistent with training calculation method
    
    Args:
        pep: List of peptide sequences
        tcr: List of TCR sequences
        residue_inter: Residue interaction matrix
        
    Returns:
        Calculated contact map matrix
    """
    c_map = [[0.0] * (MAX_LEN * MAX_LEN) for _ in range(len(pep))]
    
    for k in range(len(pep)):
        cmap_store = [[0.0] * MAX_LEN for _ in range(MAX_LEN)]
        
        for i in range(len(pep[k])):
            for j in range(len(tcr[k])):
                # Calculate position indices
                pos_i = MAX_MID + i * 2 - len(pep[k]) + 1
                pos_j = MAX_MID + j * 2 - len(tcr[k]) + 1
                
                # Get residue interaction value
                pep_residue = ord(pep[k][i]) - 65
                tcr_residue = ord(tcr[k][j]) - 65
                cmap_store[pos_i][pos_j] = residue_inter[pep_residue][tcr_residue]
        
        # Flatten matrix
        c_map[k] = [item for sublist in cmap_store for item in sublist]
    
    return np.array(c_map)


def main():
    """Main function, executes prediction process"""
    # Load trained model parameters
    WTM_input = pd.read_csv('parameter/PAWM.mat').values
    WTM_1d = WTM_input.reshape(-1, 1)
    residue_inter = pd.read_csv('parameter/RRIPM.mat').values
    
    # Read and preprocess test data
    data_read = pd.read_csv('data_input/data_test.csv')
    data_read_raw = data_read.copy()  # Save raw data for output
    data_read = data_filter(data_read)
    
    # Extract features
    pep = data_read['peptide'].tolist()
    tcr = data_read['cdr3'].tolist()
    y_real = data_read['bind'].tolist()  # Real labels for reference
    
    # Calculate contact map and predictions
    c_map = calculate_cmap(pep, tcr, residue_inter)
    y_predict = c_map @ WTM_1d
    
    # Normalize predictions
    y_mean = y_predict.mean()
    y_predict = (y_predict - y_mean / 2) / (y_mean / 2) / 2
    
    # Save prediction results
    data_read_raw['predicted_score'] = y_predict
    data_read_raw.to_csv('output/prediction_output.csv', index=None)
    
    print(f"Prediction completed, results saved to output/prediction_output.csv")


if __name__ == "__main__":
    main()