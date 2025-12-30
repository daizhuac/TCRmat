"""
Train Weight Matrix (PAWM) and Residue-Residue Interaction Potential Matrix (RRIPM) for peptide-TCR binding prediction

This script implements the functionality of training a weight matrix model based on peptide and T cell receptor (TCR) sequence data,
generating model parameters through an iterative optimization algorithm for subsequent binding prediction tasks.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# Model configuration parameters
MAX_LEN = 33  # Maximum length of the matrix
MAX_MID = MAX_LEN // 2  # Midpoint position of the matrix
ITERATION_NUM = 3  # Number of iterative optimizations
DELET_LEN = 4  # Length to delete from both ends of CDR3 sequence
WTM_INIT_PARAM = 2.5  # Weight matrix initialization parameter
WTM_SMOOTH_FACTOR = 0.1  # Weight matrix smoothing factor
RESIDUE_INTER_WEIGHT = 4000  # Residue interaction matrix weight
RESIDUE_SORT = 'ACDEFGHIKLMNPQRSTVWY'  # Amino acid order


def data_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data, mainly processing CDR3 sequences
    
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


def initialize_matrices(ref_matrix_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize residue interaction matrix and weight matrix
    
    Args:
        ref_matrix_path: Path to reference residue interaction matrix
        
    Returns:
        residue_inter: Initialized residue interaction matrix
        residue_inter_0: Copy of initial residue interaction matrix
        WTM_0: Initial weight matrix
    """
    # Read and process reference residue interaction matrix
    aainter_refer = pd.read_csv(ref_matrix_path, header=0, index_col=0)
    
    # Negate matrix values
    for i in aainter_refer.columns:
        for j in aainter_refer.index:
            aainter_refer[i][j] *= -1
    
    # Ensure matrix symmetry
    i = 0
    for aaname in aainter_refer.columns:
        i += 1
        for j in range(i, 20):
            index = aainter_refer.index[j]
            aainter_refer[aaname][index] = aainter_refer[index][aaname]
    
    # Initialize residue interaction matrix
    residue_inter = [[0] * 26 for _ in range(26)]
    for i in aainter_refer.index:
        for j in aainter_refer.index:
            residue_inter[ord(i) - ord('A')][ord(j) - ord('A')] = aainter_refer[i][j]
    
    residue_inter_0 = np.array(residue_inter)
    
    # Initialize weight matrix
    WTM_0 = [[0.0] * MAX_LEN for _ in range(MAX_LEN)]
    for i in range(MAX_LEN):
        for j in range(MAX_LEN):
            WTM_0[i][j] = 1 / (WTM_INIT_PARAM + (i - MAX_MID + 0.5)**2 + (j - MAX_MID)** 2 - 1)
    
    return np.array(residue_inter), residue_inter_0, np.array(WTM_0)


def calculate_cmap(pep: List[str], tcr: List[str], residue_inter: np.ndarray) -> np.ndarray:
    """
    Calculate contact map
    
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


def iteration_wtm(pep: List[str], tcr: List[str], y_real: List[int], 
                 WTM_0: np.ndarray, residue_inter: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iteratively optimize weight matrix
    
    Args:
        pep: List of peptide sequences
        tcr: List of TCR sequences
        y_real: Real binding labels
        WTM_0: Initial weight matrix
        residue_inter: Residue interaction matrix
        
    Returns:
        WTM: Optimized weight matrix
        WTM_1d: 1D flattened weight matrix
    """
    # Calculate contact map
    c_map = calculate_cmap(pep, tcr, residue_inter)
    
    # Solve pseudoinverse matrix to get weights
    A = np.array(c_map)
    B = np.array(y_real)
    xx = np.linalg.pinv(A) @ B
    xx_2d = xx.reshape(MAX_LEN, MAX_LEN)
    
    # Smoothing process
    WTM = (xx_2d + WTM_0 / WTM_SMOOTH_FACTOR) / 2
    WTM_1d = WTM.reshape(-1, 1)
    
    return WTM, WTM_1d


def iteration_rrintermap(pep: List[str], tcr: List[str], y_real: List[int], 
                        WTM: np.ndarray, residue_inter_0: np.ndarray) -> np.ndarray:
    """
    Iteratively optimize residue interaction matrix
    
    Args:
        pep: List of peptide sequences
        tcr: List of TCR sequences
        y_real: Real binding labels
        WTM: Weight matrix
        residue_inter_0: Initial residue interaction matrix
        
    Returns:
        Optimized residue interaction matrix
    """
    wt_total_1d = [[0.0] * (26 * 26) for _ in range(len(pep))]
    
    for k in range(len(pep)):
        wt_total = [[0.0] * 26 for _ in range(26)]
        
        for i in range(len(pep[k])):
            for j in range(len(tcr[k])):
                # Calculate position indices
                pos_i = MAX_MID + i * 2 - len(pep[k]) + 1
                pos_j = MAX_MID + j * 2 - len(tcr[k]) + 1
                
                # Update weight sum
                pep_res = ord(pep[k][i]) - 65
                tcr_res = ord(tcr[k][j]) - 65
                wt_total[pep_res][tcr_res] += WTM[pos_i][pos_j]
                wt_total[tcr_res][pep_res] += WTM[pos_i][pos_j]
        
        # Flatten matrix
        wt_total_1d[k] = [item for sublist in wt_total for item in sublist]
    
    # Solve pseudoinverse matrix to get residue interactions
    wt_total_1d = np.array(wt_total_1d)
    y_real_arr = np.array(y_real)
    rr_inter_1d = np.linalg.pinv(wt_total_1d) @ y_real_arr
    
    # Reshape and smooth
    residue_inter = rr_inter_1d.reshape(26, -1)
    residue_inter = (residue_inter + residue_inter_0 / RESIDUE_INTER_WEIGHT) / 2
    
    return residue_inter


def smooth_wtm(WTM: np.ndarray) -> np.ndarray:
    """
    Smooth weight matrix, handling odd rows
    
    Args:
        WTM: Weight matrix
        
    Returns:
        Smoothed weight matrix
    """
    def get_value(mat: np.ndarray, x: int, y: int) -> float:
        """Get matrix value, handling boundary conditions"""
        if 0 <= x < len(mat) and 0 <= y < len(mat[0]):
            return mat[x][y]
        return 0.0
    
    # Smooth odd rows
    for i in range(len(WTM)):
        for j in range(len(WTM[i])):
            if i % 2 == 1:
                WTM[i][j] = (get_value(WTM, i-1, j) + get_value(WTM, i+1, j)) / 2
    
    return WTM


def main():
    """Main function, executes model training process"""
    # Read and preprocess training data
    data_read = pd.read_csv('data_input/data_train.csv')
    data_read = data_filter(data_read)
    
    # Extract features and labels
    pep = data_read['peptide'].tolist()
    tcr = data_read['cdr3'].tolist()
    y_real = data_read['bind'].tolist()
    
    # Initialize matrices
    residue_inter, residue_inter_0, WTM_0 = initialize_matrices('parameter/rr_inter_initial.mat')
    WTM = WTM_0.copy()
    WTM_1d = WTM.reshape(-1, 1)
    
    # Iterative optimization
    for _ in range(ITERATION_NUM):
        WTM, WTM_1d = iteration_wtm(pep, tcr, y_real, WTM_0, residue_inter)
        residue_inter = iteration_rrintermap(pep, tcr, y_real, WTM, residue_inter_0)
    
    # Smooth weight matrix
    WTM = smooth_wtm(WTM)
    
    # Calculate final contact map
    c_map = calculate_cmap(pep, tcr, residue_inter)
    
    # Generate residue likelihood matrix
    residue_list = [item for item in RESIDUE_SORT]
    res_likeli_mat = pd.DataFrame([[0.0] * 20 for _ in range(20)],
                                 columns=residue_list, index=residue_list)
    
    for i in res_likeli_mat.index:
        for j in res_likeli_mat.columns:
            res_likeli_mat.at[j, i] = residue_inter[ord(i) - 65][ord(j) - 65]
    
    # Save model parameters
    pd.DataFrame(WTM).to_csv('parameter/PAWM.mat', index=None)
    pd.DataFrame(residue_inter).to_csv('parameter/RRIPM.mat', index=None)
    
    print("Model training completed, weight matrix and residue interaction matrix saved")


if __name__ == "__main__":
    main()