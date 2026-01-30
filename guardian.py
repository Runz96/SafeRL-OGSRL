import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm

def load_data(use_augmented=True):
    """
    Load CSV data and index arrays, extract features and rewards.
    """
    file_name_suffix = '_state_act' if use_augmented else '_state'
    
    # Load data and indices
    data = pd.read_csv('sepsis_final_RAW_continuous_13.csv')
    seed = 64
    os.chdir(f'./{seed}/')
    train_indices = np.load('train_indices.npy')
    val_indices   = np.load('val_indices.npy')
    test_indices  = np.load('test_indices.npy')
    
    # Extract variables (assumes fixed column order, no header)
    gender     = data.iloc[:, 2].values
    re_adm     = data.iloc[:, 3].values
    age        = data.iloc[:, 4].values
    mech       = data.iloc[:, 5].values
    GCS        = data.iloc[:, 6].values
    FiO2_1     = data.iloc[:, 7].values
    paO2       = data.iloc[:, 8].values
    PaO2_FiO2  = data.iloc[:, 9].values
    Total_bili = data.iloc[:, 10].values
    output_4h  = data.iloc[:, 11].values
    output_total = data.iloc[:, 12].values
    input_total  = data.iloc[:, 13].values
    SpO2       = data.iloc[:, 14].values
    max_dose   = data.iloc[:, 15].values
    input_4h   = data.iloc[:, 16].values  # renamed column
    reward     = data.iloc[:, 18].values   # reward: 0 = inlier, 1 = outlier
    
    # Define feature sets
    x_state = np.column_stack((gender, re_adm, age, mech, GCS, FiO2_1, paO2,
                               PaO2_FiO2, Total_bili, output_4h, output_total,
                               input_total, SpO2))
    
    x_state_act = np.column_stack((gender, re_adm, age, mech, GCS, FiO2_1, paO2,
                                   PaO2_FiO2, Total_bili, output_4h, output_total,
                                   input_total, SpO2, max_dose, input_4h))
    
    selected_data = x_state_act if use_augmented else x_state
    return selected_data, reward, train_indices, val_indices, test_indices, file_name_suffix

def normalize_features(data, train_indices):
    """
    Normalize features column-wise using the training set's min and max.
    Transformation: (2*(value - min) / (max-min) - 1)
    """
    n_features = data.shape[1]
    data_norm = np.empty_like(data, dtype=float)
    
    for j in range(n_features):
        col = data[:, j]
        col_train = col[train_indices]
        min_val = np.min(col_train)
        max_val = np.max(col_train)
        range_val = max_val - min_val
        
        if range_val == 0:
            data_norm[:, j] = col
        else:
            data_norm[:, j] = 2 * (col - min_val) / range_val - 1
    return data_norm

def compute_kernel_density(data, train_data, sigma):
    """
    Compute the kernel density for each point in `data` by comparing it to `train_data`
    using a Gaussian kernel.
    """
    n_train = train_data.shape[0]
    densities = np.zeros(data.shape[0])
    
    for i in tqdm(range(data.shape[0]), desc="Computing kernel density"):
        xs = data[i, :]
        diff = train_data - xs
        norm_sq = np.sum(diff**2, axis=1)
        kernel_vals = np.exp(-0.5 * norm_sq / sigma**2) / (sigma * np.sqrt(2 * np.pi))
        densities[i] = np.sum(kernel_vals) / n_train
    return densities

def classify_points(density, threshold):
    """
    Classify points as inliers (0) if density >= threshold, or outliers (1) otherwise.
    """
    return np.where(density >= threshold, 0, 1)

def save_results(file_name_suffix, kd_params, train_labels, val_labels, test_labels, all_labels):
    """
    Save kernel density parameters and classification labels.
    """
    np.save(f'sepsis_final_RAW_continuous_13_kd_params{file_name_suffix}.npy', kd_params)
    np.save(f'sepsis_final_RAW_continuous_13_train_outlier_labels{file_name_suffix}.npy', train_labels)
    np.save(f'sepsis_final_RAW_continuous_13_val_outlier_labels{file_name_suffix}.npy', val_labels)
    np.save(f'sepsis_final_RAW_continuous_13_test_outlier_labels{file_name_suffix}.npy', test_labels)
    np.save(f'sepsis_final_RAW_continuous_13_all_outlier_labels{file_name_suffix}.npy', all_labels)

def main():
    # -------------------------------
    # 1. Load Data and Define Variables
    # -------------------------------
    use_augmented = True
    selected_data, reward, train_idx, val_idx, test_idx, file_name_suffix = load_data(use_augmented)
    
    # -------------------------------
    # 2. Normalize Features
    # -------------------------------
    data_norm = normalize_features(selected_data, train_idx)
    
    # -------------------------------
    # 3. Split Data into Train, Validation, and Test Sets
    # -------------------------------
    train_data = data_norm[train_idx, :]
    val_data   = data_norm[val_idx, :]
    test_data  = data_norm[test_idx, :]
    
    # -------------------------------
    # 4. Set Kernel Density Parameters
    # -------------------------------
    alpha_true = 0.05
    sigma = 0.535
    print("Training outlier ratio (alpha_true):", alpha_true)
    print("Sigma set to:", sigma)
    
    # -------------------------------
    # 5. Compute Kernel Density for Training Data
    # -------------------------------
    print("Computing kernel density for training data...")
    train_density = compute_kernel_density(train_data, train_data, sigma)
    
    # -------------------------------
    # 6. Determine the Kernel Density Threshold
    # -------------------------------
    thr_kd = np.percentile(train_density, 100 * alpha_true)
    n_train = train_data.shape[0]
    estimated_outlier_fraction = 1 - np.sum(train_density >= thr_kd) / n_train
    print("Determined kernel density threshold (thr_kd):", thr_kd)
    print("Estimated outlier fraction with threshold:", estimated_outlier_fraction)
    
    kd_params = {'sigma': sigma, 'thr_kd': thr_kd}
    
    # -------------------------------
    # 7. Compute Kernel Density for Test Data
    # -------------------------------
    print("Computing kernel density for test data...")
    start_time = time.time()
    test_density = compute_kernel_density(test_data, train_data, sigma)
    elapsed_time = time.time() - start_time
    print("Test density computation time: {:.4f} seconds".format(elapsed_time))
    
    # -------------------------------
    # 8. Compute Kernel Density for Validation Data
    # -------------------------------
    print("Computing kernel density for validation data...")
    start_time = time.time()
    val_density = compute_kernel_density(val_data, train_data, sigma)
    elapsed_time = time.time() - start_time
    print("Validation density computation time: {:.4f} seconds".format(elapsed_time))
    
    # -------------------------------
    # 9. Classify Points as Inliers or Outliers
    # -------------------------------
    train_labels = classify_points(train_density, thr_kd)
    val_labels   = classify_points(val_density, thr_kd)
    test_labels  = classify_points(test_density, thr_kd)
    
    n_pred_inliers = np.sum(test_labels == 0)
    n_pred_outliers = np.sum(test_labels == 1)
    print("Test set classification: {} predicted inliers, {} predicted outliers".format(n_pred_inliers, n_pred_outliers))
    
    # -------------------------------
    # 10. Create Combined Labels for All Data
    # -------------------------------
    all_labels = np.empty_like(np.concatenate((train_labels, val_labels, test_labels)))
    all_labels[train_idx] = train_labels
    all_labels[val_idx]   = val_labels
    all_labels[test_idx]  = test_labels
    
    # -------------------------------
    # 11. Save Results
    # -------------------------------
    save_results(file_name_suffix, kd_params, train_labels, val_labels, test_labels, all_labels)

if __name__ == '__main__':
    main()
