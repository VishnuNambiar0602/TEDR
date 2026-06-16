"""
Data Augmentation Script
Augments the traffic dataset by 3x using multiple techniques
"""

import numpy as np
import json
import os
from datetime import datetime

def load_data():
    """Load all current data"""
    processed_dir = "test_data/processed"
    
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata

def add_gaussian_noise(data, noise_level=0.05):
    """Add Gaussian noise to data"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def apply_jitter(data, jitter_amount=0.1):
    """Apply jitter to data"""
    jitter = np.random.uniform(-jitter_amount, jitter_amount, data.shape)
    return data + jitter

def apply_mixup(X1, y1, X2, y2, alpha=0.2):
    """Apply mixup augmentation"""
    lam = np.random.beta(alpha, alpha)
    X_mixed = lam * X1 + (1 - lam) * X2
    y_mixed = lam * y1 + (1 - lam) * y2
    return X_mixed, y_mixed

def apply_scaling(data, scale_factor=0.1):
    """Apply random scaling to data"""
    scale = 1 + np.random.uniform(-scale_factor, scale_factor)
    return data * scale

def augment_dataset_3x(X, y):
    """
    Create 3x augmented version of dataset
    Original data + 2 augmented versions
    """
    augmented_X = [X.copy()]
    augmented_y = [y.copy()]
    
    print(f"Original data shape: {X.shape}")
    print(f"Augmenting to 3x size...")
    
    # Augmentation version 1: Noise + Jitter + Scaling
    X_aug1 = X.copy()
    X_aug1 = add_gaussian_noise(X_aug1, noise_level=0.03)
    X_aug1 = apply_jitter(X_aug1, jitter_amount=0.05)
    X_aug1 = apply_scaling(X_aug1, scale_factor=0.08)
    augmented_X.append(X_aug1)
    augmented_y.append(y.copy())
    
    # Augmentation version 2: Mixup with original data
    X_aug2 = X.copy()
    # Create mixup combinations
    for i in range(len(X)):
        random_idx = np.random.randint(0, len(X))
        X_mixed, y_mixed = apply_mixup(X[i:i+1], y[i:i+1], 
                                       X[random_idx:random_idx+1], 
                                       y[random_idx:random_idx+1], 
                                       alpha=0.3)
        X_aug2[i] = X_mixed[0]
    
    augmented_X.append(X_aug2)
    augmented_y.append(y.copy())
    
    # Concatenate all versions
    X_final = np.concatenate(augmented_X, axis=0)
    y_final = np.concatenate(augmented_y, axis=0)
    
    print(f"Augmented data shape: {X_final.shape}")
    print(f"Size increase: {len(X_final) / len(X):.2f}x")
    
    return X_final, y_final

def augment_all_data():
    """Main augmentation function"""
    print("=" * 60)
    print("Starting Data Augmentation (3x Expansion)")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_data()
    
    print(f"\nOriginal Data Sizes:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Testing:    {X_test.shape[0]} samples")
    print(f"  Total:      {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} samples")
    
    # Augment each dataset
    print("\n" + "="*60)
    print("Augmenting Training Data...")
    print("="*60)
    X_train_aug, y_train_aug = augment_dataset_3x(X_train, y_train)
    
    print("\n" + "="*60)
    print("Augmenting Validation Data...")
    print("="*60)
    X_val_aug, y_val_aug = augment_dataset_3x(X_val, y_val)
    
    print("\n" + "="*60)
    print("Augmenting Test Data...")
    print("="*60)
    X_test_aug, y_test_aug = augment_dataset_3x(X_test, y_test)
    
    # Save augmented data
    processed_dir = "test_data/processed"
    backup_dir = "test_data/processed_backup"
    
    # Create backup of original data
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"\nBacking up original data to {backup_dir}...")
        import shutil
        for file in ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]:
            shutil.copy(
                os.path.join(processed_dir, file),
                os.path.join(backup_dir, file)
            )
    
    # Save augmented data
    print(f"\nSaving augmented data to {processed_dir}...")
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_aug)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train_aug)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val_aug)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val_aug)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_aug)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test_aug)
    
    # Update metadata
    metadata["augmentation_date"] = datetime.now().isoformat()
    metadata["augmentation_factor"] = 3
    metadata["original_training_samples"] = int(X_train.shape[0])
    metadata["augmented_training_samples"] = int(X_train_aug.shape[0])
    metadata["original_validation_samples"] = int(X_val.shape[0])
    metadata["augmented_validation_samples"] = int(X_val_aug.shape[0])
    metadata["original_test_samples"] = int(X_test.shape[0])
    metadata["augmented_test_samples"] = int(X_test_aug.shape[0])
    
    with open(os.path.join(processed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE")
    print("="*60)
    print(f"\nNew Data Sizes (3x):")
    print(f"  Training:   {X_train_aug.shape[0]} samples (was {X_train.shape[0]})")
    print(f"  Validation: {X_val_aug.shape[0]} samples (was {X_val.shape[0]})")
    print(f"  Testing:    {X_test_aug.shape[0]} samples (was {X_test.shape[0]})")
    print(f"  Total:      {X_train_aug.shape[0] + X_val_aug.shape[0] + X_test_aug.shape[0]} samples")
    print(f"\nAugmentation Techniques Used:")
    print(f"  1. Gaussian noise injection")
    print(f"  2. Jitter application")
    print(f"  3. Random scaling")
    print(f"  4. Mixup augmentation")
    print(f"\nBackup of original data saved to: {backup_dir}")

if __name__ == "__main__":
    augment_all_data()
