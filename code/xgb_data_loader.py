# xgb_data_loader.py
# Data loader for XGBoost federated learning with optional GNN embeddings

import os
from typing import Tuple

import pandas as pd
import numpy as np
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader


class CreditCardDataLoader(XGBDataLoader):
    """
    Simple data loader for federated XGBoost training.
    Assumes data has already been preprocessed (scaled, outliers removed, SMOTE applied).
    """
    
    def __init__(self, root_dir: str, file_postfix: str = "_normalized.csv"):
        self.dataset_names = ["train", "test"]
        self.base_file_names = {}
        self.root_dir = root_dir
        
        for name in self.dataset_names:
            self.base_file_names[name] = name + file_postfix

    def initialize(
        self,
        client_id: str,
        rank: int,
        data_split_mode: xgb.core.DataSplitMode = xgb.core.DataSplitMode.ROW,
    ):
        super().initialize(client_id, rank, data_split_mode)

    def load_data(self) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """
        Load preprocessed data for federated XGBoost training
        """
        data = {}
        
        for ds_name in self.dataset_names:
            print(f"\nLoading {ds_name} dataset for site: {self.client_id}")
            file_name = os.path.join(
                self.root_dir, self.client_id, self.base_file_names[ds_name]
            )
            
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"File not found: {file_name}")
            
            print(f"  Loading from: {file_name}")
            # Read without assuming index column
            df = pd.read_csv(file_name)
            
            # Check if there's an unnamed index column and drop it
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            # Drop Transaction_ID if it exists (not a feature)
            if 'Transaction_ID' in df.columns:
                df = df.drop(columns=['Transaction_ID'])
                print(f"  Dropped Transaction_ID column")
            
            # Separate features and target
            if "Fraud_Label" in df.columns:
                y = df["Fraud_Label"].values
                x = df.drop(columns=["Fraud_Label"]).values
            else:
                # Assume last column is the target if Fraud_Label not found
                y = df.iloc[:, -1].values
                x = df.iloc[:, :-1].values
            
            print(f"  Feature matrix shape: {x.shape}")
            print(f"  Target shape: {y.shape}")
            print(f"  Fraud rate: {y.mean():.2%}")
            print(f"  Class distribution: {np.bincount(y.astype(int))}")
            
            data[ds_name] = (x, y)
        
        # Check if we have both training and test data
        if "train" not in data or "test" not in data:
            raise ValueError(f"Missing required datasets for {self.client_id}")
        
        # Get training and test data
        x_train, y_train = data["train"]
        x_valid, y_valid = data["test"]
        
        # Calculate scale_pos_weight for class imbalance
        # Note: Training data may already be balanced by SMOTE
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"\n  Training samples: {len(y_train)}")
        print(f"  Validation samples: {len(y_valid)}")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        
        # Create DMatrix objects
        dmat_train = xgb.DMatrix(
            x_train, 
            label=y_train,
            data_split_mode=self.data_split_mode
        )
        
        dmat_valid = xgb.DMatrix(
            x_valid, 
            label=y_valid,
            data_split_mode=self.data_split_mode
        )
        
        return dmat_train, dmat_valid


class CreditCardDataLoaderWithGNN(CreditCardDataLoader):
    """
    Data loader for federated XGBoost training with GNN embeddings.
    Uses combined features from GNN-ready data + GNN embeddings.
    """
    
    def __init__(self, root_dir: str, file_postfix: str = "_combined.csv"):
        # Note: Changed default from "_combined_normalized.csv" to "_combined.csv"
        # to match what merge_feat.py actually creates
        super().__init__(root_dir, file_postfix)
