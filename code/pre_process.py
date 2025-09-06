import argparse
import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# (1) import nvflare client API
import nvflare.client as flare

dataset_names = ["train", "test"]
datasets = {}


def main():
    print("\n pre-process starts \n ")
    args = define_parser()
    input_dir = args.input_dir
    output_dir = args.output_dir

    flare.init()
    site_name = flare.get_site_name()

    # receives global message from NVFlare
    etl_task = flare.receive()

    print("\n receive task \n ")
    
    # Process for standard ML pipeline (with SMOTE, scaling, outlier removal)
    processed_dfs_ml = process_dataset_for_ml(input_dir, site_name)
    save_normalized_files(output_dir, processed_dfs_ml, site_name, suffix="_normalized")
    
    # Process for GNN pipeline (encoding only, preserve Transaction_ID)
    processed_dfs_gnn = process_dataset_for_gnn(input_dir, site_name)
    save_normalized_files(output_dir, processed_dfs_gnn, site_name, suffix="_gnn_ready")

    print("end task")
    # send message back the controller indicating end.
    etl_task.meta["status"] = "done"
    flare.send(etl_task)


def save_normalized_files(output_dir, processed_dfs, site_name, suffix="_normalized"):
    for name in processed_dfs:
        print(f"\n dataset {name=} with suffix {suffix} \n ")
        site_dir = os.path.join(output_dir, site_name)
        os.makedirs(site_dir, exist_ok=True)

        file_name = os.path.join(site_dir, f"{name}{suffix}.csv")
        print("save to = ", file_name)
        processed_dfs[name].to_csv(file_name)


def remove_outliers_zscore(df, numerical_columns, threshold=3.0):
    """Remove outliers using Z-score method"""
    df_filtered = df.copy()
    for col in numerical_columns:
        if col in df_filtered.columns:
            mean = df_filtered[col].mean()
            std = df_filtered[col].std()
            if std > 0:  # Avoid division by zero
                z_scores = np.abs((df_filtered[col] - mean) / std)
                df_filtered = df_filtered[z_scores <= threshold]
    
    return df_filtered


def process_dataset_for_gnn(input_dir, site_name):
    """
    Minimal processing for GNN pipeline:
    - Keep Transaction_ID
    - Encode categorical variables
    - Preserve Timestamp as numeric
    - No SMOTE, no scaling, no outlier removal
    """
    processed_dfs = {}
    
    category_columns = [
        # Transaction categories
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type",
        "Authentication_Method",
        # Financial identifiers
        "Sender_BIC",
        "Receiver_BIC",
        "Beneficiary_BIC",
        "Currency",
        "Currency_Country",
    ]
    
    # Initialize LabelEncoders for categorical features
    label_encoders = {col: LabelEncoder() for col in category_columns}
    
    for ds_name in dataset_names:
        file_name = os.path.join(input_dir, site_name, f"{ds_name}_enrichment.csv")
        df = pd.read_csv(file_name)
        
        print(f"\n[GNN Pipeline] Processing {ds_name} dataset...")
        print(f"Original shape: {df.shape}")
        
        # Preserve Transaction_ID as first column
        transaction_id = df["Transaction_ID"] if "Transaction_ID" in df.columns else None
        
        # Preserve Fraud_Label if it exists
        has_label = "Fraud_Label" in df.columns
        fraud_label = df["Fraud_Label"] if has_label else None
        
        # Convert Timestamp to numeric
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["Timestamp"] = df["Timestamp"].astype(int) / 10**9  # convert to seconds
        
        # Identify all object/string columns for encoding
        actual_categorical_cols = []
        for col in df.columns:
            if col in ["Transaction_ID", "Fraud_Label"]:
                continue  # Skip these special columns
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                actual_categorical_cols.append(col)
        
        print(f"[GNN Pipeline] Found categorical columns to encode: {actual_categorical_cols}")
        
        # Encode ALL categorical variables (not just those in the predefined list)
        for col in actual_categorical_cols:
            if ds_name == "train":
                # For columns in our predefined list, use the existing encoder
                if col in category_columns:
                    df[col] = label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # For unexpected categorical columns, create a new encoder
                    print(f"[GNN Pipeline] Creating new encoder for unexpected column: {col}")
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    label_encoders[col] = encoder  # Save for test set
            else:
                # Test set - use fitted encoders
                if col in label_encoders:
                    try:
                        df[col] = label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        print(f"[GNN Pipeline] Warning: Unseen categories in {col}. Using default encoding.")
                        df[col] = df[col].apply(
                            lambda x: label_encoders[col].transform([str(x)])[0] 
                            if str(x) in label_encoders[col].classes_ else -1
                        )
                else:
                    # Column wasn't in training data
                    print(f"[GNN Pipeline] Warning: Column {col} not in training data, encoding with default")
                    df[col] = 0
        
        # Keep all columns except Transaction_ID and Fraud_Label (will add them back)
        columns_to_keep = [col for col in df.columns 
                          if col not in ["Transaction_ID", "Fraud_Label"]]
        df_processed = df[columns_to_keep].copy()
        
        # Final verification that all columns are numeric
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'string':
                print(f"[GNN Pipeline] ERROR: Column {col} is still non-numeric after encoding!")
                # Force conversion as last resort
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        
        # Add back Transaction_ID as first column
        if transaction_id is not None:
            df_processed.insert(0, "Transaction_ID", transaction_id)
        
        # Add back Fraud_Label
        if fraud_label is not None:
            df_processed["Fraud_Label"] = fraud_label
        
        print(f"[GNN Pipeline] Final shape for {ds_name}: {df_processed.shape}")
        print(f"[GNN Pipeline] Columns preserved: Transaction_ID={transaction_id is not None}, Fraud_Label={has_label}")
        
        processed_dfs[ds_name] = df_processed
    
    return processed_dfs


def process_dataset_for_ml(input_dir, site_name):
    """
    Full processing for standard ML pipeline:
    - Remove outliers
    - Encode categorical variables
    - Scale numerical features
    - Apply SMOTE for training set
    - Drop Transaction_ID (not needed for traditional ML)
    """
    processed_dfs = {}
    
    numerical_columns = [
        # Core transaction metrics
        "Transaction_Amount",
        "Account_Balance",
        "Transaction_Distance",
        # Binary flags (already numeric)
        "IP_Address_Flag",
        "Is_Weekend",
        # Engineered ratio features
        "Transaction_Ratio",
        "Failed_Transaction_Rate",
        "Fraud_History_Impact",
        "currency_deviation_ratio",
        "beneficiary_deviation_ratio",
        # Volume and amount metrics
        "trans_volume",
        "total_amount",
        "average_amount",
        "hist_trans_volume",
        "hist_total_amount",
        "hist_average_amount",
    ]
    
    category_columns = [
        # Transaction categories
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type",
        "Authentication_Method",
        # Financial identifiers
        "Sender_BIC",
        "Receiver_BIC",
        "Beneficiary_BIC",
        "Currency",
        "Currency_Country",
    ]

    # Initialize LabelEncoders and Scaler
    label_encoders = {col: LabelEncoder() for col in category_columns}
    scaler = StandardScaler()

    for ds_name in dataset_names:
        file_name = os.path.join(input_dir, site_name, f"{ds_name}_enrichment.csv")
        df = pd.read_csv(file_name)
        
        print(f"\n[ML Pipeline] Processing {ds_name} dataset...")
        print(f"Original shape: {df.shape}")
        
        # Drop Transaction_ID as it's not needed for traditional ML
        if "Transaction_ID" in df.columns:
            df = df.drop(columns=["Transaction_ID"])
            print("[ML Pipeline] Dropped Transaction_ID column")
        
        # Store the Fraud_Label separately if it exists
        has_label = "Fraud_Label" in df.columns
        if has_label:
            y = df["Fraud_Label"].copy()

        # Convert 'Timestamp' column to datetime and then to Unix timestamp
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["Timestamp"] = df["Timestamp"].astype(int) / 10**9  # convert to seconds
            # Add Timestamp to numerical columns for scaling
            if "Timestamp" not in numerical_columns:
                numerical_columns.append("Timestamp")

        # Remove outliers from training set only
        if ds_name == "train":
            print("[ML Pipeline] Removing outliers using Z-score method...")
            initial_shape = df.shape[0]
            
            df_clean = remove_outliers_zscore(df, ["Transaction_Amount"], threshold=3.0)
            
            # Update labels to match the cleaned data
            if has_label:
                remaining_indices = df_clean.index
                y = y.loc[remaining_indices]
            
            df = df_clean
            print(f"[ML Pipeline] Outliers removed: {initial_shape - df.shape[0]} rows")
            print(f"[ML Pipeline] Shape after outlier removal: {df.shape}")

        # Encode categorical variables
        print("[ML Pipeline] Encoding categorical features...")
        for col in category_columns:
            if col in df.columns:
                if ds_name == "train":
                    df[col] = label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    try:
                        df[col] = label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        print(f"[ML Pipeline] Warning: Unseen categories in {col}. Using default encoding.")
                        df[col] = df[col].apply(
                            lambda x: label_encoders[col].transform([str(x)])[0] 
                            if str(x) in label_encoders[col].classes_ else -1
                        )

        # Separate numerical and categorical features
        actual_numerical_cols = [col for col in numerical_columns if col in df.columns]
        actual_categorical_cols = [col for col in category_columns if col in df.columns]
        
        numerical_features = df[actual_numerical_cols]
        categorical_features = df[actual_categorical_cols]

        # Apply StandardScaler to numerical features
        print("[ML Pipeline] Applying StandardScaler to numerical features...")
        if ds_name == "train":
            numerical_scaled = pd.DataFrame(
                scaler.fit_transform(numerical_features),
                columns=numerical_features.columns,
                index=numerical_features.index
            )
        else:
            numerical_scaled = pd.DataFrame(
                scaler.transform(numerical_features),
                columns=numerical_features.columns,
                index=numerical_features.index
            )

        # Combine the scaled numerical features with the encoded categorical features
        df_combined = pd.concat([categorical_features, numerical_scaled], axis=1)

        # Apply SMOTE only to training data
        if ds_name == "train" and has_label:
            print("[ML Pipeline] Applying SMOTE for handling imbalanced data...")
            print(f"Original class distribution:\n{y.value_counts()}")
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(df_combined, y)
            
            df_combined = pd.DataFrame(X_resampled, columns=df_combined.columns)
            y = pd.Series(y_resampled, name="Fraud_Label")

            print(f"[ML Pipeline] Resampled class distribution:\n{y.value_counts()}")
            print(f"[ML Pipeline] Shape after SMOTE: {df_combined.shape}")
            
            df_combined["Fraud_Label"] = y
        elif has_label and ds_name == "test":
            df_combined["Fraud_Label"] = y

        print(f"[ML Pipeline] Final shape for {ds_name}: {df_combined.shape}")
        
        processed_dfs[ds_name] = df_combined

    return processed_dfs


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        nargs="?",
        help="input directory where csv files for each site are expected, default to /tmp/dataset/credit_data",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        help="output directory, default to '/tmp/dataset/credit_data'",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()