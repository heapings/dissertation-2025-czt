import argparse
import os

import pandas as pd

files = ["train", "test"]

bic_to_bank = {
    "ZHSZUS33": "Bank_1",
    "SHSHKHH1": "Bank_2",
    "YXRXGB22": "Bank_3",
    "WPUWDEFF": "Bank_4",
    "YMNYFRPP": "Bank_5",
    "FBSFCHZH": "Bank_6",
    "YSYCESMM": "Bank_7",
    "ZNZZAU3M": "Bank_8",
    "HCBHSGSG": "Bank_9",
    "XITXUS33": "Bank_10",
}


def main():
    args = define_parser()
    root_path = args.input_dir
    
    # Use the GNN-ready files since they have Transaction_ID
    gnn_feat_postfix = "_gnn_ready.csv"
    embed_feat_postfix = "_embedding.csv"
    out_feat_postfix = "_combined.csv"

    # Loop through all folders in root_path that match the pattern xxx_Bank_n
    for folder_name in os.listdir(root_path):
        if not os.path.isdir(os.path.join(root_path, folder_name)):
            continue
        # Check if folder name matches the pattern *_Bank_n
        if "_Bank_" in folder_name:
            print("Processing folder: ", folder_name)
            # Extract BIC from folder name (assume BIC is before the first underscore)
            bic = folder_name.split("_")[0]
        else:
            continue
            
        for file in files:
            gnn_feat_file = os.path.join(
                root_path, bic + "_" + bic_to_bank[bic], file + gnn_feat_postfix
            )
            embed_feat_file = os.path.join(
                root_path, bic + "_" + bic_to_bank[bic], file + embed_feat_postfix
            )
            out_feat_file = os.path.join(
                root_path, bic + "_" + bic_to_bank[bic], file + out_feat_postfix
            )

            try:
                # Load the GNN-ready features (has Transaction_ID) and embedding features
                gnn_feat = pd.read_csv(gnn_feat_file, index_col=0)  # Has saved index
                embed_feat = pd.read_csv(embed_feat_file)
                
                print(f"  Processing {file} dataset:")
                print(f"    GNN features shape: {gnn_feat.shape}")
                print(f"    Embedding shape: {embed_feat.shape}")
                
                # Check if Transaction_ID exists in both
                if "Transaction_ID" not in gnn_feat.columns:
                    print(f"    WARNING: Transaction_ID not in GNN features for {file}")
                    continue
                if "Transaction_ID" not in embed_feat.columns:
                    print(f"    WARNING: Transaction_ID not in embeddings for {file}")
                    continue
                
                # Merge on Transaction_ID
                out_feat = pd.merge(gnn_feat, embed_feat, on="Transaction_ID", suffixes=('', '_embed'))
                
                # Handle duplicate Fraud_Label columns if they exist
                if "Fraud_Label_embed" in out_feat.columns:
                    # Drop the duplicate Fraud_Label from embeddings
                    out_feat = out_feat.drop(columns=["Fraud_Label_embed"])
                
                # Reorder columns to have Transaction_ID and Fraud_Label at appropriate positions
                cols = out_feat.columns.tolist()
                
                # Move Transaction_ID to front if it exists
                if "Transaction_ID" in cols:
                    cols.remove("Transaction_ID")
                    cols = ["Transaction_ID"] + cols
                
                # Move Fraud_Label to end if it exists
                if "Fraud_Label" in cols:
                    cols.remove("Fraud_Label")
                    cols = cols + ["Fraud_Label"]
                
                out_feat = out_feat[cols]
                
                print(f"    Combined shape: {out_feat.shape}")
                print(f"    Columns: {out_feat.shape[1]} features")
                
                # Save the combined features
                out_feat.to_csv(out_feat_file, index=False)
                print(f"    Saved to: {out_feat_file}")
                
            except FileNotFoundError as e:
                print(f"    ERROR: File not found - {e}")
                continue
            except Exception as e:
                print(f"    ERROR: {e}")
                continue


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="input directory containing the processed data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
