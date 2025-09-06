import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# (1) import nvflare client API
import nvflare.client as flare


def edge_index_gen(df_feat_class, df_edges):
    # Sort the data by Transaction_ID for consistent ordering
    df_feat_class = df_feat_class.sort_values(by="Transaction_ID").reset_index(drop=True)

    # Generate UETR-index map with the feature list
    node_id = df_feat_class["Transaction_ID"].values
    map_id = {j: i for i, j in enumerate(node_id)}  # mapping nodes to indexes

    # Get class labels
    label = df_feat_class["Fraud_Label"].values

    # Map UETR to indexes in the edge map
    edges = df_edges.copy()
    edges.UETR_1 = edges.UETR_1.map(map_id)
    edges.UETR_2 = edges.UETR_2.map(map_id)
    
    # Handle any unmapped edges (from synthetic samples or missing nodes)
    edges = edges.dropna()  # Remove edges with unmapped nodes
    edges = edges.astype(int)

    # for undirected graph
    edge_index = np.array(edges.values).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weight = torch.tensor([1] * edge_index.shape[1], dtype=torch.float)

    # UETR mapped to corresponding indexes, drop UETR and class
    node_feat = df_feat_class.drop(["Transaction_ID", "Fraud_Label"], axis=1).copy()
    
    # Ensure all columns are numeric
    # Convert any remaining object columns to numeric (in case some weren't encoded)
    for col in node_feat.columns:
        if node_feat[col].dtype == 'object':
            print(f"Warning: Column {col} has object dtype, converting to numeric")
            # Try to convert to numeric, fill NaN with 0
            node_feat[col] = pd.to_numeric(node_feat[col], errors='coerce').fillna(0)
    
    # Convert to float32 for PyTorch
    node_feat = node_feat.astype(np.float32)
    node_feat = torch.tensor(node_feat.values, dtype=torch.float)

    return node_feat, edge_index, weight, node_id, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
    )
    args = parser.parse_args()

    # (2) initializes NVFlare client API
    flare.init()
    site_name = flare.get_site_name()

    # Set up tensorboard
    writer = SummaryWriter(os.path.join(args.output_path, site_name))

    # Load the data
    dataset_names = ["train", "test"]

    node_features = {}
    edge_indices = {}
    weights = {}
    node_ids = {}
    labels = {}

    for ds_name in dataset_names:
        # IMPORTANT: Use GNN-ready files instead of normalized files
        file_name = os.path.join(args.data_path, site_name, f"{ds_name}_gnn_ready.csv")
        print(f"Loading GNN-ready data from: {file_name}")
        df = pd.read_csv(file_name, index_col=0)
        
        # Verify Transaction_ID exists
        if "Transaction_ID" not in df.columns:
            raise ValueError(f"Transaction_ID column missing from {file_name}")
        
        # These columns are already encoded but NOT scaled (which is good for GNN)
        # Just ensure we're not using the columns that don't help with GNN
        df_feat_class = df
        
        # Get edge map
        file_name = os.path.join(args.data_path, site_name, f"{ds_name}_edgemap.csv")
        df = pd.read_csv(file_name, header=None)
        # Add column names to the edge map
        df.columns = ["UETR_1", "UETR_2"]
        df_edges = df

        # Preprocess data
        node_feat, edge_index, weight, node_id, label = edge_index_gen(
            df_feat_class, df_edges
        )
        
        print(f"Dataset {ds_name}: {len(node_id)} nodes, {edge_index.shape[1]} edges")
        
        node_features[ds_name] = node_feat
        edge_indices[ds_name] = edge_index
        weights[ds_name] = weight
        node_ids[ds_name] = node_id
        labels[ds_name] = label

    # Check for class imbalance in training data
    unique_labels, counts = np.unique(labels["train"], return_counts=True)
    print(f"Training class distribution: {dict(zip(unique_labels, counts))}")
    
    # Calculate class weights for handling imbalanced data (instead of SMOTE)
    class_weights = len(labels["train"]) / (len(unique_labels) * counts)
    class_weight_dict = dict(zip(unique_labels, class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Converting training data to PyG graph data format
    train_data = Data(
        x=node_features["train"],
        edge_index=edge_indices["train"],
        edge_attr=weights["train"],
    )

    # Define the dataloader for graphsage training
    loader = LinkNeighborLoader(
        train_data,
        batch_size=2048,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
        num_workers=6,
        persistent_workers=True,
    )

    # Model
    model = GraphSAGE(
        in_channels=node_features["train"].shape[1],
        hidden_channels=32,
        num_layers=2,
        out_channels=32,
    )

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}/{input_model.total_rounds}")

        # (4) loads model from NVFlare
        # Handle first round where server model might have different dimensions
        if input_model.current_round == 0:
            try:
                model.load_state_dict(input_model.params)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Model dimension mismatch on first round. Using local initialization.")
                    print(f"Local model expects {node_features['train'].shape[1]} input features")
                    # Don't load the mismatched model, use local initialization
                else:
                    raise e
        else:
            # After first round, dimensions should match
            model.load_state_dict(input_model.params)

        # (5) perform encoding for both training and test data
        def gnn_encode(model_param, node_feature, edge_index, id, label):
            # Load the model and perform inference / encoding
            model_enc = GraphSAGE(
                in_channels=node_feature.shape[1],
                hidden_channels=32,  # Reduced from 64
                num_layers=2,
                out_channels=32,  # Reduced from 64
            )
            model_enc.load_state_dict(model_param)
            model_enc.to(DEVICE)
            model_enc.eval()
            node_feature = node_feature.to(DEVICE)
            edge_index = edge_index.to(DEVICE)

            # Perform encoding
            h = model_enc(node_feature, edge_index)
            embed = pd.DataFrame(h.cpu().detach().numpy())
            # Add column names as V_0, V_1, ... V_31 (now 32 dimensions)
            embed.columns = [f"V_{i}" for i in range(embed.shape[1])]
            # Concatenate the node ids and class labels with the encoded features
            embed["Transaction_ID"] = id
            embed["Fraud_Label"] = label
            # Move the UETR and Class columns to the front
            embed = embed[
                ["Transaction_ID", "Fraud_Label"]
                + [
                    col
                    for col in embed.columns
                    if col not in ["Transaction_ID", "Fraud_Label"]
                ]
            ]
            return embed

        # Only do encoding for the last round
        if input_model.current_round == input_model.total_rounds - 1:
            print("Encoding the data with the final model")
            for ds_name in dataset_names:
                embed = gnn_encode(
                    input_model.params,
                    node_features[ds_name],
                    edge_indices[ds_name],
                    node_ids[ds_name],
                    labels[ds_name],
                )
                embed.to_csv(
                    os.path.join(
                        args.output_path, site_name, f"{ds_name}_embedding.csv"
                    ),
                    index=False,
                )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.to(DEVICE)
        steps = args.epochs * len(loader)
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = instance_count = 0
            for data in loader:
                # get the inputs data
                data = data.to(DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                h = model(data.x, data.edge_index)
                h_src = h[data.edge_label_index[0]]
                h_dst = h[data.edge_label_index[1]]
                link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.
                
                # Use regular loss without class weights for link prediction
                # (class weights are more relevant for node classification)
                loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
                
                loss.backward()
                optimizer.step()
                # add record
                running_loss += float(loss.item()) * link_pred.numel()
                instance_count += link_pred.numel()
            print(f"Epoch: {epoch:02d}, Loss: {running_loss / instance_count:.4f}")
            writer.add_scalar(
                "train_loss",
                running_loss / instance_count,
                input_model.current_round * args.epochs + epoch,
            )

        print("Finished Training")
        # Save the model
        torch.save(
            model.state_dict(), os.path.join(args.output_path, site_name, "model.pt")
        )

        # (6) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"loss": running_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (7) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()