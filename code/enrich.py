import argparse
import os

import pandas as pd

# (1) import nvflare client API
import nvflare.client as flare


def main():
    print("\n enrichment starts \n ")

    args = define_parser()
    flare.init()

    input_dir = args.input_dir
    output_dir = args.output_dir

    site_name = flare.get_site_name()
    print(f"\n {site_name=} \n ")

    # receives global message from NVFlare
    etl_task = flare.receive()
    merged_dfs = enrichment(input_dir, site_name)

    for ds_name in merged_dfs:
        save_to_csv(merged_dfs[ds_name], output_dir, site_name, ds_name)

    # send message back the controller indicating end.
    etl_task.meta["status"] = "done"
    flare.send(etl_task)


def save_to_csv(merged_df, output_dir, site_name, ds_name):
    site_dir = os.path.join(output_dir, site_name)
    os.makedirs(site_dir, exist_ok=True)
    enrich_file_name = os.path.join(site_dir, f"{ds_name}_enrichment.csv")
    print(enrich_file_name)
    merged_df.to_csv(enrich_file_name)


def load_hist_data(hist_file_name: str):
    return pd.read_csv(hist_file_name)


def get_hist_summary(df_history, group_by_name: str):
    history_summary = (
        df_history.groupby(group_by_name)
        .agg(
            hist_trans_volume=("Transaction_Amount", "count"),
            hist_total_amount=("Transaction_Amount", "sum"),
            hist_average_amount=("Transaction_Amount", "mean"),
        )
        .reset_index()
    )
    return history_summary


def enrichment(input_dir, site_name) -> dict:
    hist_file_name = os.path.join(input_dir, site_name, "history.csv")
    df_history = load_hist_data(hist_file_name)
    hist_currency_summary_df = get_hist_summary(df_history, "Currency")
    hist_ben_bic_summary_df = get_hist_summary(
        df_history, "Beneficiary_BIC"
    ).reset_index()

    dataset_names = ["train", "test"]
    results = {}
    results2 = {}

    for ds_name in dataset_names:
        file_name = os.path.join(input_dir, site_name, f"{ds_name}.csv")
        ds_df = pd.read_csv(file_name)
        ds_df["Timestamp"] = pd.to_datetime(ds_df["Timestamp"])

        # Set the Time column as the index
        ds_df.set_index("Timestamp", inplace=True)

        resampled_df = (
            ds_df.resample("1H")
            .agg(
                trans_volume=("Transaction_Amount", "count"),
                total_amount=("Transaction_Amount", "sum"),
                average_amount=("Transaction_Amount", "mean"),
            )
            .reset_index()
        )

        add_enrich_feature(
            ds_df,
            ds_name,
            hist_currency_summary_df,
            resampled_df,
            "Currency",
            "currency_deviation_ratio",
            results,
        )
        add_enrich_feature(
            ds_df,
            ds_name,
            hist_ben_bic_summary_df,
            resampled_df,
            "Beneficiary_BIC",
            "beneficiary_deviation_ratio",
            results2,
        )

    final_results = {}
    for ds_name in results:
        df = results[ds_name]
        df2 = results2[ds_name]
        df3 = df2[["Timestamp", "Beneficiary_BIC", "beneficiary_deviation_ratio"]].copy()
        df4 = pd.merge(df, df3, on=["Timestamp", "Beneficiary_BIC"])
        final_results[ds_name] = df4

    return final_results


def add_enrich_feature(
    ds_df, ds_name, hist_summary_df, resampled_df, key, new_feature, results
):
    c_df = ds_df[[key]].resample("1H").agg({key: "first"}).reset_index()
    # Add Currency_Country to the resampled data by joining with the original DataFrame
    resampled_df2 = pd.merge(resampled_df, c_df, on="Timestamp")
    resampled_df3 = pd.merge(resampled_df2, hist_summary_df, on=key)
    resampled_df4 = resampled_df3.copy()
    resampled_df4[new_feature] = (
        resampled_df4["average_amount"] / resampled_df4["hist_trans_volume"] + 1
    )
    ds_df = ds_df.sort_values("Timestamp")
    resampled_df4 = resampled_df4.sort_values("Timestamp")
    merged_df = pd.merge_asof(ds_df, resampled_df4, on="Timestamp")
    merged_df = merged_df.drop(columns=[f"{key}_y"]).rename(columns={f"{key}_x": key})
    results[ds_name] = merged_df

    return merged_df


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
