import argparse
import os
import random
import shutil

import pandas as pd

# List of example BICs for demonstration
from sklearn.model_selection import train_test_split

# %load_ext cudf.pandas
import argparse
import os
import random

from sklearn.model_selection import train_test_split


# List of example BICs for demonstration, BIC and names are random created, they are fakes.
bic_list = {
    "ZHSZUS33": "New York",  # Bank 1
    "SHSHKHH1": "Hong Kong",  # bank 2
    "YXRXGB22": "London",  # bank 3
    "WPUWDEFF": "Berlin",  # bank 4
    "YMNYFRPP": "Paris",  # bank 5
    "FBSFCHZH": "Zurich",  # Bank 6
    "YSYCESMM": "Mumbai",  # bank 7
    "ZNZZAU3M": "Sydney",  # Bank 8
    "HCBHSGSG": "Tokyo",  # bank 9
    "XITXUS33": "New York",  # bank 10
}
bic_list_rev = {v: k for k, v in bic_list.items()}

# List of currencies and their respective countries
currencies = {
    "USD": "New York",
    "GBP": "London",
    "JPY": "Tokyo",
    "AUD": "Sydney",
    "INR": "Mumbai",
}


# BIC to Bank Name mapping
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


# Function to generate random BICs and currency details
def generate_random_details(df):
    # Ensure the currency and beneficiary BIC match
    def match_currency_and_bic():
        while True:
            currency = random.choice(list(currencies.keys()))
            country = currencies[currency]
            matching_bics = [
                bic for bic, bic_country in bic_list.items() if bic_country == country
            ]
            if matching_bics:
                return currency, random.choice(matching_bics)

    df["Sender_BIC"] = [bic_list_rev[loc] for loc in df["Location"]]
    df["Receiver_BIC"] = [random.choice(list(bic_list.keys())) for _ in range(len(df))]
    

    df["Currency"], df["Beneficiary_BIC"] = zip(
        *[match_currency_and_bic() for _ in range(len(df))]
    )
    df["Currency_Country"] = df["Currency"].map(currencies)

    return df


def split_datasets(df, out_folder: str, hist_ratio=0.55, train_ratio=0.35):
    # Sort the DataFrame by the Time column
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Calculate the number of samples for each split
    total_size = len(df)
    historical_size = int(total_size * 0.55)
    train_size = int(total_size * 0.35)
    test_size = total_size - historical_size - train_size

    # Split into historical and remaining data
    df_history = df.iloc[:historical_size]
    remaining_df = df.iloc[historical_size:]
    y = remaining_df["Fraud_Label"]

    ds = remaining_df.drop("Fraud_Label", axis=1)
    # Split the remaining data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        ds, y, test_size=test_size / (train_size + test_size), random_state=42
    )

    df_train = pd.concat([y_train, x_train], axis=1)
    df_test = pd.concat([y_test, x_test], axis=1)

    # Display sizes of each dataset
    print(f"Historical DataFrame size: {len(df_history)}")
    print(f"Training DataFrame size: {len(df_train)}")
    print(f"Testing DataFrame size: {len(df_test)}")

    # Save training and testing sets
    os.makedirs(out_folder, exist_ok=True)

    df_train.to_csv(path_or_buf=os.path.join(out_folder, "train.csv"), index=False)
    df_test.to_csv(path_or_buf=os.path.join(out_folder, "test.csv"), index=False)
    df_history.to_csv(path_or_buf=os.path.join(out_folder, "history.csv"), index=False)


def split_site_datasets(out_folder):
    files = ["history", "train", "test"]
    client_names = set()

    for f in files:
        file_path = os.path.join(out_folder, f + ".csv")
        df = pd.read_csv(file_path)
        # Group the DataFrame by 'Sender_BIC'
        grouped = df.groupby("Sender_BIC")
        # Save each group to a separate file
        for name, group in grouped:
            bank_name = bic_to_bank[name].replace(" ", "_")
            client_name = f"{name}_{bank_name}"
            client_names.add(client_name)
            site_dir = os.path.join(out_folder, client_name)
            os.makedirs(site_dir, exist_ok=True)

            filename = os.path.join(site_dir, f"{f}.csv")
            group.to_csv(filename, index=False)
            print(f"Saved {name} {f} transactions to {filename}")

    print(f"client_names: {sorted(client_names)}")


def main():
    args = define_parser()

    input_data_path = args.input_data_path
    out_folder = args.output_dir

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    df = pd.read_csv(input_data_path)

    # Add random BIC and currency details to the DataFrame
    df = generate_random_details(df)

    # To avoid data leakage drop the Risk_Score
    df.drop(columns=["Risk_Score"], inplace=True)

    # Add useful derived features
    df["Transaction_Ratio"] = df["Daily_Transaction_Count"] / (df["Avg_Transaction_Amount_7d"] + 1)
    df["Failed_Transaction_Rate"] = df["Failed_Transaction_Count_7d"] / (df["Daily_Transaction_Count"] + 1)
    df["Fraud_History_Impact"] = df["Previous_Fraudulent_Activity"] / (df["Card_Age"] + 1)

    df.drop(columns=["Daily_Transaction_Count", "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d", "Previous_Fraudulent_Activity", "Card_Age"], inplace=True)
    df.head()

    split_datasets(df, out_folder=out_folder, hist_ratio=0.55, train_ratio=0.35)
    split_site_datasets(out_folder)


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_data_path",
        type=str,
        nargs="?",
        help="input data path for credit car csv file path",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        help="output directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
