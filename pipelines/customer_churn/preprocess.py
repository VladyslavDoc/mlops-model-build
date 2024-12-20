# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Feature engineers the customer churn dataset."""
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/hf/train/adventure_w_queries_and_negatives.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")

    # read in csv
    df = pd.read_csv(fn)

    # Split dataset
    # Select random 10% for validation and 10% for testing

    # Define the proportions for train, validation, and test
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the sizes of each subset
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)

    # Split the DataFrame
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:train_size + val_size]
    test_df = df_shuffled[train_size + val_size:]

    # Check the sizes
    print("Training set size:", len(train_df))
    print("Validation set size:", len(val_df))
    print("Testing set size:", len(test_df))

    train_df.to_csv(
        f"{base_dir}/train/train.csv", header=False, index=False
    )
    val_df.to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    test_df.to_csv(
        f"{base_dir}/test/test.csv", header=False, index=False
    )
    