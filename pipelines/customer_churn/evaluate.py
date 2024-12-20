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
"""Evaluation script for measuring model accuracy."""

import json
import tarfile
import logging
import pickle


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from datasets import load_dataset
from sentence_transformers.evaluation import TripletEvaluator


if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")

    logger.debug("Loading model.")
    model = pickle.load(open("checkpoint-2900", "rb")) # Change name when training script will be ready

    print("Loading test input data")
    test_path = "/opt/ml/processing/test/test.csv"
    dataset = load_dataset("csv", data_files=test_path) # CHANGE THIS TO LOADING FROM S3

    # Evaluate the trained model on the test set
    test_evaluator = TripletEvaluator(
        anchors=dataset["query"],
        positives=dataset["description"],
        negatives=dataset["neg_description"],
        name="movie-data-test",
    )
    
    print("Creating classification evaluation report")
    report_dict = test_evaluator(model)
    
    evaluation_output_path = '/opt/ml/processing/evaluation/evaluation.json'
    with open(evaluation_output_path, 'w') as f:
        f.write(json.dumps(report_dict))
    