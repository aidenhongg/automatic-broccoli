#!/bin/bash
set -e

EXPERIMENT_ID=${AWS_BATCH_JOB_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
S3_BUCKET="s3://pcrng-approx-dataset"

echo "Downloading dataset..."
aws s3 cp "${S3_BUCKET}/data.csv" ./data.csv

echo "Starting experiment..."
python3 ./network.py

echo "Uploading results to S3..."
aws s3 cp ./training_log.txt "${S3_BUCKET}/results/${EXPERIMENT_ID}_results.txt"

echo "Done."