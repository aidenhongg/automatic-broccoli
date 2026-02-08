#!/bin/bash
set -e

EXPERIMENT_ID=${AWS_BATCH_JOB_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
S3_OUTPUT_BUCKET="s3://pcrng-approx-dataset"


echo "Downloading dataset..."
aws s3 cp ${S3_OUTPUT_BUCKET}/data.csv ./data.csv

echo "starting experiment!"
python3 ./network.py


echo "uploading results to S3..."

aws s3 cp ./training_log.txt ${S3_OUTPUT_BUCKET}/results/${EXPERIMENT_ID}_results.txt

echo "Done."