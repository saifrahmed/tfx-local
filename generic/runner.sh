#!/bin/bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -u

# Output: dir for our raw=>transform function
DATE=`date "+%Y_%m_%d___%H_%M_%S"`
WORKING_DIR=./working_dir_$DATE
MODEL_DIR=$WORKING_DIR/trainer_output
mkdir $WORKING_DIR
mkdir $MODEL_DIR
DATA_DIR=./data
export SCHEMA_PATH=$WORKING_DIR/schema.pbtxt

# Output: dir for both the serving model and eval_model which will go into tfma
# evaluation

#rm -R -f $WORKING_DIR/serving_model_dir
#rm -R -f $WORKING_DIR/eval_model_dir
#rm -R -f $MODEL_DIR
rm -R -f ./data/train/local_chicago_taxi_output
rm -R -f ./data/eval/local_chicago_taxi_output

echo Working directory: $WORKING_DIR
echo Serving model directory: $WORKING_DIR/serving_model_dir
echo Eval model directory: $WORKING_DIR/eval_model_dir

echo Starting local TFDV preprocessing...

python tfdv_analyze_and_validate.py \
  --input $DATA_DIR/train.csv \
  --stats_path $WORKING_DIR/train_stats.tfrecord \
  --infer_schema \
  --schema_path $WORKING_DIR/schema.pbtxt \
  --runner DirectRunner

echo Starting local TFT preprocessing...

# Preprocess the train files, keeping the transform functions
echo Preprocessing train data...
python preprocess.py \
  --input ./data/train.csv \
  --schema_file $WORKING_DIR/schema.pbtxt \
  --output_dir $WORKING_DIR \
  --outfile_prefix train_transformed \
  --runner DirectRunner

# Preprocess the eval files
echo Preprocessing eval data...
python preprocess.py \
  --input ./data/eval.csv \
  --schema_file $WORKING_DIR/schema.pbtxt \
  --output_dir $WORKING_DIR \
  --outfile_prefix eval_transformed \
  --transform_dir $WORKING_DIR \
  --runner DirectRunner

echo Starting local training...
python trainer/task.py \
    --train-files $WORKING_DIR/train_transformed-* \
    --verbosity INFO \
    --job-dir $MODEL_DIR \
    --train-steps 10000 \
    --eval-steps 5000 \
    --tf-transform-dir $WORKING_DIR \
    --output-dir $WORKING_DIR \
    --schema-file $WORKING_DIR/schema.pbtxt \
    --eval-files $WORKING_DIR/eval_transformed-*
