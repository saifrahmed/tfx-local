from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import tensorflow_model_analysis as tfma

from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2

from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io



# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]

# TODO(b/114589822): Infer these and other properties from the schema.
CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = [
    'payment_type',
    'company',
]

#LABEL_KEY = 'tips'
#LABEL_KEY = 'trip_miles'
LABEL_KEY = 'trip_seconds'
#FARE_KEY = 'fare'

CSV_COLUMN_NAMES = [
    'pickup_community_area',
    'fare',
    'trip_start_month',
    'trip_start_hour',
    'trip_start_day',
    'trip_start_timestamp',
    'pickup_latitude',
    'pickup_longitude',
    'dropoff_latitude',
    'dropoff_longitude',
    'trip_miles',
    'pickup_census_tract',
    'dropoff_census_tract',
    'payment_type',
    'company',
    'trip_seconds',
    'dropoff_community_area',
    'tips',
]


def build_estimator(tf_transform_dir, config, hidden_units=None):
  """Build an estimator for predicting the tipping behavior of taxi riders.

  Args:
    tf_transform_dir: directory in which the tf-transform model was written
      during the preprocessing step.
    config: tf.contrib.learn.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    hidden_units: [int], the layer sizes of the DNN (input layer first)

  Returns:
    Resulting DNNLinearCombinedClassifier.
  """
  metadata_dir = os.path.join(tf_transform_dir,
                              transform_fn_io.TRANSFORMED_METADATA_DIR)
  transformed_metadata = metadata_io.read_metadata(metadata_dir)
  transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

  transformed_feature_spec.pop(transformed_name(LABEL_KEY))

  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in transformed_names(DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=VOCAB_SIZE + OOV_SIZE, default_value=0)
      for key in transformed_names(VOCAB_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=FEATURE_BUCKET_COUNT, default_value=0)
      for key in transformed_names(BUCKET_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=num_buckets, default_value=0)
      for key, num_buckets in zip(
          transformed_names(CATEGORICAL_FEATURE_KEYS),  #
          MAX_CATEGORICAL_FEATURE_VALUES)
  ]
  #return tf.estimator.DNNLinearCombinedClassifier(
  return tf.estimator.DNNLinearCombinedRegressor(
      config=config,
      linear_feature_columns=categorical_columns,
      dnn_feature_columns=real_valued_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25])


def example_serving_receiver_fn(tf_transform_dir, schema):
  """Build the serving in inputs.

  Args:
    tf_transform_dir: directory in which the tf-transform model was written
      during the preprocessing step.
    schema: the schema of the input data.

  Returns:
    Tensorflow graph which parses examples, applying tf-transform to them.
  """
  raw_feature_spec = get_raw_feature_spec(schema)
  raw_feature_spec.pop(LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  _, transformed_features = (
      saved_transform_io.partially_apply_saved_transform(
          os.path.join(tf_transform_dir, transform_fn_io.TRANSFORM_FN_DIR),
          serving_input_receiver.features))

  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)


def eval_input_receiver_fn(tf_transform_dir, schema):
  """Build everything needed for the tf-model-analysis to run the model.

  Args:
    tf_transform_dir: directory in which the tf-transform model was written
      during the preprocessing step.
    schema: the schema of the input data.

  Returns:
    EvalInputReceiver function, which contains:
      - Tensorflow graph which parses raw untranformed features, applies the
        tf-transform preprocessing operators.
      - Set of raw, untransformed features.
      - Label against which predictions will be compared.
  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = get_raw_feature_spec(schema)

  serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

  # Add a parse_example operator to the tensorflow graph, which will parse
  # raw, untransformed, tf examples.
  features = tf.parse_example(serialized_tf_example, raw_feature_spec)

  # Now that we have our raw examples, process them through the tf-transform
  # function computed during the preprocessing step.
  _, transformed_features = (
      saved_transform_io.partially_apply_saved_transform(
          os.path.join(tf_transform_dir, transform_fn_io.TRANSFORM_FN_DIR),
          features))

  # The key name MUST be 'examples'.
  receiver_tensors = {'examples': serialized_tf_example}

  # NOTE: Model is driven by transformed features (since training works on the
  # materialized output of TFT, but slicing will happen on raw features.
  features.update(transformed_features)

  return tfma.export.EvalInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors,
      labels=transformed_features[transformed_name(LABEL_KEY)])


def _gzip_reader_fn():
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.TFRecordReader(
      options=tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def input_fn(filenames, tf_transform_dir, batch_size=200):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    tf_transform_dir: directory in which the tf-transform model was written
      during the preprocessing step.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """
  metadata_dir = os.path.join(tf_transform_dir,
                              transform_fn_io.TRANSFORMED_METADATA_DIR)
  transformed_metadata = metadata_io.read_metadata(metadata_dir)
  transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

  transformed_features = tf.contrib.learn.io.read_batch_features(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

  # We pop the label because we do not want to use it as a feature while we're
  # training.
  return transformed_features, transformed_features.pop(
      transformed_name(LABEL_KEY))


def transformed_name(key):
  return key + '_xf'


def transformed_names(keys):
  return [transformed_name(key) for key in keys]


# Tf.Transform considers these features as "raw"
def get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def make_proto_coder(schema):
  raw_feature_spec = get_raw_feature_spec(schema)
  raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  return tft_coders.ExampleProtoCoder(raw_schema)


def make_csv_coder(schema):
  """Return a coder for tf.transform to read csv files."""
  raw_feature_spec = get_raw_feature_spec(schema)
  parsing_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  return tft_coders.CsvCoder(CSV_COLUMN_NAMES, parsing_schema)


def clean_raw_data_dict(input_dict, raw_feature_spec):
  """Clean raw data dict."""
  output_dict = {}

  for key in raw_feature_spec:
    if key not in input_dict or not input_dict[key]:
      output_dict[key] = []
    else:
      output_dict[key] = [input_dict[key]]
  return output_dict


def make_sql(table_name, max_rows=None, for_eval=False):
  """Creates the sql command for pulling data from BigQuery.

  Args:
    table_name: BigQuery table name
    max_rows: if set, limits the number of rows pulled from BigQuery
    for_eval: True if this is for evaluation, false otherwise

  Returns:
    sql command as string
  """
  if for_eval:
    # 1/3 of the dataset used for eval
    where_clause = 'WHERE MOD(FARM_FINGERPRINT(unique_key), 3) = 0'
  else:
    # 2/3 of the dataset used for training
    where_clause = 'WHERE MOD(FARM_FINGERPRINT(unique_key), 3) > 0'

  limit_clause = ''
  if max_rows:
    limit_clause = 'LIMIT {max_rows}'.format(max_rows=max_rows)
  return """
  SELECT
      CAST(pickup_community_area AS string) AS pickup_community_area,
      CAST(dropoff_community_area AS string) AS dropoff_community_area,
      CAST(pickup_census_tract AS string) AS pickup_census_tract,
      CAST(dropoff_census_tract AS string) AS dropoff_census_tract,
      fare,
      EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
      EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
      EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
      UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
      pickup_latitude,
      pickup_longitude,
      dropoff_latitude,
      dropoff_longitude,
      trip_miles,
      payment_type,
      company,
      trip_seconds,
      tips
  FROM `{table_name}`
  {where_clause}
  {limit_clause}
""".format(
    table_name=table_name, where_clause=where_clause, limit_clause=limit_clause)


def read_schema(path):
  """Reads a schema from the provided location.

  Args:
    path: The location of the file holding a serialized Schema proto.

  Returns:
    An instance of Schema or None if the input argument is None
  """
  result = schema_pb2.Schema()
  contents = file_io.read_file_to_string(path)
  text_format.Parse(contents, result)
  return result


SERVING_MODEL_DIR = 'serving_model_dir'
EVAL_MODEL_DIR = 'eval_model_dir'

TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40

# Number of nodes in the first layer of the DNN
FIRST_DNN_LAYER_SIZE = 100
NUM_DNN_LAYERS = 4
DNN_DECAY_FACTOR = 0.7


def train_and_maybe_evaluate(hparams):
  """Run the training and evaluate using the high level API.

  Args:
    hparams: Holds hyperparameters used to train the model as name/value pairs.

  Returns:
    The estimator that was used for training (and maybe eval)
  """
  schema = read_schema(hparams.schema_file)

  train_input = lambda: input_fn(
      hparams.train_files,
      hparams.tf_transform_dir,
      batch_size=TRAIN_BATCH_SIZE
  )

  eval_input = lambda: input_fn(
      hparams.eval_files,
      hparams.tf_transform_dir,
      batch_size=EVAL_BATCH_SIZE
  )

  train_spec = tf.estimator.TrainSpec(
      train_input, max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: example_serving_receiver_fn(
      hparams.tf_transform_dir, schema)

  exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='chicago-taxi-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  serving_model_dir = os.path.join(hparams.output_dir, SERVING_MODEL_DIR)
  run_config = run_config.replace(model_dir=serving_model_dir)

  estimator = build_estimator(
      hparams.tf_transform_dir,

      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(FIRST_DNN_LAYER_SIZE * DNN_DECAY_FACTOR**i))
          for i in range(NUM_DNN_LAYERS)
      ],
      config=run_config)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  return estimator


def run_experiment(hparams):
  """Train the model then export it for tf.model_analysis evaluation.

  Args:
    hparams: Holds hyperparameters used to train the model as name/value pairs.
  """
  estimator = train_and_maybe_evaluate(hparams)

  schema = read_schema(hparams.schema_file)

  # Save a model for tfma eval
  eval_model_dir = os.path.join(hparams.output_dir, EVAL_MODEL_DIR)

  receiver_fn = lambda: eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      hparams.tf_transform_dir, schema)

  tfma.export.export_eval_savedmodel(
      estimator=estimator,
      export_dir_base=eval_model_dir,
      eval_input_receiver_fn=receiver_fn)


def main():
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True)

  parser.add_argument(
      '--tf-transform-dir',
      help='Tf-transform directory with model from preprocessing step',
      required=True)

  parser.add_argument(
      '--output-dir',
      help="""\
      Directory under which which the serving model (under /serving_model_dir)\
      and the tf-mode-analysis model (under /eval_model_dir) will be written\
      """,
      required=True)

  parser.add_argument(
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True)
  # Training arguments
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True)

  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO',
  )
  # Experiment arguments
  parser.add_argument(
      '--train-steps',
      help='Count of steps to run the training job for',
      required=True,
      type=int)
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int)
  parser.add_argument(
      '--schema-file',
      help='File holding the schema for the input data')

  args = parser.parse_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  hparams = tf.contrib.training.HParams(**args.__dict__)
  run_experiment(hparams)


if __name__ == '__main__':
  main()
