import tensorflow as tf
from typing import Dict, Optional, Text
import glob
import numpy as np

from src.data_gen import generate_curve_input_conc_output

def one_hot_encoder_from_numpy(batch_id_numpy, num_batches):
  encoded_batch = np.array([0.0] * num_batches)
  encoded_batch[int(batch_id_numpy[0])] = 1.0
  return encoded_batch

def one_hot_encoder(batch_id_tensor, num_batches):
  result = tf.numpy_function(one_hot_encoder_from_numpy, [batch_id_tensor], tf.double)
  result = tf.reshape(result, (num_batches, ))
  return result
def separate_features_and_selector_as_labels(features: Dict, num_batches: int) -> Dict:
  return tf.keras.applications.mobilenet_v2.preprocess_input(features['feature/image/avg']), tf.concat(
    axis=-1,values=[one_hot_encoder(features['metadata/batch_id'], num_batches)])

def separate_features_and_regressor_as_labels(features: Dict, label_columns: list) -> Dict:
  return tf.keras.applications.mobilenet_v2.preprocess_input(features['feature/image/avg']), tf.concat(
    axis=-1,values=[features[label_columns[0]], features[label_columns[1]], features[label_columns[2]], features[label_columns[3]]])

def load_dataset(filename_pattern: Text, 
                 label_columns: list, 
                 batch_size: int, 
                 prefetch_size: int, 
                 selector_as_labels: bool,
                 num_batches: int = 16,
                 repeat: Optional[int] = None):
  filenames = [filename_pattern] if '*' not in filename_pattern else glob.glob(filename_pattern)
  print(filenames)
  dataset = tf.data.Dataset.list_files(filenames).interleave(
      lambda filepath: tf.data.TFRecordDataset(filepath), cycle_length=2,)
  dataset = dataset.map(generate_curve_input_conc_output.decode, num_parallel_calls=2)
  dataset = dataset.filter(lambda x: tf.reduce_any(tf.math.is_nan(x['feature/image/avg'])) == False)
  if selector_as_labels:
    dataset = dataset.map(lambda x: separate_features_and_selector_as_labels(x, num_batches))
  else: 
    dataset = dataset.map(lambda x: separate_features_and_regressor_as_labels(x, label_columns))
  dataset = dataset.repeat(repeat)
  dataset = dataset.shuffle(2048)
  dataset = dataset.batch(batch_size).prefetch(prefetch_size)
  return dataset