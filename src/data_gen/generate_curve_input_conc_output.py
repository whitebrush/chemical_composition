import tensorflow as tf
from typing import Dict, Optional, Text
import glob

def decode(serialized_example, add_gram_cam_features=False):
  # Decode examples stored in TFRecord
  # NOTE: make sure to specify the correct dimensions for the images
  feature_map = {
          'feature/image/avg': tf.io.FixedLenFeature([160, 160, 3], tf.float32),
          'feature/image/diff': tf.io.FixedLenFeature([160, 160, 3], tf.float32),
          'feature/image/derivative_1': tf.io.FixedLenFeature([160, 160, 3], tf.float32),
          'feature/image/derivative_2': tf.io.FixedLenFeature([160, 160, 3], tf.float32),
          'feature/image/derivative_3': tf.io.FixedLenFeature([160, 160, 3], tf.float32),
          'feature/image/derivative_4': tf.io.FixedLenFeature([160, 160, 3], tf.float32),
          'feature/image/derivative_5': tf.io.FixedLenFeature([160, 160, 3], tf.float32),
          'label/5HT': tf.io.FixedLenFeature([1], tf.float32),
          'label/ME': tf.io.FixedLenFeature([1], tf.float32),
          'label/DA': tf.io.FixedLenFeature([1], tf.float32),
          'label/NE': tf.io.FixedLenFeature([1], tf.float32), 
          'metadata/batch_id': tf.io.FixedLenFeature([1], tf.float32) }
  if add_gram_cam_features:
    for batch_id in range(10):
      feature_map['feature/image/avg/heatmap_%d' % batch_id] = tf.io.FixedLenFeature([5, 5], tf.float32)
  feature_tensors = tf.io.parse_single_example(
      serialized_example,
      features=feature_map)

  # NOTE: No need to cast these features, as they are already `tf.float32` values.
  return feature_tensors

def separate_features_and_labels(features: Dict, label_columns: list) -> Dict:
  return tf.keras.applications.mobilenet_v2.preprocess_input(features['feature/image/avg']), tf.concat(
    axis=-1,values=[features[label_columns[0]], features[label_columns[1]], features[label_columns[2]], features[label_columns[3]]])

def load_dataset(filename_pattern: Text, 
                 label_columns: list, 
                 batch_size: int, 
                 prefetch_size: int, 
                 repeat: Optional[int] = None):
  filenames = [filename_pattern] if '*' not in filename_pattern else glob.glob(filename_pattern)
  print(filenames)
  dataset = tf.data.Dataset.list_files(filenames).interleave(
      lambda filepath: tf.data.TFRecordDataset(filepath), cycle_length=2,)
  dataset = dataset.map(decode, num_parallel_calls=2)
  dataset = dataset.filter(lambda x: tf.reduce_any(tf.math.is_nan(x['feature/image/avg'])) == False)
  dataset = dataset.map(lambda x: separate_features_and_labels(x, label_columns))
  dataset = dataset.repeat(repeat)
  dataset = dataset.shuffle(2048)
  dataset = dataset.batch(batch_size).prefetch(prefetch_size)
  return dataset