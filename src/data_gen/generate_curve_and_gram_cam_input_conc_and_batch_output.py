import tensorflow as tf
from typing import Dict, Optional, Text
import glob

from src.data_gen import generate_curve_input_conc_output

def separate_features_and_labels(features: Dict, label_columns: list) -> Dict:
  heatmap_tensors = []
  for batch_id in range(10):
    heatmap_tensors.append(features['feature/image/avg/heatmap_%d' % batch_id])
  return tf.keras.applications.mobilenet_v2.preprocess_input(features['feature/image/avg']), tf.stack(heatmap_tensors), tf.concat(
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
  dataset = dataset.map(generate_curve_input_conc_output.decode(add_gram_cam_features=True), num_parallel_calls=2)
  dataset = dataset.filter(lambda x: tf.reduce_any(tf.math.is_nan(x['feature/image/avg'])) == False)
  dataset = dataset.map(lambda x: separate_features_and_labels(x, label_columns))
  dataset = dataset.repeat(repeat)
  dataset = dataset.shuffle(2048)
  dataset = dataset.batch(batch_size).prefetch(prefetch_size)
  return dataset