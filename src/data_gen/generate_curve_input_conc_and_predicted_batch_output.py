import tensorflow as tf
from typing import Dict, Optional, Text
import glob

from src.data_gen import generate_curve_input_conc_output

class CurveInputConcAndPredictedBatchOutputGenerator:
  def __init__(self, batch_id_model_path: Text):
    self.batch_id_model_path = batch_id_model_path

  def load_batch_id_model(self):
    self.batch_id_model = tf.keras.models.load_model(self.batch_id_model_path)

  def separate_features_and_labels(self, features: Dict, label_columns: list) -> Dict:
    input_features = tf.keras.applications.mobilenet_v2.preprocess_input(features['feature/image/avg'])
    print(self.batch_id_model(input_features).shape)
    return input_features, tf.concat(
      axis=-1,values=[features[label_columns[0]], features[label_columns[1]], features[label_columns[2]], features[label_columns[3]]])

  def add_predicted_batch_id(self, data_set):
    image_features, conc_labels = data_set
    return image_features, (self.batch_id_model(image_features), conc_labels)


  def load_dataset(self, 
                   filename_pattern: Text, 
                   label_columns: list, 
                   batch_size: int, 
                   prefetch_size: int, 
                   repeat: Optional[int] = None):
    filenames = [filename_pattern] if '*' not in filename_pattern else glob.glob(filename_pattern)
    print(filenames)
    dataset = tf.data.Dataset.list_files(filenames).interleave(
        lambda filepath: tf.data.TFRecordDataset(filepath), cycle_length=2,)
    dataset = dataset.map(generate_curve_input_conc_output.decode, num_parallel_calls=2)
    dataset = dataset.filter(lambda x: tf.reduce_any(tf.math.is_nan(x['feature/image/avg'])) == False)
    dataset = dataset.map(lambda x: self.separate_features_and_labels(x, label_columns))
    dataset = dataset.repeat(repeat)
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size).prefetch(prefetch_size)
    dataset = dataset.map(self.add_predicted_batch_id)
    return dataset