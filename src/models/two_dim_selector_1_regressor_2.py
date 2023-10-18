import os
import tensorflow as tf
import tensorflow_addons as tfa

from src.models import two_dim_and_finetune
from src.models import two_dim_hard_regressor_selector

def build_CNN_selector_model(num_batches=16):
  image_size = (160, 160)
  num_channels = 3
  inputs = tf.keras.Input(shape=image_size + (num_channels,))
  pretrained_model = two_dim_and_finetune.load_pretrained_model(
    image_size=image_size,
    num_channels=num_channels, 
    include_top=False, 
    weights='imagenet', 
    trainable=False, 
    num_top_trainable_layers=0)
  x = pretrained_model(inputs)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Normalization()(x)
  x_batch_id = tf.keras.layers.Dropout(0.2)(x)
  x_batch_id = tf.keras.layers.Dense(64)(x_batch_id)
  x_batch_id = tf.keras.layers.Dropout(0.2)(x_batch_id)
  x_batch_id = tf.keras.layers.Dense(16)(x_batch_id)
  x_batch_id = tf.keras.layers.Dropout(0.2)(x_batch_id)
  outputs_batch_id = tf.keras.layers.Dense(num_batches, activation='softmax')(x_batch_id)
  outputs_one_hot_predicted_batch = tf.keras.layers.Reshape([num_batches])(outputs_batch_id)
  model = tf.keras.Model(inputs, outputs_one_hot_predicted_batch)
  base_learning_rate = 0.0001
  tf.keras.backend.set_epsilon(0.1)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
  return model, pretrained_model


def train_CNN_selector_model(train_dataset, val_dataset, total_epochs, num_batches, model_dir):
  model, pretrained_model = build_CNN_selector_model(num_batches)
  pretrained_model = two_dim_and_finetune.freeze_layers(pretrained_model, 0)
  model.summary()
  history = model.fit(train_dataset, epochs=total_epochs, initial_epoch=0, validation_data=val_dataset)
  model_path = os.path.join(model_dir, 'selector_model')
  model.save(model_path)

def build_CNN_regressor_model(num_batches=16):
  image_size = (160, 160)
  num_channels = 3
  image_inputs = tf.keras.Input(shape=image_size + (num_channels,))
  selector_inputs = tf.keras.Input(shape=(num_batches, ))
  reshaped_selector_inputs = tf.keras.layers.Reshape([num_batches, 1])(selector_inputs)
  pretrained_model = two_dim_and_finetune.load_pretrained_model(
    image_size=image_size,
    num_channels=num_channels, 
    include_top=False, 
    weights='imagenet', 
    trainable=False, 
    num_top_trainable_layers=0)
  x = pretrained_model(image_inputs)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Normalization()(x)
  conc_regressors = []
  for i in range(num_batches):
    conc_regressors.append(two_dim_hard_regressor_selector.build_regressor(x))
  print(conc_regressors)
  x_conc = tf.stack(conc_regressors, axis=1)
  x_conc = tf.keras.layers.Multiply()([x_conc, reshaped_selector_inputs])
  x_conc = tf.keras.layers.Reshape([num_batches * 4])(x_conc)
  outputs_conc = tf.keras.layers.Dense(4, activation='linear')(x_conc)
  print(outputs_conc)
  model = tf.keras.Model([image_inputs, selector_inputs], outputs_conc)
  base_learning_rate = 0.0001
  tf.keras.backend.set_epsilon(0.1)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=[tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_error'),
                      tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_percentage_error'),] ,
                loss_weights=[0.5, 0.5],
                metrics=['mae', 'mape']
                )
  print("epsilon: = ", tf.keras.backend.epsilon())
  return model, pretrained_model

def train_CNN_regressor_model(train_dataset, val_dataset, total_epochs, num_batches, model_dir):
  model, pretrained_model = build_CNN_regressor_model(num_batches)
  pretrained_model = two_dim_and_finetune.freeze_layers(pretrained_model, 12)
  model.summary()
  history = model.fit(train_dataset, epochs=total_epochs, initial_epoch=0, validation_data=val_dataset)
  model_path = os.path.join(model_dir, 'regressor_model')
  print(model_path)
  model.save(model_path)