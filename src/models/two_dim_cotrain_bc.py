import os
import tensorflow as tf

from src.models import two_dim_and_finetune

def build_CNN_2D_and_cotrain_bc_model():
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
  x_conc = tf.keras.layers.Dropout(0.2)(x)
  x_conc = tf.keras.layers.Dense(16)(x_conc)
  x_conc = tf.keras.layers.Dropout(0.2)(x_conc)
  outputs_conc = tf.keras.layers.Dense(4, activation='linear')(x_conc)
  x_batch_id = tf.keras.layers.Dropout(0.2)(x)
  x_batch_id = tf.keras.layers.Dense(16)(x_batch_id)
  x_batch_id = tf.keras.layers.Dropout(0.2)(x_batch_id)
  outputs_batch_id = tf.keras.layers.Dense(16, activation='softmax')(x_batch_id)
  model = tf.keras.Model(inputs, [outputs_conc, outputs_batch_id])
  base_learning_rate = 0.0001
  tf.keras.backend.set_epsilon(0.1)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=[tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_error'),
                      tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_percentage_error'),] ,
                loss_weights=[0.5, 0.5],
                metrics=['mape', 'mae'],
                )
  return model, pretrained_model

def train_CNN_2D_and_cotrain_bc_model(train_dataset, val_dataset, pretrained_epochs, total_epochs, model_dir):
  model, pretrained_model = build_CNN_2D_and_cotrain_bc_model()
  model.summary()
  history = model.fit(train_dataset, epochs=pretrained_epochs, validation_data=val_dataset)

  model_path = os.path.join(model_dir, 'pre_trained_repr')
  model.save(model_path)

  pretrained_model = two_dim_and_finetune.freeze_layers(pretrained_model, 12)
  tf.keras.backend.set_epsilon(0.1)
  model.compile(loss=[tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_error'),
                      tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_percentage_error'),] ,
                loss_weights=[0.5, 0.5],
                metrics=['mape', 'mae'],
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=10e-5))

  history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=100, validation_data=val_dataset)

  model_path = os.path.join(model_dir, 'model')
  model.save(model_path)