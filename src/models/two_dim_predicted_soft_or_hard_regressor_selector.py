import os
import tensorflow as tf
import tensorflow_addons as tfa

from src.models import two_dim_and_finetune

def build_regressor(x):
  x_conc = tf.keras.layers.Dropout(0.2)(x)
  x_conc = tf.keras.layers.Dense(16)(x_conc)
  x_conc = tf.keras.layers.Dropout(0.2)(x_conc)
  x_conc = tf.keras.layers.Dense(4)(x_conc)
  return x_conc

def build_batch_id_classifier(x):
  x_batch = tf.keras.layers.Dropout(0.2, name="batch_classifier_dropout_1")(x)
  x_batch = tf.keras.layers.Dense(16)(x_batch)
  x_batch = tf.keras.layers.Dropout(0.2)(x_batch)
  outputs_batch = tf.keras.layers.Dense(10, activation='softmax', name='outputs_batch')(x_batch)
  return outputs_batch

def build_CNN_2D_predicted_soft_or_hard_regressor_selector_model(use_hard_selector=True):
  image_size = (160, 160)
  num_channels = 3
  image_inputs = tf.keras.Input(shape=image_size + (num_channels,))
  pretrained_model = two_dim_and_finetune.load_pretrained_model(image_size=image_size,
                                           num_channels=num_channels, 
                                           include_top=False, 
                                           weights='imagenet', 
                                           trainable=False, 
                                           num_top_trainable_layers=0)
  x = pretrained_model(image_inputs)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Normalization()(x)
  outputs_batch = build_batch_id_classifier(x)
  if use_hard_selector:
    x_batch = tfa.seq2seq.hardmax(outputs_batch)
    one_hot_predicted_batch = tf.keras.layers.Reshape([10, 1])(x_batch)
  else:
    one_hot_predicted_batch = tf.keras.layers.Reshape([10, 1])(outputs_batch)

  conc_regressors = []
  for i in range(10):
    conc_regressors.append(build_regressor(x))
  x_conc = tf.stack(conc_regressors, axis=1)
  x_conc = tf.keras.layers.Multiply()([x_conc, one_hot_predicted_batch])
  x_conc = tf.keras.layers.Reshape([40])(x_conc)
  outputs_conc = tf.keras.layers.Dense(4, activation='linear', name='outputs_conc')(x_conc)
  tf.keras.backend.set_epsilon(0.1)
  model = tf.keras.Model(image_inputs, [outputs_batch, outputs_conc])
  print("epsilon: = ", tf.keras.backend.epsilon())
  return model, pretrained_model


def train_CNN_2D_predicted_soft_or_hard_regressor_selector_model(train_dataset, val_dataset, pretrained_epochs, total_epochs, model_dir, use_hard_selector = True):
  model, pretrained_model = build_CNN_2D_predicted_soft_or_hard_regressor_selector_model(use_hard_selector)
  pretrained_model = two_dim_and_finetune.freeze_layers(pretrained_model, 12)
  model.summary()
  model.compile(loss={'outputs_batch': 'categorical_crossentropy',
                      'outputs_conc': [tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_error'),
                      tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_percentage_error'),]},
                loss_weights=[0.8, [0.05, 0.15]],
                metrics=['accuracy', ['mae', 'mape']],
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=10e-5))
  model.fit(train_dataset, epochs=total_epochs, initial_epoch=pretrained_epochs, validation_data=val_dataset)
  model_path = os.path.join(model_dir, 'model')
  model.save(model_path)
