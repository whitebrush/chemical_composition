import os
import tensorflow as tf

# Create the base model from the pre-trained model MobileNet V2
def load_pretrained_model(image_size, num_channels, include_top, weights, trainable, num_top_trainable_layers):
  pretrained_model = tf.keras.applications.MobileNetV2(
      input_shape=image_size + (num_channels,), 
      include_top=include_top, 
      weights=weights)
  pretrained_model.trainable = trainable
  if trainable:
    pretrained_model = freeze_layers(pretrained_model, num_top_trainable_layers)
  return pretrained_model

def freeze_layers(model, num_top_trainable_layers):
  model.trainable = True
  for layer in model.layers[:(len(model.layers) - num_top_trainable_layers)]:
    layer.trainable = False
  return model

def build_CNN_2D_and_finetune_model():
  image_size = (160, 160)
  num_channels = 3
  inputs = tf.keras.Input(shape=image_size + (num_channels,))
  pretrained_model = load_pretrained_model(image_size=image_size,
                                           num_channels=num_channels, 
                                           include_top=False, 
                                           weights='imagenet', 
                                           trainable=False, 
                                           num_top_trainable_layers=0) 
  x = pretrained_model(inputs)
  x = tf.keras.layers.Activation('linear', name='activation')(x)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Normalization()(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(16)(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = tf.keras.layers.Dense(4)(x)
  model = tf.keras.Model(inputs, outputs)
  base_learning_rate = 0.0001
  tf.keras.backend.set_epsilon(0.1)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=[tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_error'),
                      tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_percentage_error'),] ,
                loss_weights=[0.5, 0.5],
                metrics=['mape', 'mae'],
                )
  return model, pretrained_model

def train_CNN_2D_and_finetune_model(train_dataset, val_dataset, pretrained_epochs, total_epochs, model_dir, fine_tune=False):
  if not fine_tune:
    model, pretrained_model = build_CNN_2D_and_finetune_model()
    # model.summary()
    # pretrained_model.summary()
    model.fit(train_dataset, epochs=pretrained_epochs, validation_data=val_dataset)

    model_path = os.path.join(model_dir, 'pre_trained_repr')
    model.save(model_path)

    pretrained_model = freeze_layers(pretrained_model, 12)
    tf.keras.backend.set_epsilon(0.1)
    model.compile(loss=[tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_error'),
                        tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.SUM, name='mean_absolute_percentage_error'),] ,
                  loss_weights=[0.5, 0.5],
                  metrics=['mape', 'mae'],
                  optimizer = tf.keras.optimizers.RMSprop(learning_rate=10e-5))
    # model.summary()
    # pretrained_model.summary()

    model.fit(train_dataset, epochs=total_epochs, initial_epoch=100, validation_data=val_dataset)

    model_path = os.path.join(model_dir, 'model')
    model.save(model_path)
  else:
    model = tf.keras.models.load_model(os.path.join(model_dir, 'model'))
    model.fit(train_dataset, epochs=total_epochs, validation_data=val_dataset)
    model_path = os.path.join(model_dir, 'fine_tune_unseen_model')
    model.save(model_path)
