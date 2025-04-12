import os
import zipfile
if os.path.exists('imagedb_btsd.zip'):
  if os.path.isdir('imagedb') and os.path.isdir('imagedb_test') :
    print('Files already exist...')
  else:
    local_zip = 'imagedb_btsd.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('./')
    zip_ref.close()
    print("Unzip completed...")
else:
  print('imagedb_btsd.zip not found')
train_dir = 'imagedb'
test_dir = 'imagedb_test'
models_path = "models"

if not os.path.isdir('models'):
  os.makedirs(models_path)
#Loading all the models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, InceptionV3, MobileNetV2, NASNetMobile, ResNet50V2, VGG16, Xception
from keras.applications.xception import preprocess_input  
from keras import models, layers, optimizers
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
# if tf.config.list_physical_devices('GPU'):
#     print("GPU is available")
# else:
#     print("GPU is not available")
acc = []
loss = []
list_of_models = []
input_size = (128,128,3)
tf.keras.utils.set_random_seed(123)
# densenet = DenseNet121(weights='imagenet',
#                  include_top=False,
# #                  input_shape=(128, 128, 3))
# inception= InceptionV3(weights='imagenet',
#                  include_top=False,
#                  input_shape=(128, 128, 3))
# MobileNet = MobileNetV2(weights='imagenet',
#                  include_top=False,
#                  input_shape=(128, 128, 3))
# nasnet = NASNetMobile(weights='imagenet',
#                  include_top=False,
#                  input_shape=(128, 128, 3))
# resnet = ResNet50V2(weights='imagenet',
#                  include_top=False,
#                  input_shape=(128, 128, 3))
# vgg_conv = VGG16(weights='imagenet',
#                  include_top=False,
#                  input_shape=(128, 128, 3))
xception = Xception(weights='imagenet',
                 include_top=False,
                 input_shape=input_size)
# list_of_models = [ densenet, inception, MobileNet, nasnet, resnet, vgg_conv, xception]
list_of_models = [ xception]
for model in list_of_models:
  model.trainable = False
  ff = open(f'{model.name}_layer.txt', 'w')
  for layer in model.layers:
      ff.write(f'{layer}, {layer.trainable}\n')
  ff.close()

# Create the model
for base_model in list_of_models:
  model = models.Sequential(name=base_model.name)
  # Add the base model
  model.add(base_model)
  # Add new layers
  model.add(layers.Flatten())
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dense(34, activation='softmax'))
  
  # Compile the model
  print(f'Runing the {model.name} model...')
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=0.001),
                metrics=['acc'])
  # Show a summary of the model. Check the number of trainable parameters
  model.summary()
  # Add our data-augmentation parameters to ImageDataGenerator
  train_datagen = ImageDataGenerator(rescale=1./255,
                                    preprocessing_function=preprocess_input,
                                    validation_split=0.2)
  test_datagen  = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

  # --------------------
  # Flow training images in batches of 20 using train_datagen generator
  # --------------------
  train_generator = train_datagen.flow_from_directory(train_dir,
                                                      batch_size=64,
                                                      class_mode='categorical',
                                                      target_size=input_size[:2],
                                                      subset='training')
  # --------------------
  # Flow validation images in batches of 20 using test_datagen generator
  # --------------------
  validation_generator =  train_datagen.flow_from_directory(train_dir,
                                                          batch_size=64,
                                                          class_mode='categorical',
                                                          target_size=input_size[:2],
                                                          subset='validation')

  callbacks = []

  save_best_callback = tf.keras.callbacks.ModelCheckpoint(f'best_weights.keras', save_best_only=True, verbose=1)
  callbacks.append(save_best_callback)

  early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
  callbacks.append(early_stop_callback)

  # Train the model
  history = model.fit(
        train_generator,
        steps_per_epoch=int(train_generator.samples/train_generator.batch_size) ,
        epochs=30,
        validation_data=validation_generator,
        verbose=1,
        callbacks=callbacks)

  # Save the model
  model.save(f'./{models_path}/{base_model.name}.keras')
  print(f"\n======MODEL {base_model.name}======\n\n")

  # --------------------
  # Flow validation images in batches of 20 using test_datagen generator
  # --------------------
  test_generator =  test_datagen.flow_from_directory(test_dir,
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    target_size=input_size[:2])
  l, a = model.evaluate(test_generator)
  loss.append(l)
  acc.append(a)
  ff = open("Results.txt", "a")
  ff.write(f"{model.name}, ")
  ff.write(f"{round(a*100,5)}, ")
  ff.write(f"{round(l,5)}\n")
  ff.close()

  print(f"The current model is the {model.name}\n")
  print(f"Accuracy of the model is: {round(a*100,5)}%")
  print(f"\nLoss of the model is: {round(l,5)}\n\n")

  # Just an example...
  path = './imagedb_test/00038/00027_00005.ppm'
  img = load_img(path, target_size=input_size[:2], interpolation='bilinear')
  plt.imshow(img, cmap='gray')
  plt.show()

  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)

  classes_pred = model.predict(x)
  dir = os.listdir(test_dir)
  dir.sort()
  classes = [s for s in dir]
  print(f'\n\n\n{classes}')
  print(f'\n\n\nSoftmax Output: {classes_pred}')
  print(f'\n\n\n{path} is a {classes[classes_pred.argmax()]}\n\n\n')

  # Plotting the results of every model
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history.history['acc'])
  plt.title('CNN Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Test'], loc='upper right')

  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'])
  plt.title('CNN Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(['Train', 'Test'], loc='upper right')

  plt.tight_layout()
  plt.show()

