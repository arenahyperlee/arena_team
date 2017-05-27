from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, 
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, EarlyStopping

img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 5000
nb_validation_samples = 100 
batch_size = 32
epochs = 50

model = applications.VGG19(weights = "imagenet", include_top=False, 
						   input_shape = (img_width, img_height, 3))

for layer in model.layers[:10]:
    layer.trainable = False

x = model.output
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)

model_final.compile(loss = "categorical_crossentropy", 
					optimizer = optimizers.SGD(lr=0.0001), 
					metrics=["accuracy"])

train_datagen = ImageDataGenerator(
									rescale = 1./255,
									horizontal_flip = True,
									fill_mode = "nearest",
									zoom_range = 0.3,
									width_shift_range = 0.3,
									height_shift_range=0.3,
									rotation_range=30)

train_generator = datagen.flow_from_directory(
									train_data_dir,
									target_size = (img_height, img_width),
									batch_size = batch_size, 
									class_mode = "categorical")

validation_generator = datagen.flow_from_directory(
									validation_data_dir,
									target_size = (img_height, img_width),
									class_mode = "categorical")

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', 
							 verbose=1, save_best_only=True, 
							 save_weights_only=False, mode='auto', 
							 period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, 
					  verbose=1, mode='auto')

model_final.fit_generator(
							train_generator,
							samples_per_epoch = nb_train_samples,
							epochs = epochs,
							validation_data = validation_generator,
							nb_val_samples = nb_validation_samples,
							callbacks = [checkpoint, early])




























