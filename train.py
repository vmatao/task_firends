import model
import os
import keras
import numpy as np
import util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train():
    epochs = 15
    batch_size = 32
    train_samples = 10 * 5000  # 10 categories with 5000 images in each category
    validation_samples = 10 * 1000  # 10 categories with 1000 images in each category
    img_width, img_height = 32, 32
    # Get the model (10 categories)
    model_img = model.resnet50_model(10)
    # Create a data generator for training
    train_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Create a data generator for validation
    validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    t_data, t_labels = util.load_train_data()
    v_data, v_labels = util.load_test_data()
    t_data, v_data = t_data.reshape(50000, 32, 32, 3), v_data.reshape(10000, 32, 32, 3)

    temp_t_labels = np.zeros((t_labels.shape[0], 10))
    temp_v_labels = np.zeros((v_labels.shape[0], 10))
    temp_t_labels[np.arange(temp_t_labels.shape[0]), t_labels] = 1
    temp_v_labels[np.arange(temp_v_labels.shape[0]), v_labels] = 1

    # Create a train generator
    train_generator = train_data_generator.flow(t_data, temp_t_labels, batch_size=32)
    # Create a test generator
    validation_generator = validation_data_generator.flow(v_data, temp_v_labels, batch_size=32)

    x, y = train_generator.next()
    # Start training, fit the model
    model_img.fit(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        epochs=epochs,
        shuffle=True)
    # Save model to disk
    model_img.save('C:\\Users\\vladm\\Desktop\\task_firends\\models\\resnet_50.h5')
    print('Saved model to disk!')


# The main entry point for this module
def main():
    # Train a model
    train()


# Tell python to run main method
if __name__ == '__main__': main()
