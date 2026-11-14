import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuration
DATASET_DIR = r'dataset/dataset_blood_group'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def evaluate():
    if not os.path.exists('model_custom_cnn.h5'):
        print("Error: Model file 'model_custom_cnn.h5' not found.")
        return

    print("Loading model...")
    model = load_model('model_custom_cnn.h5')

    print("Preparing validation data...")
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("Evaluating model...")
    loss, accuracy = model.evaluate(validation_generator)
    print(f"\nFinal Validation Accuracy: {accuracy*100:.2f}%")
    print(f"Final Validation Loss: {loss:.4f}")

if __name__ == '__main__':
    evaluate()
