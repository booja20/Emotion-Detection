import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image as PILImage  # Ensure Pillow is imported
import os

def convert_jfif_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jfif'):
            filepath = os.path.join(directory, filename)
            with PILImage.open(filepath) as img:
                rgb_img = img.convert('RGB')
                new_filepath = filepath.rsplit('.', 1)[0] + '.jpg'
                rgb_img.save(new_filepath, 'JPEG')
                os.remove(filepath)  # Optionally remove the original .jfif file

# Specify your directories
directories = ['training_data/happy', 'training_data/sad', 'training_data/neutral']
for directory in directories:
    convert_jfif_to_jpg(directory)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: happy, sad, neutral
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% of data will be used for validation
)

train_generator = train_datagen.flow_from_directory(
    'training_data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Set subset to 'training'
)

validation_generator = train_datagen.flow_from_directory(
    'training_data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Set subset to 'validation'
)
print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)


# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save('mood_detection_model.h5')
