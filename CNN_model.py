# Importing necessary libraries
from keras.models import Sequential
from keras.layers import Convolution2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import matplotlib.pyplot as plt

# Function to build the model with hyperparameters
def build_model(hp):
    classifier = Sequential()

    classifier.add(Convolution2D(hp.Int('conv1_units', min_value=32, max_value=256, step=32), 3, 3, input_shape=(224, 224, 3), activation='relu'))
    classifier.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    classifier.add(AveragePooling2D(pool_size=(4, 4)))

    classifier.add(Convolution2D(hp.Int('conv2_units', min_value=32, max_value=256, step=32), 3, 3, activation='relu'))
    classifier.add(AveragePooling2D(pool_size=(4, 4)))

    classifier.add(Flatten())

    classifier.add(Dense(hp.Int('dense1_units', min_value=32, max_value=256, step=32), activation='relu'))
    classifier.add(Dense(hp.Int('dense2_units', min_value=32, max_value=256, step=32), activation='relu'))
    classifier.add(Dense(hp.Int('dense3_units', min_value=32, max_value=256, step=32), activation='relu'))
    classifier.add(Dense(hp.Int('dense4_units', min_value=32, max_value=256, step=32), activation='relu'))
    classifier.add(Dense(hp.Int('dense5_units', min_value=32, max_value=256, step=32), activation='relu'))
    classifier.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.6, step=0.1)))
    
    # Modify the last layer for multi-class classification
    classifier.add(Dense(3, activation='softmax'))

    # Adjust the learning rate based on the hyperparameter search space
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log'))
    
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier

# Data preprocessing with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # Ajout de la rotation aléatoire
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Update class_mode to 'categorical' for multi-class classification
training_set = train_datagen.flow_from_directory(
    "M:/Master IA/Semestre 2/Projet_2/dataset/train",
    target_size=(224, 224),
    batch_size=100,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    "M:/Master IA/Semestre 2/Projet_2/dataset/test",
    target_size=(224, 224),
    batch_size=100,
    class_mode='categorical'
)

# Instantiate the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials= 20,
    executions_per_trial=1,
    directory='my_dir',
    project_name='Projet_Synthese_DIT'
)

# Search for the best hyperparameters over 15 epochs
tuner.search(training_set, epochs = 15 , validation_data=test_set)

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the final model using the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)

# Train the final model
history = final_model.fit(training_set, epochs = 150, validation_data=test_set)

# Save the final model
final_model.save("model/Drowness_tuned.h5")

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
