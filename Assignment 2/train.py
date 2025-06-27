"""Train a model on the train dataset (train.joblib).

Requires the train.joblib file to be present in the current directory.

Usage:
    python train.py <training_data> <to_be_trained_model_name>

    model_file: Path to the model file.
"""
# Note:: The code is inspired by data Science class lectures and labs (Professor Jon Barker)

#Import necessary library and packages
from argparse import ArgumentParser
from joblib import load
import os
import numpy as np
import random
from PIL import Image
import joblib
from joblib import Memory
from tempfile import mkdtemp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shutil
from sklearn.neural_network import MLPClassifier

try:
    from train import *
except ImportError:
    print("No custom models supplied. Skipping import")

# Declare the global variables to store the splitted training and testing data
x, x_train, x_test, y_train, y_test, Y = None, None, None, None, None, None

def generate_augmented_images(X):
    # Set the random seed for reproducibility
    np.random.seed(42)  
    # X - A 2D numpy array where each row is an image pairs
    x_train_new = X.copy() #copy the original array
    
    # Perform image augmentation on the training data
    for i, image in enumerate(X):
        # Initialize an empty array to store the augmented image data
        augmented_image = np.empty((5829, ))  # Or set to appropriate size
        
        # Split each image pairs row into two parts
        image_1 = image[:2914].reshape(62, 47)
        image_2 = image[2914:].reshape(62, 47)

        # Convert numpy arrays to Image objects
        image1 = Image.fromarray(image_1)
        image2 = Image.fromarray(image_2)
        
        # Horizontal flip
        if random.choice([True, False]):
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)

        # Vertical
        if random.choice([True, False]):
            image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
            image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)

        # Rotate both images by random angles
        angle = random.choice([0, 90, 180, 270])
        image1 = image1.rotate(angle)
        image2 = image2.rotate(angle)

        # Convert back to numpy array after transformation for further concatenation
        image1 = (np.array(image1)).reshape(1, -1)
        image2 = (np.array(image2)).reshape(1, -1)
        
        # Stack the images into one augmented pair for further processing
        augmented_image = np.hstack((image1, image2))
        # print('Just before augmentation : ',augmented_image.shape)
        # Append the augmented image to the original dataset and return it
        x_train_new = np.vstack((x_train_new, augmented_image))

    return x_train_new

def generate_augmented_labels(Y):
    y_train_new = Y.copy()
    augmented_labels = Y
    y_train_new = np.vstack((y_train_new, augmented_labels))
    return y_train_new

def generate_augmented_data(X, n_times = 2):
    """Augment the train dataset (train.joblib).

    Args:
        X (array): The array representing the training data
        
    Returns:
        The augmented training dataset
        
    """
    global Y # to store the augmented levels, so that it can be accessed anywhere else
    # run over loop
    for i in range(n_times):
        X_augmented = generate_augmented_images(X)
        Y_augmented = generate_augmented_labels(Y)
        X = X_augmented
        Y = Y_augmented

    return X_augmented

def train_model(training_data,trained_model):
    """Train a model on the train dataset (train.joblib).

    Args:
        training_data (str): Path to the training data file.
        trained_model (str): Path to the trained model.
        
    Returns:
        The trained model is saved in the current working directory
        
    """
    print(f"Training started for the dataset: {training_data}")
    global x, x_train, x_test, y_train, y_test, Y
    train_data = load(open("data/train.joblib", "rb"))
    x_train = train_data["data"]
    y_train = train_data["target"]
    Y = y_train.reshape(2200,1)
    # Generated augmented images
    x = generate_augmented_data(x_train)
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.25, random_state=42)
    # call pipeline for nn model with best hyper parameters
    # Create the pipeline and intialise the memory for caching during training
    cachedir = mkdtemp()
    memory = Memory(cachedir, verbose=0) 
    nn_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Normalize the data
        ('PCA',PCA(n_components=400)),
        ('mlp', MLPClassifier(max_iter=1000, early_stopping=True, random_state=0,
                              hidden_layer_sizes = (800, 400, 200, 100, 50, 20),
                              solver = 'adam'))],
                              memory=memory)
    # Modify the pipeline to apply both features and labels augmentation
    nn_pipeline.fit(x_train, y_train)
    y_pred = nn_pipeline.score(x_test, y_test) * 100
    print(y_pred)
    # Save the trained model
    joblib.dump(nn_pipeline, 'model.joblib')
    # Remove the cache memory to clear space
    shutil.rmtree(cachedir)
    print("Temporary directory removed.")
    print(f"Trained model saved as {trained_model}")
    # Save the trained model after checking its size
    trained_model_size = os.path.getsize(trained_model)/(1024 * 1024)
    # a print message indicating the saved model size
    print(f"Trained model has size: {trained_model_size}")

if __name__ == "__main__":
    #the entry point of the script
    parser = ArgumentParser()
    parser.add_argument("training_data", type=str)
    parser.add_argument("trained_model", type=str)
    args = parser.parse_args()
    #call the function to train the model
    train_model(args.training_data,args.trained_model)
