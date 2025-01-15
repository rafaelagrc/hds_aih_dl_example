import pandas as pd
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import albumentations as A
import cv2


# Function to display a few example images
def show_images(image_ids, labels, images_path, n=5):
    """
    Displays a few example images with their labels.

    Args:
        image_ids (list): List of image IDs (filenames without extension).
        labels (list): Corresponding labels (0 for benign, 1 for malignant).
        n (int): Number of images to display. Defaults to 5.
    """
    plt.figure(figsize=(15, 5))  # Create a plot with a specific size
    for i, (image_id, label) in enumerate(zip(image_ids[:n], labels[:n])):
        img_path = os.path.join(images_path, f"{image_id}.jpg")  # Construct full path to image
        img = Image.open(img_path)  # Open the image using PIL
        plt.subplot(1, n, i + 1)  # Create a subplot in a 1xN grid
        plt.imshow(img)  # Display the image
        plt.title(f"Class: {'Malignant' if label == 1 else 'Benign'}")  # Set title based on label
        plt.axis('off')  # Hide axis labels
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot

def plot_class_distribution(metadata):
    """
    This function takes the metadata DataFrame, computes class counts and percentages,
    and plots two bar charts: one for class counts and one for class percentages.

    Args:
        metadata (DataFrame): The DataFrame containing class labels.

    Returns:
        None: This function displays the plots.
    """
    # Compute class counts and percentages
    class_counts = metadata['class'].value_counts()
    class_percentages = class_counts / class_counts.sum() * 100

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot class distribution (Counts)
    bars_counts = axes[0].bar(class_counts.index, class_counts.values, color=['skyblue', 'orange'])
    for bar, count in zip(bars_counts, class_counts):  # Add count labels to each bar
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{count}', ha='center', va='bottom')  # Place count above each bar
    axes[0].set_title('Class Distribution (Counts)')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Benign', 'Malignant'])

    # Plot class distribution (Percentages)
    bars_percentages = axes[1].bar(class_percentages.index, class_percentages.values, color=['skyblue', 'orange'])
    for bar, (percentage, count) in zip(bars_percentages, zip(class_percentages, class_counts)):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{percentage:.1f}%', ha='center', va='bottom')  # Place percentage above each bar
    axes[1].set_title('Class Distribution (Percentages)')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Percentage of Samples')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Benign', 'Malignant'])

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Function to split metadata into train, validation, and test sets and organize image files accordingly
def split_sets(base_dir, train_data, val_data, test_data, images_path):
    """
    Splits the dataset into training, validation, and test sets, and organizes the image files into respective directories.

    Args:
        base_dir: Base directory where the split datasets will be stored.
        train_data: DataFrame containing metadata for training images.
        val_data: DataFrame containing metadata for validation images.
        test_data: DataFrame containing metadata for test images.
    """
    # Create directories for each split and class
    for split in ['train', 'validation', 'test']:
        for cls in ['benign', 'malignant']:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    def copy_images(data, split):
        """
        Copies image files to their respective directories based on the split and class.

        Args:
            data: DataFrame containing image metadata and class labels.
            split: The data split (train, validation, or test).
        """
        for _, row in data.iterrows():
            image_file = row['ISIC_0000000'] + '.jpg'  # Append `.jpg` to the image ID
            class_name = row['benign']  # Class name: 'benign' or 'malignant'
            # Copy the image to its respective directory
            shutil.copy(
                os.path.join(images_path, image_file),
                os.path.join(base_dir, split, class_name, image_file)
            )

    # Copy images into respective directories for each dataset split
    copy_images(train_data, 'train')
    copy_images(val_data, 'validation')
    copy_images(test_data, 'test')

# Function to check and visualize class distribution in train, validation, and test datasets
def check_sets_distribution(train_data, val_data, test_data):
    """
    Analyzes and visualizes the class distribution across training, validation, and test datasets.

    Args:
        train_data: Pandas DataFrame containing training data and class labels.
        val_data: Pandas DataFrame containing validation data and class labels.
        test_data: Pandas DataFrame containing test data and class labels.
    """
    # Count the number of samples in each class for each dataset
    train_class_counts = train_data['class'].value_counts()
    val_class_counts = val_data['class'].value_counts()
    test_class_counts = test_data['class'].value_counts()

    # Create a DataFrame to consolidate class distributions for easier manipulation
    class_distribution = pd.DataFrame({
        'Train': train_class_counts,       # Counts for the training set
        'Validation': val_class_counts,   # Counts for the validation set
        'Test': test_class_counts         # Counts for the test set
    }).T  # Transpose for better alignment (datasets as rows, classes as columns)

    # Calculate class distribution percentages
    class_distribution_percentage = class_distribution.div(class_distribution.sum(axis=1), axis=0) * 100

    # Create subplots for visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns of plots

    # Bar plot of raw counts
    bars_counts = class_distribution.plot(kind='bar', ax=ax[0], color=['skyblue', 'orange'])
    ax[0].set_title('Class Distribution (Counts)')
    ax[0].set_xlabel('Dataset')
    ax[0].set_ylabel('Number of Samples')
    ax[0].legend(['Benign', 'Malignant'])
    ax[0].set_xticks(range(len(class_distribution.index)))
    ax[0].set_xticklabels(class_distribution.index, rotation=0)

    # Add counts to bar chart
    for container in bars_counts.containers:
        bars_counts.bar_label(container, label_type='edge')

    # Bar plot of percentages
    bars_percentage = class_distribution_percentage.plot(kind='bar', ax=ax[1], color=['skyblue', 'orange'])
    ax[1].set_title('Class Distribution (Percentages)')
    ax[1].set_xlabel('Dataset')
    ax[1].set_ylabel('Percentage (%)')
    ax[1].legend(['Benign', 'Malignant'])
    ax[1].set_xticks(range(len(class_distribution_percentage.index)))
    ax[1].set_xticklabels(class_distribution_percentage.index, rotation=0)

    # Add percentage labels to bar chart
    for container in bars_percentage.containers:
        bars_percentage.bar_label(container, fmt='%.1f%%', label_type='edge')

    # Optimize layout
    plt.tight_layout()
    plt.show()

# Function to create data loaders for training, validation, and test sets
def create_dataloaders(train_data_gen, val_test_data_gen, base_dir, img_size, batch_size):
    # Training data generator
    train_gen = train_data_gen.flow_from_directory(
        directory=f'{base_dir}/train',  # Directory containing training images
        target_size=(img_size, img_size),  # Resize images to the specified size
        batch_size=batch_size,  # Number of samples per batch
        class_mode='binary',  # Binary classification: '0' or '1'
        shuffle=True  # Shuffle training data
    )

    # Validation data generator
    val_gen = val_test_data_gen.flow_from_directory(
        directory=f'{base_dir}/validation',  # Directory containing validation images
        target_size=(img_size, img_size),  # Resize images to the specified size
        batch_size=batch_size,  # Number of samples per batch
        class_mode='binary',  # Binary classification: '0' or '1'
        shuffle=False  # Do not shuffle validation data
    )

    # Test data generator
    test_gen = val_test_data_gen.flow_from_directory(
        directory=f'{base_dir}/test',  # Directory containing test images
        target_size=(img_size, img_size),  # Resize images to the specified size
        batch_size=batch_size,  # Number of samples per batch
        class_mode='binary',  # Binary classification: '0' or '1'
        shuffle=False  # Do not shuffle test data
    )

    return train_gen, val_gen, test_gen  # Return the data generators for training, validation, and testing


# Function to initialize and compile a Keras model
def initialize_model(model, optimizer, loss):
    """
    Initializes and compiles the given model with the specified optimizer, loss function, and metrics.

    Args:
        model: The Keras model to initialize.
        optimizer: Optimizer to use for training.
        loss: Loss function to use.

    Returns:
        model: The compiled Keras model.
    """
    # Compile the model with the given optimizer, loss function, and evaluation metric(s)
    model.compile(
        optimizer=optimizer,  # Optimization algorithm, e.g., Adam
        loss=loss,            # Loss function, e.g., Binary Crossentropy
        metrics=['accuracy']  # Metric(s) to monitor during training, e.g., accuracy
    )

    return model  # Return the compiled model


# Function to train a model
def train_model(model, train_gen, val_gen, num_epochs):
    """
    Trains the given model using the training and validation data generators.

    Args:
        model: The Keras model to be trained.
        train_gen: Training data generator.
        val_gen: Validation data generator.
        num_epochs: Number of epochs to train the model.

    Returns:
        model: The trained Keras model.
    """
    # Train the model
    history = model.fit(
        train_gen,                 # Generator for training data
        validation_data=val_gen,   # Generator for validation data
        epochs=num_epochs          # Number of training epochs
    )

    # Plot the training and validation loss over epochs
    plt.plot(history.history['loss'], label='train loss')  # Plot training loss
    plt.plot(history.history['val_loss'], label='val loss')  # Plot validation loss
    plt.xticks(np.arange(num_epochs))  # Set x-axis ticks for each epoch
    plt.legend()  # Add a legend to the plot

    return model  # Return the trained model


# Function to evaluate a model's performance on the test dataset
def evaluate_model(model, test_gen):
    """
    Evaluates the model on the test dataset and prints the loss and accuracy.

    Args:
        model (tf.keras.Model): The trained Keras model to be evaluated.
        test_gen (tf.keras.preprocessing.image.DirectoryIterator): 
            The test data generator providing the test dataset.
    """
    # Evaluate the model on the test data
    eval_result = model.evaluate(test_gen)  # This returns a list with loss and accuracy values

    # Print the evaluation results
    print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")

def get_performance_metrics(model, test_gen):
    """
    Evaluates the model performance on the test set using confusion matrix and classification report.

    Args:
        model (tf.keras.Model): The trained Keras model.
        test_gen (tf.keras.preprocessing.image.DirectoryIterator): Test data generator providing the test set.
    """
    # Generate predictions from the model
    predictions = model.predict(test_gen)  # Predict probabilities for the test set
    predicted_classes = (predictions > 0.5).astype("int32").flatten()  # Convert probabilities to binary classes

    # Get true labels from the test data generator
    true_labels = test_gen.classes  # Get true labels (0 for benign, 1 for malignant)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report with precision, recall, and F1-score for both classes
    print("Classification Report:\n")
    print(classification_report(true_labels, predicted_classes, target_names=['Benign', 'Malignant']))

def visualize_predictions(test_gen, model, num_images=5):
    """
    Visualizes model predictions on a batch of test images.

    Args:
        test_gen (tf.keras.preprocessing.image.DirectoryIterator): The test data generator.
        model (tf.keras.Model): The trained model for making predictions.
        num_images (int): The number of images to visualize. Default is 5.
    """
    # Get a batch of images and their true labels from the test set
    images, labels = next(test_gen)  # Get a batch of images and corresponding true labels

    # Get predictions from the model on the batch
    predictions = model.predict(images)  # Model predicts on the batch of images

    # Visualize the first `num_images` images with true and predicted labels
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)  # Subplot for each image

        # Display the image
        plt.imshow(images[i])  # Plot the image
        plt.axis('off')  # Turn off axis to focus on the image

        # Convert labels and predictions to text labels
        true_label = 'Benign' if labels[i] == 0 else 'Malignant'  # Map label 0 to 'Benign' and 1 to 'Malignant'
        pred_label = 'Benign' if predictions[i] < 0.5 else 'Malignant'  # Threshold the prediction to 'Benign' or 'Malignant'

        # Set the title with true and predicted labels
        plt.title(f"True: {true_label}\nPred: {pred_label}")

    # Display all the images
    plt.show()


# Function to oversample the minority class in the training dataset
def oversample_minority_class_in_training_data(train_data, images_path):
    # Calculate the number of augmentations needed for each minority image
    # To balance the dataset (number of benign = number of malignant)
    # n_augs = int(np.ceil(len(train_data[train_data['class'] == 0]) / 
    #                      len(train_data[train_data['class'] == 1])))

    n_augs = 2

    # Extract image paths, benign/malignant labels, and class labels from the training data
    X_train = np.array([file for file in train_data['ISIC_0000000']])  # List of image paths
    benign_train = np.array([file for file in train_data['benign']])  # Benign/malignant labels
    y_train = np.array(train_data['class'])  # Class labels (0 or 1)

    # Lists to store augmented image data
    aug_images_ids = []  # Augmented image IDs
    aug_images_y = []    # Corresponding class labels for augmented images
    aug_benign = []      # Corresponding benign/malignant labels for augmented images
    
    # Track the current number of malignant and benign samples
    n_malignant = len(train_data[train_data['class'] == 1])
    n_benign = len(train_data[train_data['class'] == 0])

    # Loop through each training sample
    for img_id, y, benign in zip(X_train, y_train, benign_train):
        if y == 1:  # Only augment malignant cases
            img_name = img_id + '.jpg'  # Get the image filename
            
            # Load the image using OpenCV
            img = cv2.imread(os.path.join(images_path, img_name))  # Load the image
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format

            # Define the augmentation pipeline (applying multiple augmentations at once)
            augmentations = A.Compose([
                A.RandomRotate90(p=0.3),               # Random 90-degree rotation
                A.Flip(p=0.4),                         # Random horizontal/vertical flip
                A.RandomBrightnessContrast(p=0.3),     # Adjust brightness and contrast
                A.HueSaturationValue(p=0.2),           # Modify hue, saturation, and value
                A.CLAHE(p=0.2),                        # Apply adaptive histogram equalization
                A.Rotate(limit=30, p=0.2),             # Rotate within a limit of 30 degrees
                A.ElasticTransform(alpha=0.5, p=0.2),  # Elastic deformation (lower intensity)
                A.GridDistortion(p=0.2),               # Apply grid distortion
                A.GaussianBlur(p=0.2)                  # Blur to simulate low-quality images
            ])


            # Apply augmentations until the number of malignant images matches the benign images
            for i in range(n_augs):
                if n_malignant == n_benign:  # Stop when the dataset is balanced
                    break
                
                # Apply the augmentation pipeline to the image
                augmented = augmentations(image=img)
                aug_image = augmented['image']
                
                # Save the augmented image to the directory
                cv2.imwrite(os.path.join(images_path, img_id + f'_aug{i}.jpg'), aug_image)
                
                # Append augmented data to the lists
                aug_images_ids.append(img_id + f'_aug{i}')  # Augmented image ID
                aug_images_y.append(y)  # Class label
                aug_benign.append(benign)  # Benign/malignant label
                
                # Update the malignant image count
                n_malignant += 1

    # Create a new DataFrame containing both the original and augmented training data
    new_train_data = pd.DataFrame({
        'ISIC_0000000': list(X_train) + aug_images_ids,  # Image paths (original + augmented)
        'benign': list(benign_train) + aug_benign,  # Benign/malignant labels
        'class': list(y_train) + aug_images_y  # Class labels (original + augmented)
    })

    return new_train_data