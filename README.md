# Traffic Sign Classifier 🚦

This project uses a deep learning model to classify traffic signs from images. It's built with Python and the TensorFlow/Keras library, employing a Convolutional Neural Network (CNN) to achieve high accuracy in identifying various traffic signs.

-----

## Features

  - **High Accuracy**: Achieves high classification accuracy on the test dataset.
  - **Multiple Classes**: Capable of classifying numerous distinct traffic sign categories.
  - **Data Visualization**: Includes scripts for visualizing the dataset and training progress.
  - **Scalable Architecture**: The CNN model can be extended or modified for other image classification tasks.

-----

## Technologies Used

  - **Python**
  - **TensorFlow** & **Keras** for building and training the deep learning model
  - **OpenCV** for image processing
  - **NumPy** for numerical operations
  - **Matplotlib** & **Seaborn** for data visualization
  - **Scikit-learn** for performance metrics

-----

## Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dharmareddy06/Traffic-sign-classifier.git
    cd traffic-sign-classifier
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----


## Model Architecture

The core of this classifier is a **Convolutional Neural Network (CNN)**. The typical architecture includes:

1.  **Convolutional Layers**: Two or more `Conv2D` layers with ReLU activation functions to extract features like edges, corners, and textures from the images.
2.  **Pooling Layers**: `MaxPooling2D` layers to reduce the spatial dimensions (down-sampling) of the feature maps, making the model more robust to variations in object position.
3.  **Flatten Layer**: To convert the 2D feature maps into a 1D vector.
4.  **Dense Layers**: Fully connected layers for classification, with a final layer using a `softmax` activation function to output a probability distribution over the 43 classes.
5.  **Dropout Layers**: Included to prevent overfitting by randomly setting a fraction of input units to 0 during training.

-----

## Usage

You can train the model or use a pre-trained model to classify new images.

### Training the Model

To train the model from scratch, run the training script:

```bash
python train.py
```

The script will load the dataset, build the CNN model, train it, and save the trained model weights to a file (e.g., `traffic_model.h5`).

### Classifying an Image

To test the classifier with a new image, use the prediction script:

```bash
python predict.py --image path/to/your/image.png
```

The script will load the pre-trained model, process the input image, and print the predicted traffic sign class.

-----

## Evaluation

The model's performance is evaluated based on accuracy and loss, which are plotted during training.

  - **Training & Validation Accuracy**: Typically reaches over 95% on the validation set.
  - **Training & Validation Loss**: Decreases steadily, indicating that the model is learning effectively.

A **confusion matrix** and a **classification report** are also generated to provide a detailed look at the model's performance across all classes, highlighting any potential misclassifications.

-----
