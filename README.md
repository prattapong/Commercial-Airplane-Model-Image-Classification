# Commercial Airplane Model Image Classification

## Overview
This project is a deep learning-based image classification model that identifies different commercial airplane models using a Convolutional Neural Network (CNN) based on ResNet50. The model is trained on images of three airplane types: **Airbus A320, Airbus A350, and Boeing 787**.

## Dataset
The dataset is organized into three folders within the `images` directory:

```
images/
│── Airbus A320/   # Contains images of Airbus A320 aircraft
│── Airbus A350/   # Contains images of Airbus A350 aircraft
│── Boeing 787/    # Contains images of Boeing 787 aircraft
```

## Project Structure
```
├── images/
│   ├── Airbus A320/
│   ├── Airbus A350/
│   ├── Boeing 787/
├── model_training.py
├── model_prediction.py
├── best_airplane_model.h5
├── requirements.txt
├── README.md
```

## Model Training
The model is built using **ResNet50** as a feature extractor, with additional fully connected layers for classification. The training pipeline includes:

- **Data Augmentation:** To enhance generalization
- **Transfer Learning:** Using a pre-trained ResNet50 model
- **Checkpointing:** Saving the best model based on validation accuracy

### Training Script
The `image_classification.ipynb` script fetches images, preprocesses them, and trains the model. The best-performing model is saved as `best_airplane_model.h5`.

```python
best_model = train_model(
    X = X, y = y, batch_size = 8, epochs = 100
)
```

## Model Prediction
Once trained, the model can classify new images. The `image_classification.ipynb` script takes an image as input and outputs the predicted airplane type.

### Example Usage:
```python
show_prediction(best_model, X, y, index=0, class_labels=["Airbus A320", "Airbus A350", "Boeing 787"])
```
This function displays the image along with the predicted and actual class labels.

## Installation & Requirements
Ensure you have the necessary dependencies installed before running the scripts.

### Install Requirements:
```sh
pip install -r requirements.txt
```

### Dependencies:
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

## How to Run
### Clone this repository:
```sh
git clone https://github.com/prattapong/Commercial-Airplane-Model-Image-Classification.git
cd Commercial-Airplane-Model-Image-Classification
```

### Install dependencies:
```sh
pip install -r requirements.txt
```

## Future Enhancements
- Expand dataset with more airplane models
- Improve model accuracy with hyperparameter tuning
- Deploy as a web application for real-time classification
