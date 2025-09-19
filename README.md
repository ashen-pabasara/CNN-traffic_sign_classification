# CNN-traffic_sign_classification

## Overview
This project implements a Convolutional Neural Network (CNN) to classify traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Traffic sign recognition is a crucial component of driver assistance systems and autonomous vehicles. The work emphasizes preprocessing, CNN design, and evaluation while highlighting limitations such as class imbalance and computational cost.

The model achieves ~84% accuracy on the test set, demonstrating that CNNs are highly effective for traffic sign classification tasks.

## Dataset

Source: [GTSRB dataset.](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

* **Classes:** 43 traffic sign categories (speed limits, warnings, prohibitions, etc.)
* **Size:** 
    * 51,839 images
    * 39,209 for training/validation
    * 12,630 for testing
* **Properties:** Images vary in size, angle, lighting, and background clutter.

## Methodology

1. **Data Preprocessing**
    * Resize all images to 48×48 pixels
    * Normalize pixel values
    * Split into training, validation, and test sets

2. **CNN Architecture (TensorFlow/Keras)**
    * Convolutional + ReLU activation layers
    * MaxPooling layers for dimensionality reduction
    * Dropout for regularization
    * Dense layer with softmax activation for final classification

4. **Training**

    * Loss: Sparse Categorical Cross-Entropy
    * Optimizer: Adam
    * Epochs: up to 25 with early stopping and learning rate scheduling

5. Evaluation

    * Accuracy: ~99% training, ~84% test
    * Metrics: classification report, confusion matrix, training/validation curves

## Results

* The model performs well for common sign classes but struggles with rare or visually similar categories.
* Dense layer dominates parameter count (~8.38M out of 8.83M total), making the model heavy for embedded deployment.
* Regularization reduced overfitting, but class imbalance remains a challenge.

## Limitations

* Assumes cropped images of traffic signs (no detection/localization).
* Dataset limited to German signs.
* Large parameter count affects deployment on resource-constrained devices.

## Repository Structure
```
.
├── traffic_sign_classification.ipynb   # Colab notebook with code
├── Mini Research Report.pdf            # Full research report
└── README.md                           # Project documentation

```
             
## Requirements
* Python 3.x
* Tensorflow/Keras
* Numpy, Pandas, Matplotlib, Scikit-learn

Install dependencies with:
```
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone the repository:
    ```
    git clone https://github.com/ashen-pabasara/CNN-traffic_sign_classification.git
    cd <repo>
    ```

2. Open the notebook in Google Colab or Jupyter:
    ```
    jupyter notebook traffic_sign_classification.ipynb
    ```

3. Run all cells to train and evaluate the CNN.

## Citation  

If you use this work, please cite the original **GTSRB dataset paper**:  

> Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. Neural Networks, 32, 323–332.  
