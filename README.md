# Brain Tumor Classification (CNN)

Classify brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, No Tumor** using a **Convolutional Neural Network (CNN)** in TensorFlow/Keras.  
!!! Currently reimplementing the project on **AWS SageMaker** for cloud-based training and deployment !!!

Dataset: [Brain MRI Images Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download)

## Steps

- **Data Preprocessing**
  - Loaded MRI images from `Training` and `Testing` folders.
  - Resized all images to **256×256** pixels.
  - Normalized pixel values to `[0, 1]` using a `Rescaling` layer.
  - Applied **data augmentation**:
    - Random horizontal/vertical flips
    - Random rotations
    - Random zoom
  - Removed duplicate images using **perceptual hashing (dHash)** following (https://pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/):
    - Removed duplicates within each dataset
    - Removed duplicates between training and test sets

- **Data Splitting**
  - Split training set into:
    - **80% Training Data**
    - **20% Validation Data**
  - Used TensorFlow's `image_dataset_from_directory` for loading and batching.
  - Cached and prefetched data for faster training.

- **Modeling**
  - Defined a custom CNN architecture:
    - 6 convolutional layers (filters: 32, 64, 128) with ReLU activations
    - MaxPooling layers after each convolution block
    - Dropout layer for regularization
    - Dense layer with Softmax activation for multi-class classification
  - Compiled model with:
    - Optimizer: **Adam**
    - Loss: **SparseCategoricalCrossentropy**
    - Metric: **Accuracy**
  - Trained for **30 epochs** on local machine.

- **Evaluation**
  - Plotted **training & validation accuracy** and **loss curves**.
  - Predicted classes on test set with **confidence scores**.
  - Achieved **~95% test accuracy**.

## Currently Learning to train and deploy on AWS
  - Creating an **AWS SageMaker notebook** (`cnn_brain_tumor_AWS.ipynb`).
  - Setting up:
    - S3 bucket for data storage
    - SageMaker training jobs for model training
    - Deployment as an endpoint 



