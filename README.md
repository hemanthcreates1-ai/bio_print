BioPrint: CNN-Based Fingerprint Image Classification for Eight Blood-Group Labels

📌 Project Overview

BioPrint is a deep learning-based experimental project that uses Convolutional Neural Networks (CNNs) to classify fingerprint images into eight blood-group labels: A+, A-, B+, B-, AB+, AB-, O+, and O-.

The project evaluates two CNN architectures: a baseline CNN and an improved custom CNN. It explores fingerprint image preprocessing, model training, classification, Test-Time Augmentation (TTA), and performance evaluation.

A Streamlit-based web application is also developed to allow users to upload fingerprint images and view the model's predicted class, classification score, and class-wise results.

🎯 Objectives

- Develop a CNN-based fingerprint image classification system.
- Compare the performance of baseline and improved CNN models.
- Apply image preprocessing techniques for consistent model input.
- Evaluate the impact of Test-Time Augmentation (TTA).
- Build an interactive web interface for image upload and prediction.
- Analyse model performance using standard classification metrics.

⚙️ Key Features

- Eight-Class Classification: Classifies fingerprint images into eight supplied blood-group labels.
- Image Preprocessing: Resizes images to 128 × 128 pixels and converts them to grayscale.
- Custom CNN Architecture: Implements and evaluates an improved CNN alongside a baseline model.
- Test-Time Augmentation: Averages predictions from five image views: original, horizontal flip, and rotations of +10°, -10°, and +5°.
- Performance Evaluation: Uses accuracy, balanced accuracy, precision, recall, F1-score, macro-F1, and confusion matrix.
- Duplicate Image Audit: Uses SHA-256 hashing to identify exact duplicate images and examine their effect on validation results.
- Interactive Web Application: Provides image upload, preview, prediction scores, and visual results through Streamlit.

🧠 Model Architecture and Workflow

1. Dataset Collection: Loads fingerprint images from the Kaggle Fingerprint Blood Group Classification Dataset.
2. Image Preprocessing: Resizes images to 128 × 128 and converts them into grayscale.
3. CNN Model Development: Builds and trains baseline and improved CNN architectures using TensorFlow and Keras.
4. Model Training: Trains the models and monitors validation loss to restore the best model weights.
5. Image Classification: Predicts one of the eight supplied labels using an eight-class Softmax output layer.
6. Test-Time Augmentation: Optionally averages predictions across five transformed image views.
7. Model Evaluation: Compares model performance using multiple classification metrics.
8. Web Application: Integrates the trained model into a Streamlit application for interactive predictions.

📊 Dataset Information

- Dataset: Fingerprint Blood Group Classification Dataset
- Source: Kaggle
- Total Images: 6,000
- Training Images: 4,803
- Validation Images: 1,197
- Image Size: 128 × 128 pixels
- Image Format: Grayscale
- Number of Classes: 8

📈 Experimental Results

The improved CNN achieved the following results on the validation dataset:

Evaluation Metric| Improved CNN
Accuracy| 86.38%
Balanced Accuracy| 86.21%
Macro-F1 Score| 0.8654

Test-Time Augmentation Results

With five-view Test-Time Augmentation, the improved CNN achieved:

Evaluation Metric| Improved CNN + TTA
Accuracy| 87.55%
Balanced Accuracy| 87.39%
Macro-F1 Score| 0.8763

These results show the performance observed on the project's validation split.

🛠️ Technologies Used

- Programming Language: Python 3.12
- Deep Learning: TensorFlow, Keras
- Model Architecture: Convolutional Neural Networks (CNN)
- Numerical Computing: NumPy
- Data Handling: pandas
- Image Processing: Pillow (PIL)
- Model Evaluation: scikit-learn
- Data Visualization: Matplotlib
- Web Application: Streamlit
- Data Integrity: SHA-256 Hashing

⚠️ Important Disclaimer

BioPrint is an experimental academic project that evaluates how CNN models classify fingerprint images against the labels supplied in a public dataset.

The results do not establish a scientifically validated relationship between fingerprint patterns and blood groups. The model is not medically validated and must not be used for blood-group identification, diagnosis, treatment, or any medical decision. Its prediction scores are uncalibrated model outputs, not clinical probabilities.

🎓 Project Type

Academic Major Project – Bachelor of Technology (B.Tech), Computer Science and Engineering.

BioPrint demonstrates the application of deep learning, image processing, CNN architecture evaluation, and web application development in an experimental image-classification task.
