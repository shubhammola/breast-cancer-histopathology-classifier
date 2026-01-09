# 🧬 Breast Cancer Histopathology Image Classification

**Breast Cancer Histopathology Image Classification** is a deep learning–based project that focuses on the automated classification of microscopic breast tissue images as **Benign** or **Malignant**.

The system leverages **Convolutional Neural Networks (CNNs)** and **transfer learning techniques** to assist in early breast cancer detection and improve diagnostic accuracy.

This project was developed as a **major academic project**, with the objective of combining **machine learning**, **deep learning**, and **web technologies** to build an **end-to-end medical image classification system** that can support healthcare professionals in diagnosis.

## 🎯 Problem Statement

Histopathology is the **gold standard** for diagnosing breast cancer, but manual analysis of microscopic images is **time-consuming** and highly dependent on the expertise of pathologists.

The objective of this project is to use **AI-driven image classification** to:

- Reduce diagnostic time  
- Improve consistency and accuracy  
- Assist medical professionals with reliable decision support

## 🛠️ Tech Stack

### Programming & Machine Learning
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV  
- Scikit-learn  

### Web Technologies
- Flask  
- HTML, CSS, Bootstrap  

### Tools
- Google Colab (model training)  
- GitHub (version control)

## 📂 Dataset

- **BreakHis Dataset**
- Microscopic breast cancer images
- Multiple magnification levels (40X, 100X, 200X, 400X)
- Binary classification: **Benign** vs **Malignant**

> The dataset is not included in this repository due to size and licensing constraints.

## 🧠 Model & Approach

- **Convolutional Neural Networks (CNNs)** used for feature extraction and classification
- Image preprocessing includes resizing, normalization, and data augmentation
- **Transfer learning** applied using pretrained architectures to improve performance on limited data
- Evaluation metrics include:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

The model achieved **high classification accuracy (>90%)** on validation data.

## 🔄 System Workflow

1. User uploads a histopathology image  
2. Image preprocessing (resizing and normalization)  
3. CNN-based prediction  
4. Result displayed as **Benign** or **Malignant**  
5. Recommendation to consult medical professionals if required
## 🧩 Project Structure

```
breast-cancer-histopathology-classifier/
│
├── app.py                  # Flask application & model integration
├── models/                 # Trained CNN models
├── static/                 # Images & static assets
├── templates/              # HTML templates
├── sample_test_images/     # Sample images for testing
├── Breast_Cancer_Prediction.ipynb
├── requirements.txt
├── README.md
```

## ▶️ How to Run Locally

### 1. Clone the repository
```
git clone https://github.com//breast-cancer-histopathology-classifier.git
cd breast-cancer-histopathology-classifier
```

### 2. Install dependencies

`pip install -r requirements.txt`

### 3. Run the application
`python app.py or python3 app.py - Based on python verison on your Mac or PC`

### 4. Open in browser
`http://127.0.0.1:5000/`

## 🧪 Testing & Validation

- Unit testing for image preprocessing  
- Model testing using labeled validation data  
- End-to-end system testing via Flask interface  
- User acceptance testing with medical professionals  
- Performance testing for response time and reliability  

## 🚀 Future Enhancements

- Explainable AI (Grad-CAM visualization)  
- Multi-class classification of cancer subtypes  
- Integration with clinical and genomic data  
- Model optimization for faster inference  
- Dockerized deployment  
- Federated learning for privacy-preserving training  

## 👥 Team Contribution

This project was developed as a **team-based major academic project**.

**My contributions included:**
- Image preprocessing pipeline  
- CNN model training and evaluation  
- Integration of trained model with Flask backend  
- Result analysis and validation  

## ⚠️ Disclaimer

This project is intended for **academic and research purposes only**.  
It is **not a replacement for professional medical diagnosis**. Patients must consult certified healthcare professionals for medical decisions.

## 👤 Author

**Shubham Mola**
