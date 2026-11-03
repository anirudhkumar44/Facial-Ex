# 😊 Facial Expression Recognition Using CNN

## 📌 Project Overview
Facial Expression Recognition is a deep learning-based image classification project that identifies visible human facial expressions using a Convolutional Neural Network (CNN).

The project uses Python, TensorFlow, Keras, and Streamlit to predict facial expressions from uploaded images.

## 🎯 Objectives
- Recognize facial expressions using deep learning.
- Classify images into seven expression categories.
- Develop an interactive web application using Streamlit.
- Display prediction confidence and class probabilities.

## 🧠 Expression Categories
The model identifies seven facial expression categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## 📊 Dataset Description

The project uses a facial expression dataset containing seven categories.

Each category contains approximately 50–60 images.

| Expression | Images |
|---|---:|
| Angry | 50–60 |
| Disgust | 50–60 |
| Fear | 50–60 |
| Happy | 50–60 |
| Neutral | 50–60 |
| Sad | 50–60 |
| Surprise | 50–60 |
| **Total** | **350–420** |

### Dataset Structure

```text
FER_56_per_class/
└── train/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### Dataset Information
- Image format: JPG, JPEG, PNG
- Image type: RGB facial images
- Image size: 64 × 64 pixels (model input)
- Number of classes: 7
- Images per class: 50–60
- Total images: Approximately 350–420

**Note:** The dataset is used for training the CNN model separately. The Streamlit application loads the previously trained model and performs predictions without training.

## 🛠️ Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Streamlit
- Pillow
- Convolutional Neural Network (CNN)

## 📂 Project Structure

```text
FACIAL-EXPRESSION/
│
├── app.py
├── facial_expression.keras
├── classes.json
├── requirements.txt
├── README.md
│
└── FER_56_per_class/
    └── train/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```

## ⚙️ CNN Model Architecture

The CNN model consists of:

1. Input Layer – 64 × 64 × 3 RGB image.
2. Rescaling Layer – Normalizes pixel values.
3. Convolutional Layer – 16 filters with ReLU activation.
4. Max Pooling Layer.
5. Convolutional Layer – 32 filters with ReLU activation.
6. Max Pooling Layer.
7. Flatten Layer.
8. Dense Layer – 64 neurons with ReLU activation.
9. Output Layer – Softmax activation for seven classes.

## 📦 Installation

### Step 1: Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open Project Folder

```bash
cd FACIAL-EXPRESSION
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

## ▶️ Run Application

Execute the following command:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📸 How to Use

1. Open the application.
2. Wait for the trained model to load.
3. Upload a facial image.
4. Click the Predict Expression button.
5. View the predicted expression.
6. Check the prediction confidence.
7. View probabilities for all expression classes.

## 📁 Required Model Files

The following files are required:

- `facial_expression.keras` – Trained CNN model.
- `classes.json` – Expression class labels.

Both files must be in the same directory as `app.py`.

## 🚀 Deployment

This project can be deployed using Streamlit Community Cloud.

1. Upload the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Click Deploy.

Make sure all required model files are available in the repository.

## 🔮 Future Scope

- Improve prediction accuracy using larger datasets.
- Add real-time facial expression detection.
- Integrate webcam support.
- Improve CNN architecture.
- Add more visualization features.

## ⚠️ Disclaimer

This project is developed for educational purposes only. Facial expression predictions may be inaccurate and do not establish a person's actual emotions or mental state. Use facial images only with appropriate permission.

## 👨‍💻 Conclusion

The Facial Expression Recognition project demonstrates how CNN and deep learning techniques can be used for image classification. The Streamlit application provides a user-friendly interface for predicting visible facial expression categories.
