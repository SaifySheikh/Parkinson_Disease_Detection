---

# **Parkinson's Disease Detection Using Voice and Spiral Image Data**

This project leverages machine learning models to detect Parkinson's disease by analyzing **voice data** and **hand-drawn spiral images**. By utilizing advanced classification techniques, the system provides an accurate diagnosis and aids in the early detection of Parkinson's disease.

---

## **Features**
- **Dual Data Analysis**:
  - **Voice Data**: Extracts features such as jitter, shimmer, and pitch variation.
  - **Spiral Images**: Analyzes hand-drawn spiral patterns for tremor detection.
- **Hybrid Model**: Combines results from voice and image models, selecting the output with the higher probability for the final prediction.
- **ML Models**:
  - **Voice Data**: SVM, GRU, LSTM, XGBoost, CNN-LSTM.
  - **Image Data**: Convolutional Neural Networks (CNNs).
- **Efficient Processing**: Uses feature extraction, data augmentation, and optimized architectures to improve accuracy and reduce latency.

---

## **Technologies Used**
### **Programming Languages & Frameworks**:
- Python
- TensorFlow, Keras
- Scikit-learn
- OpenCV (for image preprocessing)

### **Libraries**:
- Numpy, Pandas (data manipulation)
- Matplotlib, Seaborn (data visualization)
- Librosa (audio feature extraction)

### **Dataset**:
- **Voice Data**: [Oxford Parkinson’s Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Spiral Images**: Hand-drawn spiral datasets for detecting tremors.

---

## **Project Workflow**
### **1. Data Preprocessing**:
- **Voice Data**: Extract features like jitter, shimmer, and harmonic-to-noise ratio.
- **Image Data**: Resize, normalize, and augment spiral drawings.

### **2. Model Training**:
- Train separate models for voice and image data.
- Evaluate models based on accuracy, precision, recall, and F1-score.

### **3. Hybrid Classification**:
- Combine the predictions from the voice and image models.
- Select the output with the higher confidence probability.

### **4. Deployment**:
- Develop a user-friendly interface for uploading voice recordings and images.
- Display diagnostic results with probability scores.

---

## **Setup Instructions**
### **1. Clone the Repository**:
```bash
git clone https://github.com/your-username/parkinsons-detection.git
cd parkinsons-detection
```

### **2. Install Dependencies**:
Make sure you have Python 3.8+ installed. Then, run:
```bash
pip install -r requirements.txt
```

### **3. Prepare the Dataset**:
- Download the [voice dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) and place it in the `data/voice/` directory.
- Add spiral images to the `data/images/` directory.

### **4. Run the Application**:
```bash
python main.py
```

---

## **Usage**
1. Upload a voice recording or a spiral image.
2. The system processes the input and provides a probability score for Parkinson's disease detection.
3. View results and download the report.

---

## **Future Enhancements**
- Integrate real-time voice recording and spiral drawing inputs.
- Use more diverse datasets to improve model robustness.
- Deploy the system as a web or mobile application.

---

### **Implementation**:
- **Demonstration**: [click here for demo](https://youtu.be/rxswqWIWhds)

