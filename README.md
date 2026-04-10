# Parkinson’s Disease Detection using CNN

## Overview
This project focuses on detecting Parkinson’s Disease using a Convolutional Neural Network (CNN). The model is trained on medical data to identify patterns and features that help in early diagnosis. The goal is to build an efficient and reliable AI-based system for supporting healthcare professionals.

## Features
- Deep Learning model based on CNN
- Data preprocessing and normalization
- Model training and evaluation
- Accuracy and performance visualization
- Easy-to-use prediction interface

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

## Dataset
The dataset used in this project contains medical or imaging data related to Parkinson’s Disease.  
(You can add dataset link here if public)

## Project Structure

Parkinson_Disease_Detection/
│
├── dataset/
│   ├── training/
│   │   ├── no/
│   │   └── yes/
│   ├── testing/
│   │   ├── no/
│   │   └── yes/
│
├── parkinsons_train.py        # CNN model training script
├── predict_patient.py         # Predict single or batch patient reports
├── parkinsons_cnn_brain.h5    # Saved trained model (after training)
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

## Results
- Achieved good accuracy in detecting Parkinson’s Disease
- Model performance evaluated using metrics like accuracy, precision, recall

## Future Improvements
- Improve dataset size and diversity
- Deploy as a web application
- Integrate with real-time medical systems
- Optimize model performance

## Contributing
Contributions are welcome. Feel free to fork the repo and submit a pull request.

## License
This project is licensed under the MIT License.

## Author
Ashwin Yadav
