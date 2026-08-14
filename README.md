# Gamified Sign Language Tutor

A real-time sign language recognition and learning system using computer vision and machine learning.

## About the Project

This project recognizes hand gestures through a webcam and converts them into letters and symbols. Recognized letters can be combined to form sentences.

## Features

- Real-time hand gesture detection
- Alphabet recognition using one hand
- Symbol recognition using two hands
- Real-time sentence formation
- Background segmentation
- Interactive learning interface

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Random Forest

## Project Files

- `word.py` - Main application
- `collect_data.py` - Alphabet data collection
- `collect_symbols.py` - Symbol data collection
- `train_model.py` - Alphabet model training
- `train_symbols.py` - Symbol model training
- `dataset.csv` - Alphabet dataset
- `dataset_symbols.csv` - Symbol dataset
- `model.pkl` - Alphabet model
- `model_symbols.pkl` - Symbol model

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt

# RUN THIS PROJECT
python word.py

#AUTHOR
Ritik Raj

B.Tech Information Technology
Narula Institute Of Technology