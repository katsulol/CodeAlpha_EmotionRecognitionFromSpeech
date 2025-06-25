## Emotion Recognition from Speech

This project detects human emotions (such as happy, sad, angry, etc.) from speech audio using deep learning.

We used the TESS dataset, which contains clear audio recordings labeled with different emotions.

## Project Flow:

1. Audio Preprocessing
   - Extracted MFCC (Mel-Frequency Cepstral Coefficients) features from .wav files using Librosa.

2. Model Architecture
   - Built and trained a Convolutional Neural Network (CNN) to classify the MFCCs into emotion categories.

3. Training Optimization
   - Used EarlyStopping to prevent overfitting.
   - Achieved nearly 100% accuracy on the test set.

4. Explainable AI
   - Applied Grad-CAM using tf-keras-vis to visualize which parts of the MFCC input the model focused on when making decisions.

## Tech Stack:

- Python (Google Colab)
- TensorFlow / Keras
- Librosa
- scikit-learn
- matplotlib, seaborn
- tf-keras-vis (for Grad-CAM)

This pipeline is accurate, interpretable, and suitable for research or real-world emotion detection use cases.
