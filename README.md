# Skin Tone Analyzer

This project analyzes skin tones from images using the FairFace dataset and suggests clothing color recommendations. It uses an SVM model trained on HSV features extracted from face regions.

## Files
- `train_skin_tone_model.py`: Trains the SVM model.
- `skin_tone_analyzer.py`: Analyzes skin tone and suggests colors.
- `download_fairface.py`: Downloads the FairFace dataset.
- `skin_tone_svm_model.pkl`: Trained model.
- `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`: DNN face detection files.
- `fairface_train.csv`, `fairface_validation.csv`: Dataset metadata.
- `predictions.txt`, `training_output.txt`: Debug outputs.

## Setup
1. Install dependencies:
   ```bash
   pip install opencv-python numpy pandas scikit-learn joblib imbalanced-learn

2. Download FairFace dataset using download_fairface.py.
3. Run skin_tone_analyzer.py to analyze images.

Notes
The FairFace dataset images are not included due to size. Use download_fairface.py to fetch them.
The model may have biases due to dataset label issues