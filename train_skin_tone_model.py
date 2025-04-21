import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib
from collections import Counter
from imblearn.over_sampling import SMOTE

# Load face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

def extract_hsv_features(image_path):
    """Extract mean HSV features from the face region or entire image if detection fails."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    
    # Resize image to model’s expected size
    img = cv2.resize(img, (300, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # DNN-based detector
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    faces = []
    method = "DNN"
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (x, y, x2, y2) = box.astype("int")
            if x >= 0 and y >= 0 and x2 > x and y2 > y:
                faces.append((x, y, x2-x, y2-y))
                break
    
    # Fallback to Haar Cascade
    if len(faces) == 0:
        method = "Haar"
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2, minSize=(15, 15))
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1, minSize=(10, 10))
    
    # Fallback to entire image
    if len(faces) == 0:
        method = "Entire"
        print(f"No face detected in: {image_path}, using entire image")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return [np.mean(h), np.mean(s), np.mean(v)], method
    
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    if face_img.size == 0:
        method = "Entire"
        print(f"Invalid face region in: {image_path}, using entire image")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return [np.mean(h), np.mean(s), np.mean(v)], method
    
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return [np.mean(h), np.mean(s), np.mean(v)], method

def map_race_to_category(race):
    """Map FairFace race labels to skin tone categories (original mappings)."""
    race = int(float(race)) if isinstance(race, (str, float)) else race
    
    # Original mappings from paper
    if race == 0:  # Black
        return "Dark"
    elif race == 1:  # White
        return "Cool"
    elif race == 2:  # Indian
        return "Dark"
    elif race in [3, 4]:  # East Asian, Southeast Asian
        return "Neutral"
    elif race == 5:  # Middle Eastern
        return "Warm"
    elif race == 6:  # Latino
        return "Warm"
    else:
        print(f"Unknown race value: {race}")
        return "Neutral"

def prepare_dataset(data_dir, label_file, max_images=5000):
    """Prepare dataset by extracting HSV features and mapping labels."""
    df = pd.read_csv(label_file)
    # Sample evenly across race groups
    df = df.groupby('race').apply(lambda x: x.sample(min(len(x), max_images//7)), include_groups=False).reset_index(drop=True)
    features = []
    labels = []
    count = 0
    skipped = 0
    detection_methods = {'DNN': 0, 'Haar': 0, 'Entire': 0}
    race_counts = Counter()
    
    for _, row in df.iterrows():
        if count >= max_images:
            break
        img_path = row['image_path'].replace('\\', '/')  # Normalize path separators
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            skipped += 1
            continue
        hsv, method = extract_hsv_features(img_path)
        if hsv is None:
            skipped += 1
            continue
        race = row['race']
        race_counts[race] += 1
        label = map_race_to_category(race)
        print(f"Image: {img_path}, Race: {race}, Label: {label}, HSV: {hsv}")  # Debug
        features.append(hsv)
        labels.append(label)
        detection_methods[method] += 1
        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images, Skipped {skipped} images")
            print(f"Detection methods: {detection_methods}")
            print(f"Race distribution: {race_counts}")
    
    # Debug: Print distributions
    label_counts = Counter(labels)
    print("Label distribution:", label_counts)
    print("Race distribution:", race_counts)
    print(f"Total images processed: {count}, Total images skipped: {skipped}")
    print(f"Detection methods used: {detection_methods}")
    if len(label_counts) <= 1:
        print("Error: Only one class found. Check race values or mapping.")
        return np.array([]), np.array([])
    
    return np.array(features), np.array(labels)

def train_model():
    """Train SVM model and save it."""
    # Paths to FairFace dataset
    data_dir = "fairface/train"
    label_file = "fairface/fairface_train.csv"
    
    # Prepare dataset
    print("Preparing dataset...")
    X, y = prepare_dataset(data_dir, label_file)
    if len(X) == 0:
        print("No valid data extracted. Check dataset paths, images, or face detection.")
        return None, 0
    
    # Balance classes using SMOTE
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    label_counts = Counter(y)
    print("Balanced label distribution:", label_counts)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM
    print("Training SVM model...")
    model = SVC(kernel='rbf', C=10.0, probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    X_val, y_val = prepare_dataset("fairface/validation", "fairface/fairface_validation.csv", max_images=1000)
    if len(X_val) > 0:
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    # Save model
    joblib.dump(model, "skin_tone_svm_model.pkl")
    print("Model saved as skin_tone_svm_model.pkl")
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = train_model()
    if model is None:
        print("Training failed.")
    elif accuracy >= 0.85:
        print("Model meets accuracy requirement (>=85%).")
    else:
        print("Accuracy below 85%. Consider tuning hyperparameters or adding more data.")