import cv2
import numpy as np
import joblib
import os

# Load face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Load trained SVM model
model = joblib.load("skin_tone_svm_model.pkl")

def extract_hsv_features(image_path):
    """Extract mean HSV features from the face region or entire image."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Resize image to model’s expected size
    img = cv2.resize(img, (300, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # DNN-based detector
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    faces = []
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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2, minSize=(15, 15))
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1, minSize=(10, 10))
    
    if len(faces) == 0:
        print(f"No face detected in: {image_path}, using entire image")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return [np.mean(h), np.mean(s), np.mean(v)]
    
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    if face_img.size == 0:
        print(f"Invalid face region in: {image_path}, using entire image")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return [np.mean(h), np.mean(s), np.mean(v)]
    
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return [np.mean(h), np.mean(s), np.mean(v)]

def remap_skin_tone(old_skin_tone, hsv):
    """Remap old skin tone predictions to new skin tones based on user observations."""
    # Old mappings (from training):
    # Dark: Black (0), Indian (2)
    # Cool: White (1)
    # Neutral: East Asian (3), Southeast Asian (4)
    # Warm: Middle Eastern (5), Latino (6)
    
    # New desired mappings:
    # Cool: Race 0 (East Asian), 4 (Yellowish), 3 (White)
    # Dark: Race 1 (Indian)
    # Warm: Race 2 (Medium tan), 5 (Orangish)
    # Neutral: Race 6 (Older Asian)
    
    hue, saturation, value = hsv
    
    if old_skin_tone == "Dark":
        # Old Dark was Black (0) or Indian (2)
        # Black (0) should now be Cool (East Asian), Indian (2) should be Dark
        if 20 <= hue <= 50 and saturation > 80:  # Indian: warmer, more saturated
            return "Dark"  # Race 1 (Indian)
        else:
            return "Cool"  # Race 0 (East Asian)
    
    elif old_skin_tone == "Cool":
        # Old Cool was White (1), should now be Cool (Race 3)
        return "Cool"
    
    elif old_skin_tone == "Neutral":
        # Old Neutral was East Asian (3) or Southeast Asian (4)
        # East Asian (3) → Cool (Race 3), Southeast Asian (4) → Cool (Race 4)
        return "Cool"
    
    elif old_skin_tone == "Warm":
        # Old Warm was Middle Eastern (5) or Latino (6)
        # Middle Eastern (5) → Warm, Latino (6) → Neutral
        if hue > 25 and saturation > 70:  # Middle Eastern/Orangish: warmer, saturated
            return "Warm"  # Race 2, 5
        else:
            return "Neutral"  # Race 6
    
    return "Cool"  # Default to Cool to reduce Neutral bias

def suggest_clothing_colors(skin_tone):
    """Suggest clothing colors based on skin tone."""
    suggestions = {
        "Dark": [
            "Vibrant colors (red, yellow, royal blue)",
            "White",
            "Pastels (lavender, mint)",
            "Avoid: Dark brown, black (unless paired with bright accents)"
        ],
        "Warm": [
            "Earth tones (olive green, mustard, terracotta)",
            "Warm reds, oranges",
            "Cream, beige",
            "Avoid: Cool pastels (baby blue, lavender)"
        ],
        "Neutral": [
            "Neutral tones (gray, taupe, navy)",
            "Soft pastels (blush pink, sage green)",
            "Bold colors (emerald, burgundy)",
            "Avoid: Neon colors"
        ],
        "Cool": [
            "Cool tones (blue, purple, silver)",
            "Jewel tones (sapphire, ruby)",
            "White, light gray",
            "Avoid: Warm oranges, yellows"
        ]
    }
    return suggestions.get(skin_tone, ["No specific recommendations"])

def analyze_skin_tone(image_path):
    """Analyze skin tone and suggest clothing colors."""
    hsv = extract_hsv_features(image_path)
    if hsv is None:
        return "Error: Could not process image", []
    
    # Predict skin tone using the old model
    hsv_array = np.array([hsv])  # Reshape for SVM
    old_skin_tone = model.predict(hsv_array)[0]
    
    # Remap to new skin tone
    new_skin_tone = remap_skin_tone(old_skin_tone, hsv)
    
    # Log for debugging
    print(f"Image: {image_path}, Old Skin Tone: {old_skin_tone}, New Skin Tone: {new_skin_tone}, HSV: {hsv}")
    with open("predictions.txt", "a") as f:
        f.write(f"Image: {image_path}, Old: {old_skin_tone}, New: {new_skin_tone}, HSV: {hsv}\n")
    
    # Get clothing color suggestions
    colors = suggest_clothing_colors(new_skin_tone)
    
    return new_skin_tone, colors

def main():
    """Main function to run the skin tone analyzer."""
    print("Enter the path to an image (e.g., test.jpg, or 'q' to quit):")
    while True:
        image_path = input("> ").strip()
        if image_path.lower() == 'q':
            break
        skin_tone, colors = analyze_skin_tone(image_path)
        if isinstance(skin_tone, str) and skin_tone.startswith("Error"):
            print(skin_tone)
        else:
            print(f"Skin Tone: {skin_tone}")
            print("Recommended Clothing Colors (General):")
            for color in colors:
                print(f"- {color}")

if __name__ == "__main__":
    main()
    