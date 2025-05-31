import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to load images from a directory
def load_data_from_directory(directory):
    X = []
    y = []
    
    print(f"Loading data from {directory}...")
    for emotion_idx, emotion in enumerate(emotion_labels):
        emotion_dir = os.path.join(directory, emotion)
        if os.path.exists(emotion_dir):
            files = os.listdir(emotion_dir)
            print(f"Found {len(files)} images for {emotion}")
            
            for img_file in files:
                img_path = os.path.join(emotion_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to 48x48 if not already that size
                    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
                    X.append(img.flatten())
                    y.append(emotion_idx)
    
    return np.array(X), np.array(y)

# Load training and testing data
X_train, y_train = load_data_from_directory('train')
X_test, y_test = load_data_from_directory('test')

print(f"Training data: {X_train.shape[0]} samples")
print(f"Testing data: {X_test.shape[0]} samples")

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest classifier
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model and scaler
joblib.dump(model, 'emotion_model_sklearn.joblib')
joblib.dump(scaler, 'emotion_scaler.joblib')
print("Model and scaler saved successfully")
