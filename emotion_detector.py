import cv2
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('emotion_model_sklearn.joblib')
scaler = joblib.load('emotion_scaler.joblib')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            # Prepare image for prediction
            roi_flat = roi_gray.flatten().reshape(1, -1)
            roi_normalized = scaler.transform(roi_flat)
            
            # Predict emotion
            prediction = model.predict_proba(roi_normalized)[0]
            label = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Display results
            label_position = (x, y - 10)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"{label} {confidence:.1f}%", label_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Found", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("Real-Time Emotion Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
