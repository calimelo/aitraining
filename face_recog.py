import cv2
import face_recognition
import numpy as np

# Load a known image (Database)
known_image = face_recognition.load_image_file("bill1.png")
known_encoding = face_recognition.face_encodings(known_image)[0]  # Extract facial features

# Load an unknown image for testing
unknown_image = face_recognition.load_image_file("bill2.jpg")
unknown_encodings = face_recognition.face_encodings(unknown_image)

# Check if a face was found
if len(unknown_encodings) > 0:
    unknown_encoding = unknown_encodings[0]  # Use the first detected face

    # Compare the faces
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)

    if results[0]:
        print(f"✅ Match Found! Confidence Score: {1 - distance[0]:.2f}")
    else:
        print("❌ No Match Found.")
else:
    print("⚠ No face detected in the image.")

# Display the image
cv2.imshow("Unknown Face", cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
