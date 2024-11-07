import streamlit as st
import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model  # Add this import

# Load the sign language interpretation model
model = load_model("best_model.keras")  # Change this path to your model file

# Define class labels
class_labels = {
    0: 'அன்பு',
    1: 'அருள்',
    2: 'உண்மை',
    3: 'நல்வாழ்த்துக்கள்',
    4: 'மகிழ்ச்சி',
    5: 'மனம்',
    6: 'வணக்கம்',
    7: 'வாழ்க்கை',
    8: 'வெற்றி'
}

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to make predictions on webcam frames
@st.cache_data
def predict(image):
    # Preprocess the image (resize, normalize, etc.)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    image = image / 255.0

    # Make prediction using the model
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    return predicted_label

# Function to speak the predicted sign
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit app
def main():
    st.title("Real-time Sign Language Interpretation")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open webcam.")
        return

    if 'stop' not in st.session_state:
        st.session_state['stop'] = False

    if 'predicted_label' not in st.session_state:
        st.session_state['predicted_label'] = ""

    def stop():
        st.session_state['stop'] = True

    stframe = st.empty()
    prediction_box = st.empty()
    st.button("Stop", on_click=stop, key="stop_button")

    # Main loop to capture and process webcam frames
    while cap.isOpened() and not st.session_state['stop']:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to capture frame.")
            break

        # Make prediction on the current frame
        st.session_state['predicted_label'] = predict(frame)

        # Display the webcam frame and predicted label
        stframe.image(frame, channels="BGR")
        prediction_box.markdown(f"**Predicted Sign:** {st.session_state['predicted_label']}")

    # Release webcam
    cap.release()

    # Button to trigger text-to-speech
    if st.button("Speak"):
        if st.session_state['predicted_label']:
            speak(st.session_state['predicted_label'])
        else:
            st.warning("No sign predicted yet.")

if __name__ == "__main__":
    main()
