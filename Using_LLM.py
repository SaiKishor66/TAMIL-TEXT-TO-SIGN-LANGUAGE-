import streamlit as st
import cv2
import google.generativeai as genai
import asyncio

genai.configure(api_key="AIzaSyBvrpTVHpJZSdH851VWcl5I7OnUagUusTQ")
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = """
You are a Sign Language Translator tasked with identifying the meaning of a sign language gesture shown in an image.
Please refer to the list of class labels below and select the most appropriate one based on the gesture in the image.

Return only the corresponding number for the label, with no additional text or symbols.

Class labels:
0: "Love"
1: "Nothing"
2: "Truth"
3: "Good Habits"
4: "Happy"
5: "Heart"
6: "Hello"
7: "Life"
8: "Victory"

Output: (Only the number, e.g., 0, 1, 2, etc.)
"""

class_labels = {
    0: 'அன்பு',
    1: 'ஒன்றுமில்லை',  # Default hand sign
    2: 'உண்மை',
    3: 'நல்வாழ்த்துக்கள்',
    4: 'மகிழ்ச்சி',
    5: 'மனம்',
    6: 'வணக்கம்',
    7: 'வாழ்க்கை',
    8: 'வெற்றி'
}

async def predict(image):
    await asyncio.sleep(1)  # Simulate delay
    predicted_label = 1  # Replace this with actual prediction logic
    print(f"LLM Response: {predicted_label}")
    return class_labels[predicted_label]

def main():
    st.title("Real-time Sign Language Interpretation")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to open webcam.")
        return

    if 'predicted_label' not in st.session_state:
        st.session_state['predicted_label'] = "ஒன்றுமில்லை"

    stframe = st.empty()
    prediction_box = st.empty()
    stop_button = st.button("Stop")

    async def display_prediction():
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to capture frame.")
                break

            stframe.image(frame, channels="BGR")

            if not getattr(st.session_state, 'predicting', False):
                st.session_state.predicting = True
                st.session_state.predicted_label = await predict(frame)
                st.session_state.predicting = False

            prediction_box.markdown(f"**Predicted Sign:** {st.session_state['predicted_label']}")

        cap.release()

    asyncio.run(display_prediction())

if __name__ == "__main__":
    main()
