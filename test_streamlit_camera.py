import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import shutil
import uuid

def app():
    st.header('Real-Time Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write('Welcome!')

    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())

    selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person'])
    min_confidence = st.slider('Confidence score', 0.0, 1.0, 0.5)

    if 'run_detection' not in st.session_state:
        st.session_state.run_detection = False
    if 'tfile_name' not in st.session_state:
        st.session_state.tfile_name = None
    if 'video_saved' not in st.session_state:
        st.session_state.video_saved = False

    start_detection = st.button('Start Camera')
    stop_detection = st.button('Stop Camera')

    if start_detection:
        st.session_state.run_detection = True
        unique_id = str(uuid.uuid4().hex)[:8]  # Generate unique ID
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        st.session_state.tfile_name = tfile.name
        st.session_state.output_path = os.path.join(os.getcwd(), f"output_{unique_id}.mp4")  # Output video path
        tfile.close()

    if stop_detection:
        st.session_state.run_detection = False

    if st.session_state.run_detection:
        cap = cv2.VideoCapture(0)  # Use the first webcam
        tfile_name = st.session_state.tfile_name
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 5  # Set FPS to 24 (adjust as needed)
        
        # Create VideoWriter object with correct FPS
        out_video = cv2.VideoWriter(tfile_name, cv2.VideoWriter_fourcc(*'h264'), fps, (width, height))

        stframe = st.empty()  # Placeholder for video frames

        while st.session_state.run_detection:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(frame)
            for detection in result[0].boxes.data:
                x0, y0, x1, y1, score, cls = detection[:6]
                score = float(score)
                cls = int(cls)
                object_name = model.names[cls]
                label = f'{object_name} {score:.2f}'

                if object_name in selected_objects and score > min_confidence:
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            out_video.write(frame)
            stframe.image(frame, channels="BGR")

        cap.release()
        out_video.release()

    if not st.session_state.run_detection and st.session_state.tfile_name:
        # Display saved video
        st.video(st.session_state.tfile_name)

        # Save video on button click
        if not st.session_state.video_saved and st.button("Save Video"):
            save_path = st.session_state.output_path
            shutil.move(st.session_state.tfile_name, save_path)
            st.session_state.video_saved = True
            st.write(f"Video saved to {save_path}")

if __name__ == "__main__":
    app()
