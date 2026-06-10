import base64
import cv2
import numpy as np
import mediapipe as mp
import pickle  # 📦 Added to load your trained model file!
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# MediaPipe Tasks API Imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Define the 21-point hand joint connections map
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# 💾 1. Safely load your trained sign language model brain
try:
    with open("sign_model.pkl", "rb") as f:
        ai_brain = pickle.load(f)
    print("🧠 AI Brain model linked and ready for real-time translation!")
except Exception as e:
    ai_brain = None
    print(f"⚠️ Could not load sign_model.pkl: {e}. Ensure the file is in your root folder.")

# Configure the MediaPipe detector
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

def process_frame(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return "-", None

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (320, 240))
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect(mp_image)
    prediction = "-"
    
    if result.hand_landmarks:
        # Default fallback string
        prediction = "READING..."
        
        for hand_landmarks in result.hand_landmarks:
            pixel_pts = []
            row_features = []
            
            for lm in hand_landmarks:
                # 📊 2. Collect the spatial coordinates for the AI model
                row_features.extend([lm.x, lm.y, lm.z])
                
                px = int(lm.x * w)
                py = int(lm.y * h)
                pixel_pts.append((px, py))
            
            # 🔮 3. Pass those coordinates to the ML model for a prediction!
            if ai_brain is not None:
                try:
                    features_array = np.array(row_features).reshape(1, -1)
                    prediction = str(ai_brain.predict(features_array)[0])
                except Exception:
                    prediction = "ERROR READING"
            else:
                prediction = "READY (NO BRAIN)"
            
            # Draw the skeleton lines
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(pixel_pts) and end_idx < len(pixel_pts):
                    cv2.line(frame, pixel_pts[start_idx], pixel_pts[end_idx], (14, 165, 233), 2)
            
            # Draw tracking nodes
            for pt in pixel_pts:
                cv2.circle(frame, pt, 4, (248, 250, 252), -1)
                
    _, buffer = cv2.imencode('.jpg', frame)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return prediction, processed_img_base64

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            
            prediction, processed_frame = process_frame(image_bytes)
            
            await websocket.send_json({
                'prediction': prediction,
                'processed_frame': processed_frame
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Frame handling error: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)