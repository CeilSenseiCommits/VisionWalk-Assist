import cv2
import torch
import numpy as np
import threading
import time
import os
import random
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = 'best.pt'       # Your trained model
DEPTH_SCALE_FACTOR = 400     # Adjusted for Laptop/Webcam
SAFE_DISTANCE = 20.0         # Large distance to ensure it speaks for testing

# ==========================================
# 1. AUDIO SYSTEM (COLLISION-PROOF FIX)
# ==========================================
class VoiceAlert:
    def __init__(self):
        self.is_busy = False 

    def speak(self, text):
        # If busy, ignore new requests immediately (Anti-Stacking)
        if self.is_busy: return 

        self.is_busy = True
        
        # Run in background thread so video doesn't freeze
        t = threading.Thread(target=self._run_speech_logic, args=(text,))
        t.daemon = True 
        t.start()

    def _run_speech_logic(self, text):
        # Generate a unique filename every time to prevent File Locks
        unique_name = f"alert_{int(time.time())}_{random.randint(1,999)}.mp3"
        
        try:
            print(f"🔊 GENERATING: {text}")
            
            # 1. Create the audio file
            tts = gTTS(text=text, lang='en')
            tts.save(unique_name)
            
            # 2. Play it
            playsound(unique_name)
            
            # 3. Cleanup: Delete the file immediately after playing
            if os.path.exists(unique_name):
                os.remove(unique_name)
            
        except Exception as e:
            print(f"❌ AUDIO ERROR: {e}")
            
        finally:
            # 4. Release the lock so new alerts can come in
            self.is_busy = False
            
            # Double check cleanup (in case crash happened before delete)
            if os.path.exists(unique_name):
                try:
                    os.remove(unique_name)
                except:
                    pass

voice = VoiceAlert()

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("Loading YOLOv8 (best.pt)...")
model = YOLO(MODEL_NAME)  

print("Loading MiDaS (Depth)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# ==========================================
# 3. MAIN LOOP
# ==========================================
# Switch to your camera (0 for Laptop, URL for Phone)
cap = cv2.VideoCapture('http://100.80.95.47:4747/video') 

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

print("System Active. Point camera at a door. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- Depth Calculation ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transforms(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # --- Detection ---
    results = model(frame, verbose=False, conf=0.5)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Identify Class
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id] 
            
            # Calculate Distance
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cy = min(cy, depth_map.shape[0]-1)
            cx = min(cx, depth_map.shape[1]-1)
            raw_depth = depth_map[cy, cx]
            dist_meters = DEPTH_SCALE_FACTOR / raw_depth if raw_depth > 0 else 0

            # Debug Print (Verify logic)
            print(f"DEBUG: Found '{class_name}' at {dist_meters:.2f}m")

            # ==========================================
            # 4. FLEXIBLE LOGIC (Handles 'open', 'open_door', etc.)
            # ==========================================
            
            # Check if name contains 'open' (Handles "open" OR "open_door")
            if 'open' in class_name:
                color = (0, 255, 0) # Green
                status_text = "OPEN"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{status_text}: {dist_meters:.1f}m"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if dist_meters <= SAFE_DISTANCE:
                    msg = f"Opened door {int(dist_meters)} meters ahead."
                    voice.speak(msg)

            # Check if name contains 'closed' (Handles "closed" OR "closed_door")
            elif 'closed' in class_name:
                color = (0, 0, 255) # Red
                status_text = "CLOSED"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{status_text}: {dist_meters:.1f}m"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if dist_meters <= SAFE_DISTANCE:
                    msg = "Closed door."
                    voice.speak(msg)

    # Show result
    cv2.imshow("Navigation Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()