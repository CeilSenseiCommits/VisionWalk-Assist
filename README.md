# DoorSense-AI: Assistive Navigation System 🚪👁️

**DoorSense-AI** is a real-time assistive navigation system designed to help visually impaired users detect doors and navigate safely. By combining **YOLOv8** for object detection and **MiDaS** for monocular depth estimation, the system identifies whether doors are "Open" or "Closed," calculates the distance to them, and provides collision-prevention audio alerts.

## 🌟 Key Features
* **Real-Time Detection:** Instantly identifies open vs. closed doors using a custom-trained YOLOv8 model.
* **Depth Estimation:** Uses Intel's MiDaS to calculate the precise distance of the door in meters.
* **Smart Audio Alerts:** Uses Text-to-Speech (gTTS) to announce hazards (e.g., *"Opened door 3 meters ahead"*) without freezing the video feed.
* **Flexible Input:** Supports both laptop webcams and IP cameras (e.g., smartphones).

---

## ⚙️ Installation & Setup

Follow these steps to get the project running on your local machine.

### 1. Prerequisites
* **Python 3.8** or higher installed.
* A webcam (built-in or USB) or a smartphone with an IP Camera app.

### 2. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/YOUR_USERNAME/DoorSense-AI.git](https://github.com/YOUR_USERNAME/DoorSense-AI.git)
cd DoorSense-AI