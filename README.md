# SVD Image Denoising Studio - Backend

This is the FastAPI backend for the **SVD Image Denoising Studio**, designed to demonstrate the mathematical concept of Singular Value Decomposition (SVD) for image compression and noise reduction. 

It exposes a REST API that injects controlled Gaussian noise into an uploaded image and then dynamically reconstructs the image by truncating less significant singular values.

## 🚀 Tech Stack
* **Python 3.10+**
* **FastAPI** (High-performance web framework)
* **OpenCV (`opencv-python-headless`)** (Image processing)
* **NumPy** (Linear algebra & Matrix operations)
* **Uvicorn** (ASGI server)

## ⚙️ Core Pipeline Features
1. **Gaussian Noise Injection**: Adds synthetic noise to the original image based on user-defined standard deviation ($\sigma$).
2. **Per-Channel SVD**: Decomposes the Red, Green, and Blue channels independently into $U$, $\Sigma$, and $V^T$ matrices.
3. **Adaptive Rank Truncation**: Discards high-frequency (noise) singular values, retaining only the dominant values (controlled by the user parameter $k$).
4. **Post-Processing**: Applies a gentle bilateral filter to preserve edges while removing minor artifacts, followed by slight sharpening.
5. **Difference Mapping**: Calculates the absolute difference between the original and denoised image, mapping it to a colorful heatmap for visualization.

## 🛠️ Local Development Setup

**1. Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the server:**
```bash
uvicorn main:app --reload
```
The server will start at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.

## 🌐 Deployment
This backend is optimized for deployment on cloud platforms like **Render**, **Railway**, or **Heroku**. 
* Ensure the build command is `pip install -r requirements.txt`
* Ensure the start command is `uvicorn main:app --host 0.0.0.0 --port $PORT`
* We specifically use `opencv-python-headless` to avoid missing UI library (`libGL`) errors on cloud servers.
