import base64
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="SVD Image Denoising Studio")

K_RANGE = (10, 150)
DEFAULT_K = 50
ENERGY_THRESHOLD = 0.90
ADAPTIVE_K_FLOOR = 30
DEFAULT_NOISE_STD = 25

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Logger:
    def __init__(self):
        self.logs = []

    def log(self, msg: str):
        self.logs.append(msg)


def load_image(file_bytes: bytes, logger: Logger) -> np.ndarray:
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ValueError("Invalid image format")

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float64)
        logger.log(f"Image uploaded — {rgb_img.shape[1]}×{rgb_img.shape[0]} RGB")
        return rgb_img
    except Exception:
        logger.log("ERROR: Invalid image format")
        raise HTTPException(status_code=400, detail="Invalid image format")


def encode_image(img: np.ndarray) -> str:
    rgb_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", bgr_uint8)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def add_gaussian_noise(img: np.ndarray, std: float, logger: Logger) -> np.ndarray:
    noise = np.random.normal(0, std, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255)
    logger.log(f"Gaussian noise added (σ = {std})")
    return noisy


def select_adaptive_rank(singular_values: np.ndarray) -> int:
    if singular_values.size == 0:
        return 1
    energy = np.cumsum(singular_values) / np.sum(singular_values)
    k = int(np.searchsorted(energy, ENERGY_THRESHOLD) + 1)
    k = max(k, ADAPTIVE_K_FLOOR)
    return min(max(1, k), len(singular_values))


def denoise_channel_with_svd(channel: np.ndarray, selected_k: int) -> tuple[np.ndarray, int, int]:
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    adaptive_k = select_adaptive_rank(S)
    k = min(max(1, selected_k), len(S))

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    denoised_channel = U_k @ np.diag(S_k) @ Vt_k
    return denoised_channel, k, adaptive_k


def denoise_svd(
    noisy_img: np.ndarray,
    original_img: np.ndarray,
    selected_k: int,
    highlight_diff: bool,
    logger: Logger,
) -> tuple[np.ndarray, np.ndarray | None, list[int], list[int]]:
    logger.log("Applying SVD on noisy image (A = UΣVᵀ)")
    logger.log(f"Using 90% energy retention — user k = {selected_k}")

    denoised_channels = []
    channel_names = ("R", "G", "B")
    final_rank_values = []
    adaptive_rank_values = []

    for channel_name, channel in zip(channel_names, cv2.split(noisy_img)):
        denoised_channel, k, adaptive_k = denoise_channel_with_svd(channel, selected_k)
        denoised_channels.append(denoised_channel)
        final_rank_values.append(k)
        adaptive_rank_values.append(adaptive_k)
        logger.log(f"  {channel_name}: adaptive k={adaptive_k}, applied k={k}")

    logger.log(f"Top {final_rank_values[0]} singular values retained per channel")
    logger.log("Image reconstructed from truncated SVD")

    denoised_img = cv2.merge(denoised_channels)
    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)

    # Post-processing: edge-preserving smoothing + gentle sharpening
    denoised_img = cv2.bilateralFilter(denoised_img, d=9, sigmaColor=75, sigmaSpace=75)
    logger.log("Bilateral edge-preserving filter applied")

    # Gentle unsharp mask — just enough to restore clarity
    blurred = cv2.GaussianBlur(denoised_img, (0, 0), 2)
    denoised_img = cv2.addWeighted(denoised_img, 1.3, blurred, -0.3, 0)
    logger.log("Gentle sharpening applied")

    diff_img = None
    if highlight_diff:
        diff_img = cv2.absdiff(original_img.astype(np.uint8), denoised_img)
        diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
        logger.log("Difference map generated (|Original − Denoised|)")

    return denoised_img.astype(np.float64), diff_img, final_rank_values, adaptive_rank_values


@app.post("/denoise")
async def api_denoise(
    image: UploadFile = File(...),
    k: str = Form(str(DEFAULT_K)),
    noise_std: str = Form(str(DEFAULT_NOISE_STD)),
    highlight_diff: str = Form("false"),
):
    try:
        logger = Logger()

        # Parse params
        try:
            selected_k = int(k)
            selected_k = max(10, min(selected_k, 150))
        except ValueError:
            selected_k = DEFAULT_K

        try:
            noise_level = int(noise_std)
            noise_level = max(5, min(noise_level, 30))
        except ValueError:
            noise_level = DEFAULT_NOISE_STD

        show_diff = highlight_diff.lower() == "true"

        # Step 1: Load original
        original_img = load_image(await image.read(), logger)

        # Step 2: Add synthetic noise
        noisy_img = add_gaussian_noise(original_img, noise_level, logger)

        logger.log("Applying SVD on noisy image")
        denoised_img, diff_img, channel_ranks, adaptive_ranks = denoise_svd(
            noisy_img, original_img, selected_k, show_diff, logger
        )

        logger.log("Denoising completed")

        response_data = {
            "original_image": encode_image(original_img),
            "noisy_image": encode_image(noisy_img),
            "denoised_image": encode_image(denoised_img),
            "k": selected_k,
            "noise_std": noise_level,
            "energy_threshold": ENERGY_THRESHOLD,
            "channel_ranks": channel_ranks,
            "adaptive_ranks": adaptive_ranks,
            "width": int(original_img.shape[1]),
            "height": int(original_img.shape[0]),
            "logs": logger.logs,
        }

        if diff_img is not None:
            response_data["diff_image"] = encode_image(diff_img)

        return JSONResponse(response_data)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
