from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image


PROJECT_DIR = Path(__file__).parent
REFERENCE_IMAGE = PROJECT_DIR / "palma africana.png"
PLANTATION_IMAGE = PROJECT_DIR / "DJI_11.jpg"
MODEL_FILE = PROJECT_DIR / "palma_radial_model.npz"
SAMPLES = 180

st.set_page_config(page_title="Modelo de palma africana", page_icon="🌴", layout="wide")


def load_rgb(source, fallback: Path) -> np.ndarray:
    image = Image.open(fallback if source is None else source)
    return np.asarray(image.convert("RGB"))


def segment_green(rgb: np.ndarray, hue_min: int, hue_max: int, saturation_min: int, value_min: int, close_size: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = ((hue >= hue_min) & (hue <= hue_max) & (saturation >= saturation_min) & (value >= value_min)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = [contour for contour in contours if cv2.contourArea(contour) >= 10]
    return max(valid, key=cv2.contourArea) if valid else None


def individual_palm_contours(mask: np.ndarray, resolution_cm: float, minimum_diameter_m: float) -> list[np.ndarray]:
    diameter_px = max(minimum_diameter_m * 100.0 / resolution_cm, 8.0)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    window = max(11, int(round(diameter_px * 0.45)))
    peaks = (distance == cv2.dilate(distance, np.ones((window, window), np.uint8))) & (distance >= diameter_px * 0.25)
    count, labels = cv2.connectedComponents(peaks.astype(np.uint8))
    if count <= 1:
        contour = largest_contour(mask)
        return [contour] if contour is not None else []

    markers = labels + 1
    background = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=3)
    unknown = cv2.subtract(background, peaks.astype(np.uint8) * 255)
    markers[unknown > 0] = 0
    watershed_labels = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
    minimum_area = max(50.0, np.pi * (diameter_px * 0.20) ** 2)
    maximum_area = np.pi * (diameter_px * 3.0) ** 2
    result = []
    for label in range(2, int(watershed_labels.max()) + 1):
        region = (watershed_labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if minimum_area <= area <= maximum_area:
            result.append(contour)
    return result


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        points = contour[:, 0, :].astype(float)
        return float(points[:, 0].mean()), float(points[:, 1].mean())
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def contour_signature(contour: np.ndarray) -> np.ndarray:
    points = contour[:, 0, :].astype(float)
    center = np.asarray(contour_centroid(contour))
    vectors = points - center
    angles = (np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(angles / (2 * np.pi) * SAMPLES).astype(int) % SAMPLES
    radii = np.hypot(vectors[:, 0], vectors[:, 1])
    signature = np.zeros(SAMPLES, dtype=np.float32)
    np.maximum.at(signature, bins, radii)
    missing = signature == 0
    known = np.flatnonzero(~missing)
    if np.any(missing) and known.size >= 2:
        signature[missing] = np.interp(np.flatnonzero(missing), known, signature[known], period=SAMPLES)
    return signature / max(float(signature.max()), 1.0)


def compare_signature(prototypes: np.ndarray, candidate: np.ndarray) -> float:
    shifts = np.arange(0, SAMPLES, 4)
    rotated = np.stack([np.roll(candidate, int(shift)) for shift in shifts])
    errors = np.mean(np.abs(prototypes[:, None, :] - rotated[None, :, :]), axis=2)
    centered_prototypes = prototypes - prototypes.mean(axis=1, keepdims=True)
    centered_rotated = rotated - rotated.mean(axis=1, keepdims=True)
    prototype_norms = np.linalg.norm(centered_prototypes, axis=1, keepdims=True)
    rotated_norms = np.linalg.norm(centered_rotated, axis=1, keepdims=True)
    correlations = (centered_prototypes @ centered_rotated.T) / np.maximum(prototype_norms * rotated_norms.T, 1e-8)
    correlations = np.clip((correlations + 1.0) / 2.0, 0.0, 1.0)
    scores = 0.6 * correlations + 0.4 * (1.0 - errors)
    return float(scores.max())


def model_bytes(signatures: np.ndarray, resolution_cm: float, minimum_diameter_m: float, settings: tuple[int, int, int, int, int]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        signatures=signatures.astype(np.float32),
        resolution_cm=np.float32(resolution_cm),
        minimum_diameter_m=np.float32(minimum_diameter_m),
        hsv_settings=np.asarray(settings, dtype=np.int32),
        samples=np.int32(SAMPLES),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    return buffer.getvalue()


def read_model(source) -> dict[str, np.ndarray]:
    with np.load(source, allow_pickle=False) as model:
        return {key: model[key] for key in model.files}


def draw_results(rgb: np.ndarray, contours: list[np.ndarray], recognized: list[bool]) -> np.ndarray:
    output = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    for index, contour in enumerate(contours, start=1):
        color = (0, 0, 255) if recognized[index - 1] else (0, 165, 255)
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 4, cv2.LINE_AA)
        cv2.putText(output, str(index), (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


st.title("Modelo de detección de palma africana")
st.caption("Construye y aplica un modelo basado en firmas radiales para contar palmas en nuevas plantaciones")

with st.sidebar:
    st.header("Calibración")
    resolution_cm = st.number_input("Resolución de terreno (cm/píxel)", min_value=0.1, max_value=100.0, value=6.0, step=0.1)
    minimum_diameter_m = st.number_input("Diámetro mínimo de copa (m)", min_value=0.5, max_value=10.0, value=4.0, step=0.5)
    hue_min = st.slider("Tono mínimo (HSV)", 0, 179, 25)
    hue_max = st.slider("Tono máximo (HSV)", 0, 179, 100)
    saturation_min = st.slider("Saturación mínima", 0, 255, 45)
    value_min = st.slider("Valor mínimo", 0, 255, 35)
    close_size = st.slider("Cierre morfológico", 3, 31, 9, step=2)
    certainty_limit = st.slider("Certeza mínima (%)", 0, 100, 60)
    build_requested = st.button("Crear / actualizar modelo", type="primary", use_container_width=True)
    detect_requested = st.button("Detectar en nueva plantación", use_container_width=True)

settings = (hue_min, hue_max, saturation_min, value_min, close_size)
reference_upload = st.file_uploader("Firma de referencia individual (opcional)", type=["png", "jpg", "jpeg", "tif", "tiff"])
training_upload = st.file_uploader("Plantación para ampliar el modelo (opcional)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="training")
future_upload = st.file_uploader("Nueva plantación para contar (opcional)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="future")
model_upload = st.file_uploader("Modelo NPZ existente (opcional)", type=["npz"], key="model")

if build_requested:
    reference_rgb = load_rgb(reference_upload, REFERENCE_IMAGE)
    training_rgb = load_rgb(training_upload, PLANTATION_IMAGE)
    reference_mask = segment_green(reference_rgb, *settings)
    training_mask = segment_green(training_rgb, *settings)
    reference_contour = largest_contour(reference_mask)
    training_contours = individual_palm_contours(training_mask, resolution_cm, minimum_diameter_m)
    if reference_contour is None or not training_contours:
        st.error("No se encontraron firmas suficientes. Ajusta HSV o el diámetro mínimo de copa.")
        st.stop()
    signatures = np.vstack([contour_signature(reference_contour), *[contour_signature(contour) for contour in training_contours]])
    data = model_bytes(signatures, resolution_cm, minimum_diameter_m, settings)
    MODEL_FILE.write_bytes(data)
    st.success(f"Modelo guardado: {len(signatures)} firmas en {MODEL_FILE.name}.")
    st.download_button("Descargar modelo NPZ", data, MODEL_FILE.name, "application/octet-stream")
    st.image(draw_results(training_rgb, training_contours, [False] * len(training_contours)), caption="Firmas incorporadas al modelo", use_container_width=True)

if detect_requested:
    if future_upload is None and not MODEL_FILE.exists():
        st.error("Carga una nueva plantación o crea primero un modelo.")
        st.stop()
    model = read_model(model_upload) if model_upload is not None else (read_model(MODEL_FILE) if MODEL_FILE.exists() else None)
    if model is None:
        st.error("El modelo no está disponible.")
        st.stop()
    future_rgb = load_rgb(future_upload, PLANTATION_IMAGE)
    future_mask = segment_green(future_rgb, *settings)
    future_contours = individual_palm_contours(future_mask, resolution_cm, minimum_diameter_m)
    if not future_contours:
        st.warning("No se detectaron palmas en la nueva plantación.")
        st.stop()
    prototypes = model["signatures"]
    scores = [compare_signature(prototypes, contour_signature(contour)) for contour in future_contours]
    recognized = [score * 100 >= certainty_limit for score in scores]
    st.success(f"Palmas reconocidas: {sum(recognized)} de {len(future_contours)} detectadas.")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Palmas detectadas", len(future_contours))
    metric_2.metric("Palmas reconocidas", sum(recognized))
    metric_3.metric("Umbral", f"{certainty_limit}%")
    st.image(draw_results(future_rgb, future_contours, recognized), caption="Rojo: reconocida; naranja: candidata", use_container_width=True)
    scores_csv = "palma,certeza_porcentaje,reconocida\n" + "\n".join(f"{index},{score * 100:.3f},{accepted}" for index, (score, accepted) in enumerate(zip(scores, recognized), start=1))
    st.download_button("Descargar conteo CSV", scores_csv.encode("utf-8"), "conteo_palmas.csv", "text/csv")

if not build_requested and not detect_requested:
    st.info("Crea el modelo con la referencia y DJI_11, o carga un modelo NPZ existente desde el código de la app y pulsa detectar.")
    if MODEL_FILE.exists():
        with np.load(MODEL_FILE, allow_pickle=False) as existing:
            st.caption(f"Modelo local disponible: {len(existing['signatures'])} firmas.")
