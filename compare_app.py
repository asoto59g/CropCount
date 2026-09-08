from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image


PROJECT_DIR = Path(__file__).parent
REFERENCE_IMAGE = PROJECT_DIR / "palma africana.png"
DJI_IMAGE = PROJECT_DIR / "DJI_11.jpg"
SAMPLES = 180

st.set_page_config(page_title="Comparador de firmas", page_icon="📐", layout="wide")


def load_rgb(source, fallback: Path) -> np.ndarray:
    image = Image.open(fallback if source is None else source)
    return np.asarray(image.convert("RGB"))


def segment_green(rgb: np.ndarray, hue_min: int, hue_max: int, saturation_min: int, value_min: int, close_size: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = ((hue >= hue_min) & (hue <= hue_max) & (saturation >= saturation_min) & (value >= value_min)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = [contour for contour in contours if cv2.contourArea(contour) >= 10]
    return max(valid, key=cv2.contourArea) if valid else None


def individual_palm_contours(mask: np.ndarray, ground_resolution_cm: float, minimum_diameter_m: float) -> list[np.ndarray]:
    """Separate touching crowns using local crown centers and watershed."""
    diameter_px = max(minimum_diameter_m * 100.0 / ground_resolution_cm, 8.0)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak_window = max(11, int(round(diameter_px * 0.45)))
    peak_kernel = np.ones((peak_window, peak_window), np.uint8)
    peaks = (distance == cv2.dilate(distance, peak_kernel)) & (distance >= diameter_px * 0.25)
    peak_count, peak_labels = cv2.connectedComponents(peaks.astype(np.uint8))
    if peak_count <= 1:
        contour = largest_contour(mask)
        return [contour] if contour is not None else []

    markers = peak_labels + 1
    sure_background = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=3)
    unknown = cv2.subtract(sure_background, (peaks.astype(np.uint8) * 255))
    markers[unknown > 0] = 0
    watershed_labels = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))

    minimum_area_px = max(50.0, np.pi * (diameter_px * 0.20) ** 2)
    maximum_area_px = np.pi * (diameter_px * 3.0) ** 2
    contours = []
    for label in range(2, int(watershed_labels.max()) + 1):
        region = (watershed_labels == label).astype(np.uint8) * 255
        region_contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not region_contours:
            continue
        contour = max(region_contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if minimum_area_px <= area <= maximum_area_px:
            contours.append(contour)
    return contours


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        points = contour[:, 0, :].astype(float)
        return float(points[:, 0].mean()), float(points[:, 1].mean())
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def radial_signature(mask: np.ndarray, center: tuple[float, float], samples: int = SAMPLES) -> np.ndarray:
    height, width = mask.shape
    center_x, center_y = center
    max_radius = float(np.hypot(max(center_x, width - center_x), max(center_y, height - center_y)))
    radii = np.linspace(0, max_radius, int(max_radius) + 1)
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    values = np.zeros(samples, dtype=float)
    for index, angle in enumerate(angles):
        x = np.rint(center_x + radii * np.cos(angle)).astype(int)
        y = np.rint(center_y + radii * np.sin(angle)).astype(int)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        occupied = np.zeros_like(valid)
        occupied[valid] = mask[y[valid], x[valid]] > 0
        positions = np.flatnonzero(occupied)
        if positions.size:
            values[index] = radii[positions[-1]]
    return values


def contour_signature(contour: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    points = contour[:, 0, :].astype(float)
    center = contour_centroid(contour)
    vectors = points - np.asarray(center)
    angles = (np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(angles / (2 * np.pi) * samples).astype(int) % samples
    radii = np.hypot(vectors[:, 0], vectors[:, 1])
    signature = np.zeros(samples, dtype=float)
    np.maximum.at(signature, bins, radii)
    missing = signature == 0
    if np.any(missing):
        known = np.flatnonzero(~missing)
        if known.size >= 2:
            signature[missing] = np.interp(np.flatnonzero(missing), known, signature[known], period=samples)
    return signature


def normalized_signature(rgb: np.ndarray, settings: tuple[int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = segment_green(rgb, *settings)
    contour = largest_contour(mask)
    if contour is None:
        raise ValueError("No se encontró una región vegetal. Ajusta los parámetros HSV.")
    center = contour_centroid(contour)
    signature = contour_signature(contour)
    scale = max(float(signature.max()), 1.0)
    return mask, contour, signature / scale


def region_signature(contour: np.ndarray) -> np.ndarray:
    signature = contour_signature(contour)
    return signature / max(float(signature.max()), 1.0)


def best_rotated_similarity(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, int, float, float]:
    best = (-1.0, 0, 1.0, 1.0)
    for shift in range(0, len(candidate), 4):
        rotated = np.roll(candidate, shift)
        error = float(np.mean(np.abs(reference - rotated)))
        correlation = float(np.corrcoef(reference, rotated)[0, 1])
        correlation = max(0.0, min(1.0, (correlation + 1.0) / 2.0))
        similarity = 0.6 * correlation + 0.4 * (1.0 - error)
        if similarity > best[0]:
            best = (similarity, shift, correlation, error)
    return best


def overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    contours: list[np.ndarray],
    color: tuple[int, int, int],
    labels: list[str] | None = None,
    recognized: list[np.ndarray] | None = None,
) -> np.ndarray:
    output = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    if labels:
        for index, contour in enumerate(contours):
            center = contour_centroid(contour)
            cv2.putText(output, labels[index], (round(center[0]), round(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for contour in recognized or []:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 4, cv2.LINE_AA)
    output[mask > 0] = (output[mask > 0] * 0.78 + np.array(color) * 0.22).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


st.title("Comparador de firmas radiales")
st.caption("Compara la extracción vectorizada de una imagen DJI contra la firma de referencia")

with st.sidebar:
    st.header("Imágenes")
    reference_upload = st.file_uploader("Referencia", type=["png", "jpg", "jpeg", "tif", "tiff"])
    dji_upload = st.file_uploader("Imagen DJI", type=["png", "jpg", "jpeg", "tif", "tiff"])
    st.header("Segmentación común")
    hue_min = st.slider("Tono mínimo (HSV)", 0, 179, 25)
    hue_max = st.slider("Tono máximo (HSV)", 0, 179, 100)
    saturation_min = st.slider("Saturación mínima", 0, 255, 45)
    value_min = st.slider("Valor mínimo", 0, 255, 35)
    close_size = st.slider("Cierre morfológico", 3, 31, 9, step=2)
    ground_resolution_cm = st.number_input("Resolución de terreno (cm/píxel)", min_value=0.1, max_value=100.0, value=6.0, step=0.1)
    minimum_diameter_m = st.number_input("Diámetro mínimo de copa (m)", min_value=0.5, max_value=10.0, value=4.0, step=0.5)
    certainty_limit = st.slider("Certeza mínima (%)", 0, 100, 60)
    compare_requested = st.button("Comparar firmas", type="primary", use_container_width=True)

reference_rgb = load_rgb(reference_upload, REFERENCE_IMAGE)
dji_rgb = load_rgb(dji_upload, DJI_IMAGE)

if not compare_requested:
    st.info("Carga las imágenes o usa los ejemplos y pulsa 'Comparar firmas'.")
    preview_1, preview_2 = st.columns(2)
    with preview_1:
        st.image(reference_rgb, caption="Referencia: palma africana", use_container_width=True)
    with preview_2:
        st.image(dji_rgb, caption="DJI_11", use_container_width=True)
    st.stop()

settings = (hue_min, hue_max, saturation_min, value_min, close_size)
try:
    reference_mask, reference_contour, reference_signature = normalized_signature(reference_rgb, settings)
    dji_mask = segment_green(dji_rgb, *settings)
    dji_contours = individual_palm_contours(dji_mask, ground_resolution_cm, minimum_diameter_m)
    if not dji_contours:
        raise ValueError("No se pudieron separar palmas individuales en DJI_11. Reduce el diámetro mínimo o ajusta HSV.")
    reference_mask, reference_contour, reference_signature = normalized_signature(reference_rgb, settings)
except ValueError as error:
    st.error(str(error))
    st.stop()

dji_results = []
for index, contour in enumerate(dji_contours, start=1):
    signature = region_signature(contour)
    similarity, shift, correlation, error = best_rotated_similarity(reference_signature, signature)
    dji_results.append({"id": index, "contour": contour, "signature": signature, "similarity": similarity, "shift": shift, "correlation": correlation, "error": error, "accepted": similarity * 100 >= certainty_limit})
dji_results.sort(key=lambda result: result["similarity"], reverse=True)
best_result = dji_results[0]
certainty = best_result["similarity"] * 100
accepted = best_result["accepted"]
aligned_dji = np.roll(best_result["signature"], best_result["shift"])
accepted_count = sum(result["accepted"] for result in dji_results)

if accepted:
    st.success(f"Coincidencia aceptada: {certainty:.1f}% de certeza (umbral {certainty_limit}%).")
else:
    st.warning(f"Coincidencia por debajo del umbral: {certainty:.1f}% de certeza (umbral {certainty_limit}%).")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Certeza", f"{certainty:.1f}%")
metric_2.metric("Umbral", f"{certainty_limit}%")
metric_3.metric("Palmas detectadas", f"{len(dji_results):,}")
metric_4.metric("Sobre el umbral", f"{accepted_count:,}")

image_1, image_2 = st.columns(2)
with image_1:
    st.image(overlay(reference_rgb, reference_mask, [reference_contour], (30, 150, 70)), caption="Referencia vectorizada", use_container_width=True)
with image_2:
    labels = [str(result["id"]) for result in dji_results]
    recognized_contours = [result["contour"] for result in dji_results if result["accepted"]]
    st.image(overlay(dji_rgb, dji_mask, [result["contour"] for result in dji_results], (230, 100, 20), labels, recognized_contours), caption="DJI_11: palmas vectorizadas; reconocidas en rojo", use_container_width=True)

st.subheader("Resultados por palma")
table = [
    {
        "palma": result["id"],
        "certeza_%": round(result["similarity"] * 100, 3),
        "correlacion_%": round(result["correlation"] * 100, 3),
        "error_medio_%": round(result["error"] * 100, 3),
        "aceptada": result["accepted"],
    }
    for result in dji_results
]
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Comparación de firmas")
angles = np.linspace(0, 360, SAMPLES, endpoint=False)
figure, axis = plt.subplots(figsize=(10, 3.5), facecolor="#f7f8f4")
axis.plot(angles, reference_signature, color="#1e8f52", linewidth=1.2, label="Referencia")
axis.plot(angles, aligned_dji, color="#d66522", linewidth=1.2, label=f'DJI_11 palma {best_result["id"]}')
axis.set_xlim(0, 360)
axis.set_ylim(0, 1.05)
axis.set_xlabel("Ángulo (grados)")
axis.set_ylabel("Radio normalizado")
axis.grid(alpha=0.25)
axis.legend()
figure.tight_layout()
st.pyplot(figure, clear_figure=True, use_container_width=True)

report = f"""resultado,valor
certeza_porcentaje,{certainty:.3f}
umbral_porcentaje,{certainty_limit}
aceptada,{accepted}
palmas_detectadas,{len(dji_results)}
palmas_sobre_umbral,{accepted_count}
mejor_correlacion_porcentaje,{best_result["correlation"] * 100:.3f}
mejor_error_medio_porcentaje,{best_result["error"] * 100:.3f}
rotacion_muestras,{best_result["shift"]}
""".encode("utf-8")
st.download_button("Descargar resultado CSV", report, "comparacion_dji_11.csv", "text/csv", use_container_width=True)
