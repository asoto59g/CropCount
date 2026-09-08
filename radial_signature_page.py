from __future__ import annotations

import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image


PROJECT_DIR = Path(__file__).parent
DEFAULT_IMAGE = PROJECT_DIR / "palma africana.png"

st.set_page_config(page_title="Radial Signature", page_icon="🌴", layout="wide")


def load_rgb(uploaded_file) -> np.ndarray:
    image = Image.open(DEFAULT_IMAGE if uploaded_file is None else uploaded_file)
    return np.asarray(image.convert("RGB"))


def segment_green(rgb: np.ndarray, hue_min: int, hue_max: int, saturation_min: int, value_min: int, close_size: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = ((hue >= hue_min) & (hue <= hue_max) & (saturation >= saturation_min) & (value >= value_min)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def vector_contours(mask: np.ndarray, minimum_area: float = 4.0) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]


def simplify_contour(contour: np.ndarray, minimum_percent: float, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    return cv2.approxPolyDP(contour, float(np.hypot(width, height)) * minimum_percent / 100.0, True)


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        points = contour[:, 0, :].astype(float)
        return float(points[:, 0].mean()), float(points[:, 1].mean())
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def radial_signature(mask: np.ndarray, center: tuple[float, float], samples: int = 720) -> tuple[np.ndarray, np.ndarray]:
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
    return np.degrees(angles), values


def overlay_contours(rgb: np.ndarray, contours: list[np.ndarray], simplified: list[np.ndarray], center: tuple[float, float]) -> np.ndarray:
    output = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    cv2.drawContours(output, contours, -1, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.drawContours(output, simplified, -1, (20, 40, 220), 1, cv2.LINE_AA)
    cv2.circle(output, (round(center[0]), round(center[1])), 5, (255, 0, 0), -1, cv2.LINE_AA)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def svg_from_contours(contours: list[np.ndarray], width: int, height: int) -> bytes:
    paths = []
    for contour in contours:
        points = contour[:, 0, :]
        path = "M " + " L ".join(f"{int(x)},{int(y)}" for x, y in points) + " Z"
        paths.append(f'  <path d="{path}" fill="none" stroke="#146b3a" stroke-width="1"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{chr(10).join(paths)}
</svg>
'''.encode("utf-8")


st.title("Radial Signature")
st.caption("Vectorización del contorno vegetal y extracción de firma radial")

with st.sidebar:
    st.header("Imagen")
    uploaded_file = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg", "tif", "tiff"])
    output_subdirectory = st.text_input("Subdirectorio de salida", value="outputs")
    st.caption("Se crea dentro del proyecto. En Streamlit Cloud no es almacenamiento permanente.")
    st.header("Segmentación")
    hue_min = st.slider("Tono mínimo (HSV)", 0, 179, 25)
    hue_max = st.slider("Tono máximo (HSV)", 0, 179, 100)
    saturation_min = st.slider("Saturación mínima", 0, 255, 45)
    value_min = st.slider("Valor mínimo", 0, 255, 35)
    close_size = st.slider("Cierre morfológico", 3, 31, 9, step=2)
    st.header("Vectorización")
    minimum_percent = st.slider("Longitud mínima / simplificación (%)", 0.0, 1.0, 1.0, step=0.1)
    st.caption("Escala disponible: 0% a 1% de la diagonal de la imagen.")
    extract_requested = st.button("Extraer firma", type="primary", use_container_width=True)

try:
    output_dir = (PROJECT_DIR / output_subdirectory).resolve()
    if PROJECT_DIR not in output_dir.parents and output_dir != PROJECT_DIR:
        raise ValueError("El subdirectorio debe estar dentro del proyecto.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(uploaded_file)
except Exception as error:
    st.error(f"No se pudo preparar la imagen o salida: {error}")
    st.stop()

if not extract_requested:
    st.info("Ajusta los parámetros y pulsa 'Extraer firma' para generar el contorno y la firma radial.")
    st.image(rgb, caption="Imagen de entrada", use_container_width=True)
    st.stop()

mask = segment_green(rgb, hue_min, hue_max, saturation_min, value_min, close_size)
contours = vector_contours(mask)
if not contours:
    st.error("No se encontró un contorno verde. Reduce la saturación mínima o amplía el rango de tono.")
    st.stop()

contour = max(contours, key=cv2.contourArea)
simplified_contours = [simplify_contour(item, minimum_percent, mask.shape) for item in contours]
center = contour_centroid(max(simplified_contours, key=cv2.contourArea))
angles, radii = radial_signature(mask, center)
overlay = overlay_contours(rgb, contours, simplified_contours, center)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Resolución", f"{rgb.shape[1]} x {rgb.shape[0]}")
metric_2.metric("Área segmentada", f"{cv2.contourArea(contour):,.0f} px²")
metric_3.metric("Contornos vectoriales", f"{len(simplified_contours):,}")
metric_4.metric("Vértices vectoriales", f"{sum(len(item) for item in simplified_contours):,}")

left, middle, right = st.columns(3)
with left:
    st.image(rgb, caption="Entrada", use_container_width=True)
with middle:
    st.image(mask, caption="Máscara segmentada", use_container_width=True)
with right:
    st.image(overlay, caption="Contorno original + vector simplificado", use_container_width=True)

st.subheader("Firma radial")
figure, axis = plt.subplots(figsize=(10, 3.4), facecolor="#f7f8f4")
axis.plot(angles, radii, color="#146b3a", linewidth=1.25)
axis.fill_between(angles, radii, color="#9bcf7a", alpha=0.25)
axis.set_xlim(0, 360)
axis.set_xlabel("Ángulo (grados)")
axis.set_ylabel("Radio (px)")
axis.grid(alpha=0.25)
figure.tight_layout()
st.pyplot(figure, clear_figure=True, use_container_width=True)

normalizer = max(float(np.hypot(rgb.shape[1], rgb.shape[0])), 1.0)
csv_data = ("angle_deg,radius_px,radius_normalized\n" + "\n".join(f"{angle:.3f},{radius:.3f},{radius / normalizer:.6f}" for angle, radius in zip(angles, radii))).encode("utf-8")
svg_data = svg_from_contours(simplified_contours, rgb.shape[1], rgb.shape[0])
(output_dir / "contour.svg").write_bytes(svg_data)
(output_dir / "radial_signature.csv").write_bytes(csv_data)

st.subheader("Exportar resultados")
export_1, export_2 = st.columns(2)
with export_1:
    st.download_button("Descargar contorno SVG", svg_data, "contorno.svg", "image/svg+xml", use_container_width=True)
with export_2:
    st.download_button("Descargar firma CSV", csv_data, "radial_signature.csv", "text/csv", use_container_width=True)
st.caption(f"Archivos guardados en: {output_dir.relative_to(PROJECT_DIR)}")
