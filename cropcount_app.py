from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image


PROJECT_DIR = Path(__file__).parent
SAMPLES = 180
PROFILES = {
    "Oil palm": {"slug": "oil_palm", "diameter_m": 4.0},
    "Banana": {"slug": "banana", "diameter_m": 2.0},
    "Pineapple": {"slug": "pineapple", "diameter_m": 1.0},
    "Aloe vera": {"slug": "aloe_vera", "diameter_m": 0.5},
    "Custom crop": {"slug": "custom", "diameter_m": 1.0},
}

st.set_page_config(page_title="CropCount", page_icon="🌱", layout="wide")


def load_rgb(source, fallback: Path | None = None) -> np.ndarray:
    if source is None and fallback is None:
        raise ValueError("Select an image or provide a default image.")
    image = Image.open(fallback if source is None else source)
    return np.asarray(image.convert("RGB"))


def segment_crop(rgb: np.ndarray, hue_min: int, hue_max: int, saturation_min: int, value_min: int, close_size: int) -> np.ndarray:
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


def individual_contours(mask: np.ndarray, resolution_cm: float, minimum_diameter_m: float) -> list[np.ndarray]:
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


def centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        points = contour[:, 0, :].astype(float)
        return float(points[:, 0].mean()), float(points[:, 1].mean())
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def signature(contour: np.ndarray) -> np.ndarray:
    points = contour[:, 0, :].astype(float)
    vectors = points - np.asarray(centroid(contour))
    angles = (np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(angles / (2 * np.pi) * SAMPLES).astype(int) % SAMPLES
    radii = np.hypot(vectors[:, 0], vectors[:, 1])
    result = np.zeros(SAMPLES, dtype=np.float32)
    np.maximum.at(result, bins, radii)
    missing = result == 0
    known = np.flatnonzero(~missing)
    if np.any(missing) and known.size >= 2:
        result[missing] = np.interp(np.flatnonzero(missing), known, result[known], period=SAMPLES)
    return result / max(float(result.max()), 1.0)


def compare_signatures(prototypes: np.ndarray, candidate: np.ndarray) -> float:
    shifts = np.arange(0, SAMPLES, 4)
    rotated = np.stack([np.roll(candidate, int(shift)) for shift in shifts])
    errors = np.mean(np.abs(prototypes[:, None, :] - rotated[None, :, :]), axis=2)
    left = prototypes - prototypes.mean(axis=1, keepdims=True)
    right = rotated - rotated.mean(axis=1, keepdims=True)
    correlations = (left @ right.T) / np.maximum(np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(right, axis=1)[None, :], 1e-8)
    correlations = np.clip((correlations + 1.0) / 2.0, 0.0, 1.0)
    return float((0.6 * correlations + 0.4 * (1.0 - errors)).max())


def model_bytes(signatures: np.ndarray, crop_name: str, resolution_cm: float, diameter_m: float, settings: tuple[int, int, int, int, int]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, signatures=signatures.astype(np.float32), crop=np.asarray(crop_name), method=np.asarray("radial_signature_hsv_watershed"), resolution_cm=np.float32(resolution_cm), minimum_diameter_m=np.float32(diameter_m), hsv_settings=np.asarray(settings, dtype=np.int32), samples=np.int32(SAMPLES), created_at=np.asarray(datetime.now(timezone.utc).isoformat()))
    return buffer.getvalue()


def read_model(source) -> dict[str, np.ndarray]:
    with np.load(source, allow_pickle=False) as model:
        return {key: model[key] for key in model.files}


def draw_results(rgb: np.ndarray, contours: list[np.ndarray], recognized: list[bool], excluded: set[int] | None = None, selected: int | None = None) -> np.ndarray:
    output = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    for index, contour in enumerate(contours):
        if index + 1 in (excluded or set()):
            continue
        color = (0, 255, 255) if index + 1 == selected else ((0, 0, 255) if recognized[index] else (0, 165, 255))
        x, y, width, height = cv2.boundingRect(contour)
        thickness = 6 if index + 1 == selected else 4
        cv2.rectangle(output, (x, y), (x + width, y + height), color, thickness, cv2.LINE_AA)
        cv2.putText(output, str(index + 1), (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


st.title("CropCount")
st.caption("Count individual plants from aerial imagery using reusable radial signatures")
st.info("Detection engine: Radial Signature + HSV segmentation + watershed. This version does not use YOLO or a neural network.")

with st.sidebar:
    st.header("Crop profile")
    crop_name = st.selectbox("Crop type", list(PROFILES))
    profile = PROFILES[crop_name]
    output_subdirectory = st.text_input("Output subdirectory", value="outputs")
    st.caption("Created inside the project. Streamlit Cloud storage is temporary; download the model for persistence.")
    output_dir = (PROJECT_DIR / output_subdirectory).resolve()
    if PROJECT_DIR not in output_dir.parents and output_dir != PROJECT_DIR:
        st.error("Output subdirectory must stay inside the project.")
        st.stop()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"cropcount_{profile['slug']}.npz"
    resolution_cm = st.number_input("Ground resolution (cm/pixel)", 0.1, 100.0, 6.0, 0.1)
    minimum_diameter_m = st.number_input("Minimum plant diameter (m)", 0.1, 20.0, float(profile["diameter_m"]), 0.1)
    certainty_limit = st.slider("Recognition threshold (%)", 0, 100, 60)
    st.header("Segmentation")
    hue_min = st.slider("Hue min (HSV)", 0, 179, 25)
    hue_max = st.slider("Hue max (HSV)", 0, 179, 100)
    saturation_min = st.slider("Minimum saturation", 0, 255, 45)
    value_min = st.slider("Minimum value", 0, 255, 35)
    close_size = st.slider("Morphological closing", 3, 31, 9, step=2)
    build_requested = st.button("Build / update crop model", type="primary", use_container_width=True)
    detect_requested = st.button("Count plants in new image", use_container_width=True)

reference_upload = st.file_uploader("Reference plant image (optional)", type=["png", "jpg", "jpeg", "tif", "tiff"])
training_upload = st.file_uploader("Training plantation image (optional)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="training")
future_upload = st.file_uploader("New plantation image (optional)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="future")
model_upload = st.file_uploader("Existing crop model (.npz, optional)", type=["npz"], key="model")
settings = (hue_min, hue_max, saturation_min, value_min, close_size)

if build_requested:
    if reference_upload is None or training_upload is None:
        st.error("Select a reference plant image and a training plantation image.")
        st.stop()
    reference_rgb = load_rgb(reference_upload)
    training_rgb = load_rgb(training_upload)
    reference_contour = largest_contour(segment_crop(reference_rgb, *settings))
    training_contours = individual_contours(segment_crop(training_rgb, *settings), resolution_cm, minimum_diameter_m)
    if reference_contour is None or not training_contours:
        st.error("No plants were found. Adjust HSV values or the minimum diameter.")
        st.stop()
    signatures = np.vstack([signature(reference_contour), *[signature(contour) for contour in training_contours]])
    data = model_bytes(signatures, crop_name, resolution_cm, minimum_diameter_m, settings)
    model_path.write_bytes(data)
    st.success(f"{crop_name} model saved with {len(signatures)} plant signatures: {model_path.name}")
    st.download_button("Download crop model", data, model_path.name, "application/octet-stream")
    st.image(draw_results(training_rgb, training_contours, [False] * len(training_contours)), caption="Training plants included in the model", use_container_width=True)

if detect_requested:
    model = read_model(model_upload) if model_upload is not None else (read_model(model_path) if model_path.exists() else None)
    if model is None or future_upload is None:
        st.error("Load a crop model and a new plantation image first.")
        st.stop()
    future_rgb = load_rgb(future_upload)
    future_contours = individual_contours(segment_crop(future_rgb, *settings), resolution_cm, minimum_diameter_m)
    prototypes = model["signatures"]
    scores = [compare_signatures(prototypes, signature(contour)) for contour in future_contours]
    st.session_state["crop_detection"] = {
        "crop_name": crop_name,
        "rgb": future_rgb,
        "contours": future_contours,
        "scores": scores,
        "certainty_limit": certainty_limit,
        "output_dir": output_dir,
        "slug": profile["slug"],
    }

if "crop_detection" in st.session_state:
    detection = st.session_state["crop_detection"]
    contours = detection["contours"]
    scores = detection["scores"]
    threshold = detection["certainty_limit"]
    candidate_ids = [index for index, score in enumerate(scores, start=1) if score * 100 >= threshold]
    st.subheader("Edit detections before export")
    st.caption("First click inside a red or orange rectangle. The selected plant turns yellow, then use the button below to exclude it.")
    clicked = streamlit_image_coordinates(
        draw_results(detection["rgb"], contours, [score * 100 >= threshold for score in scores], set(st.session_state.get("crop_excluded_ids", [])), st.session_state.get("crop_selected_id")),
        key="crop_detection_image",
    )
    if clicked:
        click_x, click_y = float(clicked["x"]), float(clicked["y"])
        containing = []
        for index, contour in enumerate(contours, start=1):
            x, y, width, height = cv2.boundingRect(contour)
            if x <= click_x <= x + width and y <= click_y <= y + height:
                containing.append((cv2.pointPolygonTest(contour, (click_x, click_y), True), index))
        if containing:
            st.session_state["crop_selected_id"] = max(containing)[1]

    selected_id = st.session_state.get("crop_selected_id")
    action_1, action_2 = st.columns(2)
    with action_1:
        exclude_clicked = st.button("Exclude selected plant", type="primary", disabled=selected_id is None, use_container_width=True)
    with action_2:
        clear_clicked = st.button("Clear selection", disabled=selected_id is None, use_container_width=True)

    if exclude_clicked and selected_id is not None:
        excluded = set(st.session_state.get("crop_excluded_ids", []))
        excluded.add(selected_id)
        st.session_state["crop_excluded_ids"] = sorted(excluded)
        st.session_state["crop_selected_id"] = None
        st.rerun()
    if clear_clicked:
        st.session_state["crop_selected_id"] = None
        st.rerun()

    if selected_id:
        selected_score = scores[selected_id - 1] * 100
        st.info(f"Planta seleccionada: {selected_id} ({selected_score:.1f}%). El rectángulo amarillo indica la selección.")

    excluded_ids = st.multiselect(
        "Excluir falsos positivos",
        options=candidate_ids,
        default=st.session_state.get("crop_excluded_ids", []),
        key="crop_excluded_selector",
        format_func=lambda plant_id: f"Plant {plant_id} ({scores[plant_id - 1] * 100:.1f}%)",
        help="Selecciona las detecciones incorrectas antes de descargar el conteo.",
    )
    if st.button("Restaurar todas las detecciones", use_container_width=True):
        st.session_state["crop_excluded_ids"] = []
        st.session_state["crop_excluded_selector"] = []
        st.rerun()
    st.session_state["crop_excluded_ids"] = excluded_ids
    recognized = [score * 100 >= threshold and index not in excluded_ids for index, score in enumerate(scores, start=1)]
    st.success(f"{detection['crop_name']}: {sum(recognized)} recognized plants out of {len(contours)} detected.")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Plants detected", len(contours))
    metric_2.metric("Plants recognized", sum(recognized))
    metric_3.metric("Excluded false positives", len(excluded_ids))
    st.image(draw_results(detection["rgb"], contours, recognized, set(excluded_ids), selected_id), caption="Rojo: aceptada; naranja: candidata; amarillo: seleccionada; excluidas: ocultas", use_container_width=True)
    report = "plant,confidence_percent,recognized,excluded\n" + "\n".join(f"{index},{score * 100:.3f},{accepted},{index in excluded_ids}" for index, (score, accepted) in enumerate(zip(scores, recognized), start=1))
    (detection["output_dir"] / f"{detection['slug']}_plant_count.csv").write_text(report, encoding="utf-8")
    st.download_button("Download plant count CSV", report.encode("utf-8"), f"{detection['slug']}_plant_count.csv", "text/csv")

if not build_requested and not detect_requested:
    st.info("Choose a crop profile, build its model from representative images, then count plants in a new plantation.")
    if model_path.exists():
        with np.load(model_path, allow_pickle=False) as existing:
            st.caption(f"Local model available for {crop_name}: {len(existing['signatures'])} signatures.")
