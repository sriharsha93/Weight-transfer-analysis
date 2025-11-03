import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import tempfile
from pathlib import Path
import imageio
import urllib.request

# =============================
# UI defaults
# =============================
DEFAULT_SPEED = 0.5         # output playback speed (0.1..2.0)
DEFAULT_ALPHA = 0.25        # EMA smoothing (0..1)
DEFAULT_ROLL_DEG = 0.0      # camera roll correction

# =============================
# Colors
# =============================
COL_LEFT  = (255, 180, 60)    # left-side bones
COL_RIGHT = (180, 100, 255)   # right-side bones
COL_MID   = (230, 230, 230)   # mid connections
COL_TRUNK = (0, 255, 0)       # trunk line
COL_TEXT1 = (0, 255, 0)
COL_BACK  = (80, 80, 255)     # back (red-ish)
COL_FRONT = (60, 200, 120)    # front (green-ish)
COL_PANEL_BG = (30, 30, 30)
COL_PANEL_BORDER = (180, 180, 180)

# =============================
# Landmark indices (MediaPipe Pose 33-keypoint layout)
# =============================
POSE_IDX = {
    "NOSE":0, "LEFT_EYE_INNER":1, "LEFT_EYE":2, "LEFT_EYE_OUTER":3,
    "RIGHT_EYE_INNER":4, "RIGHT_EYE":5, "RIGHT_EYE_OUTER":6,
    "LEFT_EAR":7, "RIGHT_EAR":8, "MOUTH_LEFT":9, "MOUTH_RIGHT":10,
    "LEFT_SHOULDER":11, "RIGHT_SHOULDER":12,
    "LEFT_ELBOW":13, "RIGHT_ELBOW":14,
    "LEFT_WRIST":15, "RIGHT_WRIST":16,
    "LEFT_PINKY":17, "RIGHT_PINKY":18,
    "LEFT_INDEX":19, "RIGHT_INDEX":20,
    "LEFT_THUMB":21, "RIGHT_THUMB":22,
    "LEFT_HIP":23, "RIGHT_HIP":24,
    "LEFT_KNEE":25, "RIGHT_KNEE":26,
    "LEFT_ANKLE":27, "RIGHT_ANKLE":28,
    "LEFT_HEEL":29, "RIGHT_HEEL":30,
    "LEFT_FOOT_INDEX":31, "RIGHT_FOOT_INDEX":32
}

# =============================
# Helpers
# =============================
def ema_filter(alpha=0.2):
    y = None
    def f(x):
        nonlocal y
        if x is None:
            return y
        if y is None:
            y = x
        else:
            y += alpha * (x - y)
        return y
    return f

def rotate2d(vec, deg):
    if deg == 0.0:
        return vec
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    R = np.array([[c, -s], [s,  c]], dtype=np.float32)
    return R @ vec

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def np_avg(p1, p2):
    return (p1 + p2) * 0.5

def to_px(img, xy):
    h, w = img.shape[:2]
    x = int(clamp(xy[0], 0, 1) * w)
    y = int(clamp(xy[1], 0, 1) * h)
    return (x, y)

def draw_joint(img, xy, color, r=5, thick=-1):
    cv2.circle(img, to_px(img, xy), r, color, thick)

def draw_text(img, text, org, color=(255,255,255), scale=0.7, thickness=2, bg=False):
    if bg:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x-4, y-th-6), (x+tw+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_bone(img, p, q, color, thickness=2):
    cv2.line(img, to_px(img, p), to_px(img, q), color, thickness, cv2.LINE_AA)

# =============================
# Metrics
# =============================
def trunk_lean_deg(mid_hip_xy, mid_sh_xy, camera_roll_deg=0.0, forward_sign=+1):
    v = (mid_sh_xy - mid_hip_xy).astype(np.float32)
    v = rotate2d(v, -camera_roll_deg)
    angle_rad = math.atan2(float(v[0]), float(-v[1]) + 1e-9)
    angle_deg = math.degrees(angle_rad)
    return forward_sign * angle_deg

def weight_transfer_index(pelvis_x, left_ankle_x, right_ankle_x, front_on_right=True):
    aL, aR = left_ankle_x, right_ankle_x
    if front_on_right:
        front = max(aL, aR)
        back  = min(aL, aR)
    else:
        front = min(aL, aR)
        back  = max(aL, aR)
    denom = (front - back)
    if abs(denom) < 1e-4:
        return None
    w = (pelvis_x - back) / denom
    return clamp(w, 0.0, 1.0)

# =============================
# Drawing: skeleton and weight panel
# =============================
def draw_skeleton(img, lm_list, vis_thr=0.45, thick=3):
    def get_xyv(idx):
        lm = lm_list[idx]
        v = getattr(lm, "visibility", 1.0)
        return np.array([lm.x, lm.y]), v

    def bone(a_name, b_name, color):
        a, va = get_xyv(POSE_IDX[a_name])
        b, vb = get_xyv(POSE_IDX[b_name])
        if va >= vis_thr and vb >= vis_thr:
            draw_bone(img, a, b, color, thick)

    # Left chain
    bone("LEFT_SHOULDER","LEFT_ELBOW", COL_LEFT)
    bone("LEFT_ELBOW",   "LEFT_WRIST", COL_LEFT)
    bone("LEFT_HIP",     "LEFT_KNEE",  COL_LEFT)
    bone("LEFT_KNEE",    "LEFT_ANKLE", COL_LEFT)
    # Right chain
    bone("RIGHT_SHOULDER","RIGHT_ELBOW", COL_RIGHT)
    bone("RIGHT_ELBOW",   "RIGHT_WRIST", COL_RIGHT)
    bone("RIGHT_HIP",     "RIGHT_KNEE",  COL_RIGHT)
    bone("RIGHT_KNEE",    "RIGHT_ANKLE", COL_RIGHT)
    # Torso / mid
    bone("LEFT_SHOULDER",  "RIGHT_SHOULDER", COL_MID)
    bone("LEFT_HIP",       "RIGHT_HIP",      COL_MID)
    bone("LEFT_SHOULDER",  "LEFT_HIP",       COL_MID)
    bone("RIGHT_SHOULDER", "RIGHT_HIP",      COL_MID)

def draw_weight_panel(img, wti, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), COL_PANEL_BG, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), COL_PANEL_BORDER, 1)

    pad = 8
    draw_text(img, "BACK",  (x+pad, y+18), COL_BACK,  scale=0.6, thickness=2, bg=False)
    draw_text(img, "FRONT", (x+w-60, y+18), COL_FRONT, scale=0.6, thickness=2, bg=False)

    bar_x = x + pad
    bar_w = w - 2*pad
    bar_y = y + 24
    bar_h = max(12, h - 32)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80,80,80), 1)

    if wti is None:
        draw_text(img, "WTI: --", (bar_x + bar_w//2 - 30, bar_y + bar_h - 3), (200,200,200), 0.6, 2, False)
        return

    back_frac  = 1.0 - clamp(wti, 0.0, 1.0)
    front_frac = clamp(wti, 0.0, 1.0)
    mid_x = bar_x + int(back_frac * bar_w)

    if back_frac > 0:
        cv2.rectangle(img, (bar_x, bar_y), (mid_x, bar_y+bar_h), COL_BACK, -1)
    if front_frac > 0:
        cv2.rectangle(img, (mid_x, bar_y), (bar_x+bar_w, bar_y+bar_h), COL_FRONT, -1)

    cv2.line(img, (mid_x, bar_y), (mid_x, bar_y+bar_h), (30,30,30), 1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80,80,80), 1)

    draw_text(img, f"{back_frac*100:3.0f}%", (bar_x+4, bar_y+bar_h-3), (255,255,255), 0.6, 2, False)
    draw_text(img, f"{front_frac*100:3.0f}%", (bar_x+bar_w-44, bar_y+bar_h-3), (255,255,255), 0.6, 2, False)

# =============================
# MediaPipe Tasks: Pose Landmarker
# =============================
MODEL_URLS = {
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}

def get_model_path(variant="heavy") -> Path:
    url = MODEL_URLS[variant]
    models_dir = Path(tempfile.gettempdir()) / "mp_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    local_path = models_dir / Path(url).name
    if not local_path.exists() or local_path.stat().st_size < 1024:
        urllib.request.urlretrieve(url, local_path)  # download to writable temp
    return local_path

class PoseLandmarkerWrapper:
    def __init__(self, model_path: Path, fps_hint: float):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        self.running_mode = vision.RunningMode.VIDEO
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.dt_ms = 1000.0 / (fps_hint if fps_hint and fps_hint > 1e-3 else 30.0)

    def process(self, frame_bgr):
        from mediapipe import Image as MPImage
        from mediapipe import ImageFormat
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, int(self.timestamp_ms))
        self.timestamp_ms += self.dt_ms
        # Return first pose landmarks (33 points) or None
        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]
        return None

    def close(self):
        if self.landmarker:
            self.landmarker.close()

# =============================
# Core processing (uses imageio to write MP4)
# =============================
def process_video_to_temp(in_path: Path, speed=DEFAULT_SPEED, batter_hand="Right-hand batter",
                          roll_deg=DEFAULT_ROLL_DEG, alpha=DEFAULT_ALPHA):
    front_on_right = (batter_hand == "Right-hand batter")
    forward_sign = +1 if front_on_right else -1

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-3 else 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1.0, fps * clamp(speed, 0.1, 2.0))

    # Temp output file
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_out_path = Path(tmp_out.name)
    tmp_out.close()

    writer = imageio.get_writer(
        str(tmp_out_path),
        format="FFMPEG",
        mode="I",
        fps=out_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None
    )

    # Smoothing
    lean_smooth = ema_filter(alpha=alpha)
    wti_smooth  = ema_filter(alpha=alpha)

    # MediaPipe Tasks Pose Landmarker (model in /tmp, avoids permission issues)
    model_path = get_model_path("heavy")
    landmarker = PoseLandmarkerWrapper(model_path, fps_hint=fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    prog = st.progress(0)
    status = st.empty()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            lm_list = landmarker.process(frame)
            lean_deg_val = None
            wti_val = None

            if lm_list is not None and len(lm_list) == 33:
                # Gather keypoints
                L_SH = np.array([lm_list[POSE_IDX["LEFT_SHOULDER"]].x,  lm_list[POSE_IDX["LEFT_SHOULDER"]].y])
                R_SH = np.array([lm_list[POSE_IDX["RIGHT_SHOULDER"]].x, lm_list[POSE_IDX["RIGHT_SHOULDER"]].y])
                L_HP = np.array([lm_list[POSE_IDX["LEFT_HIP"]].x,       lm_list[POSE_IDX["LEFT_HIP"]].y])
                R_HP = np.array([lm_list[POSE_IDX["RIGHT_HIP"]].x,      lm_list[POSE_IDX["RIGHT_HIP"]].y])
                L_AN = np.array([lm_list[POSE_IDX["LEFT_ANKLE"]].x,     lm_list[POSE_IDX["LEFT_ANKLE"]].y])
                R_AN = np.array([lm_list[POSE_IDX["RIGHT_ANKLE"]].x,    lm_list[POSE_IDX["RIGHT_ANKLE"]].y])

                mid_sh = np_avg(L_SH, R_SH)
                mid_hp = np_avg(L_HP, R_HP)

                # Metrics
                lean_deg_val = trunk_lean_deg(mid_hp, mid_sh, camera_roll_deg=roll_deg, forward_sign=forward_sign)
                pelvis_x = float(mid_hp[0])
                wti_val = weight_transfer_index(pelvis_x, float(L_AN[0]), float(R_AN[0]), front_on_right=front_on_right)

                # Skeleton + trunk line
                draw_skeleton(frame, lm_list, vis_thr=0.45, thick=3)
                cv2.line(frame, to_px(frame, mid_hp), to_px(frame, mid_sh), COL_TRUNK, 3)

                # Emphasize key joints
                for name, color in [
                    ("LEFT_KNEE",  (255,200,0)), ("RIGHT_KNEE", (255,200,0)),
                    ("LEFT_ELBOW", (0,255,255)), ("RIGHT_ELBOW",(0,255,255)),
                    ("LEFT_WRIST", (0,200,255)), ("RIGHT_WRIST",(0,200,255)),
                ]:
                    j = POSE_IDX[name]
                    draw_joint(frame, np.array([lm_list[j].x, lm_list[j].y]), color, r=6, thick=-1)

            # Smooth overlays
            lean_s = lean_smooth(lean_deg_val)
            wti_s  = wti_smooth(wti_val)

            # Lean text
            if lean_s is not None:
                lean_dir = "FWD" if lean_s >= 0 else "BACK"
                draw_text(frame, f"Lean: {lean_s:+.1f} deg ({lean_dir})", (20, 40), COL_TEXT1, 0.8, 2, bg=True)
            else:
                draw_text(frame, "Lean: --", (20, 40), COL_TEXT1, 0.8, 2, bg=True)

            # Weight panel (bottom-center)
            panel_w = max(240, int(width * 0.30))
            panel_h = 50
            px = (width - panel_w) // 2
            py = height - panel_h - 12
            draw_weight_panel(frame, wti_s, px, py, panel_w, panel_h)

            # Write frame
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_idx += 1
            if total_frames:
                prog.progress(min(1.0, frame_idx / total_frames))
                status.text(f"Processing... {frame_idx}/{total_frames} frames")
        status.text("Done.")
        prog.progress(1.0)
    finally:
        cap.release()
        landmarker.close()
        writer.close()

    return tmp_out_path

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Cricket Lean & Weight Transfer", layout="wide")
st.title("Cricket: Lean + Weight Transfer (Side-on)")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload a side-on batting video", type=["mp4","mov","avi","mkv"])
    speed = st.slider("Playback speed (output)", min_value=0.1, max_value=2.0, value=DEFAULT_SPEED, step=0.05)
    batter_hand = st.radio("Batter hand", ["Right-hand batter", "Left-hand batter"], index=0,
                           help="Used to interpret 'forward' and which foot is 'front'.")
    roll_deg = st.slider("Camera roll correction (°)", -5.0, 5.0, DEFAULT_ROLL_DEG, 0.1)
    alpha = st.slider("Smoothing (EMA alpha)", 0.05, 0.8, DEFAULT_ALPHA, 0.05)
    run = st.button("Process video")

st.markdown("- Overlays: skeleton, trunk line, lean angle, compact Back/Front bar at bottom.")
st.markdown("- Tip: If FWD/BACK or Back/Front look flipped, toggle the 'Batter hand'.")

if run:
    if not uploaded:
        st.warning("Please upload a video.")
    else:
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        in_tmp.write(uploaded.read())
        in_tmp.flush()
        in_tmp.close()
        in_path = Path(in_tmp.name)

        try:
            out_path = process_video_to_temp(
                in_path=in_path,
                speed=speed,
                batter_hand=batter_hand,
                roll_deg=roll_deg,
                alpha=alpha
            )
            st.subheader("Annotated output")
            st.video(str(out_path))
            with open(out_path, "rb") as f:
                st.download_button(
                    label="Download annotated video",
                    data=f.read(),
                    file_name=f"{Path(uploaded.name).stem}_annotated.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Processing failed: {e}")
        finally:
            try:
                in_path.unlink(missing_ok=True)
            except Exception:
                pass
