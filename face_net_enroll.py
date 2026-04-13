import cv2
import time
import pickle
import queue
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# =========================
# Config
# =========================
DB_FILE = Path("face_encodings.pkl")
TMP_DIR = Path("tmp_face_crops")
TMP_DIR.mkdir(exist_ok=True)

CAPTURE_SAMPLES_DEFAULT = 12
CAPTURE_INTERVAL_SEC = 0.25
MATCH_THRESHOLD = 0.75  # cosine similarity threshold for recognition display

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================
# Models
# =========================
# MTCNN detects + aligns faces. InceptionResnetV1 gives FaceNet-like embeddings.
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True,
    post_process=True,
    device=DEVICE
)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


# =========================
# Persistence
# =========================
def load_db(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db, path: Path):
    with open(path, "wb") as f:
        pickle.dump(db, f)


# =========================
# Embedding / matching
# =========================
@torch.no_grad()
def embed_face_tensor(face_tensor: torch.Tensor) -> np.ndarray:
    """
    face_tensor: shape [3, 160, 160], output from MTCNN with post_process=True
    returns: normalized embedding vector
    """
    face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
    emb = resnet(face_tensor).detach().cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def best_match(embedding: np.ndarray, db: dict, threshold: float = MATCH_THRESHOLD):
    """
    db format:
    {
        "Alice": [
            {"mean_embedding": np.ndarray, "sample_count": int, "created_at": str},
            ...
        ],
        ...
    }
    """
    best_label = "Unknown"
    best_score = -1.0

    for label, entries in db.items():
        for entry in entries:
            ref = np.asarray(entry["mean_embedding"], dtype=np.float32)
            score = cosine_similarity(embedding, ref)
            if score > best_score:
                best_score = score
                best_label = label

    if best_score < threshold:
        return "Unknown", best_score
    return best_label, best_score


# =========================
# Drawing helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_box(frame, box, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def expand_and_crop(frame_bgr, box, pad_ratio=0.15):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = clamp(x1 - pad_x, 0, w - 1)
    y1 = clamp(y1 - pad_y, 0, h - 1)
    x2 = clamp(x2 + pad_x, 0, w - 1)
    y2 = clamp(y2 + pad_y, 0, h - 1)

    crop = frame_bgr[y1:y2, x1:x2].copy()
    return crop


def save_temp_crop(frame_bgr, box, label):
    label_dir = TMP_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    crop = expand_and_crop(frame_bgr, box, pad_ratio=0.15)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = label_dir / f"{ts}.jpg"
    cv2.imwrite(str(out_path), crop)
    return out_path


# =========================
# Command handling
# =========================
cmd_queue = queue.Queue()


def stdin_worker():
    print("\nCommands:")
    print("  label NAME         -> set the identity name for saving")
    print("  next               -> choose next detected face")
    print("  prev               -> choose previous detected face")
    print("  capture [N]        -> save N samples and store mean embedding")
    print("  cancel             -> stop capturing")
    print("  quit               -> exit\n")

    while True:
        try:
            line = input("cmd> ").strip()
        except EOFError:
            break
        if line:
            cmd_queue.put(line)
        if line.lower() in {"quit", "exit"}:
            break


# =========================
# Main
# =========================
def main():
    db = load_db(DB_FILE)

    current_label = "person_1"
    selected_delta = 0  # changed by next/prev commands
    capture_mode = False
    capture_target = CAPTURE_SAMPLES_DEFAULT
    capture_embeddings = []
    last_sample_time = 0.0

    running = True
    frame_count = 0

    thread = threading.Thread(target=stdin_worker, daemon=True)
    thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: webcam not accessible.")
        return

    print("Webcam started.")
    print("Type commands in terminal while the preview window is open.")

    while running:
        # Process terminal commands
        try:
            while True:
                cmd = cmd_queue.get_nowait()
                parts = cmd.split()
                key = parts[0].lower()

                if key in {"quit", "exit"}:
                    running = False
                    break

                elif key == "label" and len(parts) >= 2:
                    current_label = " ".join(parts[1:])
                    print(f"Label set to: {current_label}")

                elif key == "next":
                    selected_delta += 1
                    print("Will move to next face.")

                elif key == "prev":
                    selected_delta -= 1
                    print("Will move to previous face.")

                elif key in {"capture", "save", "start"}:
                    if len(parts) >= 2 and parts[1].isdigit():
                        capture_target = int(parts[1])
                    else:
                        capture_target = CAPTURE_SAMPLES_DEFAULT

                    capture_mode = True
                    capture_embeddings = []
                    last_sample_time = 0.0
                    print(
                        f"Capture started for label '{current_label}' "
                        f"({capture_target} samples)."
                    )

                elif key == "cancel":
                    capture_mode = False
                    capture_embeddings = []
                    print("Capture cancelled.")

                else:
                    print("Unknown command.")
        except queue.Empty:
            pass

        if not running:
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame.")
            break

        # Mirror camera view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Detect boxes
        boxes, probs = mtcnn.detect(pil_img)

        faces = None
        num_faces = 0 if boxes is None else len(boxes)

        # Update selection safely
        if num_faces > 0:
            if selected_delta != 0:
                selected_delta = selected_delta % num_faces
                selected_idx = selected_delta
                selected_delta = 0
            else:
                selected_idx = 0 if num_faces == 1 else 0
        else:
            selected_idx = 0
            selected_delta = 0

        # Draw detections
        selected_name = "Unknown"
        selected_score = 0.0

        if boxes is not None:
            for i, box in enumerate(boxes):
                color = (0, 255, 0) if i == selected_idx else (0, 0, 255)
                thickness = 3 if i == selected_idx else 2
                draw_box(frame, box, color, thickness)

            # Only compute aligned face tensors if we need embeddings or capture
            need_faces = capture_mode or len(db) > 0
            if need_faces:
                faces = mtcnn(pil_img)  # shape: [N, 3, 160, 160] or None

            if faces is not None and len(faces) > 0 and selected_idx < len(faces):
                selected_face = faces[selected_idx]
                emb = embed_face_tensor(selected_face)

                # Recognition preview from saved database
                if len(db) > 0:
                    selected_name, selected_score = best_match(emb, db)

                # Capture samples
                if capture_mode and (time.time() - last_sample_time) >= CAPTURE_INTERVAL_SEC:
                    saved_crop = save_temp_crop(frame, boxes[selected_idx], current_label)
                    capture_embeddings.append(emb)
                    last_sample_time = time.time()
                    print(
                        f"Captured {len(capture_embeddings)}/{capture_target}: "
                        f"{saved_crop.name}"
                    )

                    if len(capture_embeddings) >= capture_target:
                        mean_emb = np.mean(np.stack(capture_embeddings), axis=0)
                        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)

                        db.setdefault(current_label, []).append({
                            "mean_embedding": mean_emb.astype(np.float32),
                            "sample_count": len(capture_embeddings),
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        })
                        save_db(db, DB_FILE)

                        capture_mode = False
                        capture_embeddings = []
                        print(f"Saved encodings for '{current_label}' to {DB_FILE}")

        # HUD text
        y = 25
        hud_lines = [
            f"Label: {current_label}",
            f"Faces: {num_faces}",
            f"Selected face: {selected_idx + 1 if num_faces else 0}",
            f"Capture: {'ON' if capture_mode else 'OFF'}",
        ]

        if len(db) > 0 and num_faces > 0:
            hud_lines.append(f"Match: {selected_name} ({selected_score:.3f})")

        if capture_mode:
            hud_lines.append(f"Samples: {len(capture_embeddings)}/{capture_target}")

        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y += 26

        cv2.imshow("FaceNet Enrollment / Recognition", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()