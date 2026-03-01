import argparse
import ctypes
import time


def show_msgbox(text: str):
    # MB_SYSTEMMODAL = 0x1000
    ctypes.windll.user32.MessageBoxW(0, text, "DINO trigger", 0x00001000)


def show_banner(text: str, seconds: float):
    """
    Minimal OpenCV banner. Falls back to msgbox if OpenCV isn't available.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        show_msgbox(text)
        return

    seconds = max(0.2, float(seconds))
    win = "TRIGGER"

    img = np.zeros((140, 560, 3), dtype=np.uint8)
    img[:] = (30, 30, 200)  # reddish background (BGR)
    cv2.putText(img, "trigger smth", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, img)
    cv2.waitKey(1)

    t0 = time.time()
    while time.time() - t0 < seconds:
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyWindow(win)


def main():
    p = argparse.ArgumentParser(description="Simple trigger popup/banner (no model/screen work).")
    p.add_argument("--text", type=str, default="trigger smth", help="Text to display")
    p.add_argument("--seconds", type=float, default=2.0, help="Banner duration (banner mode)")
    p.add_argument("--mode", type=str, default="banner", choices=["banner", "msgbox"], help="Popup mode")
    args = p.parse_args()

    if args.mode == "msgbox":
        show_msgbox(args.text)
    else:
        show_banner(args.text, seconds=args.seconds)


if __name__ == "__main__":
    main()
    raise SystemExit(0)
import ctypes
import time


def show_msgbox(text: str):
    # MB_SYSTEMMODAL = 0x1000
    ctypes.windll.user32.MessageBoxW(0, text, "DINO trigger", 0x00001000)


def show_banner(text: str, seconds: float):
    try:
        import cv2
        import numpy as np
    except Exception:
        show_msgbox(text)
        return

    seconds = max(0.2, float(seconds))
    win = "TRIGGER"

    img = np.zeros((140, 560, 3), dtype=np.uint8)
    img[:] = (30, 30, 200)  # reddish background (BGR)
    cv2.putText(img, "trigger smth", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, img)
    cv2.waitKey(1)

    t0 = time.time()
    while time.time() - t0 < seconds:
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyWindow(win)


def main():
    p = argparse.ArgumentParser(description="Simple trigger popup/banner (no model work).")
    p.add_argument("--text", type=str, default="trigger smth", help="Text to display")
    p.add_argument("--seconds", type=float, default=2.0, help="Banner duration (banner mode)")
    p.add_argument("--mode", type=str, default="banner", choices=["banner", "msgbox"], help="Popup mode")
    args = p.parse_args()

    if args.mode == "msgbox":
        show_msgbox(args.text)
    else:
        show_banner(args.text, seconds=args.seconds)


if __name__ == "__main__":
    main()

import argparse
import ctypes
import time


def show_msgbox(text: str):
    # MB_SYSTEMMODAL = 0x1000
    ctypes.windll.user32.MessageBoxW(0, text, "DINO trigger", 0x00001000)


def show_banner(text: str, seconds: float):
    # Lightweight banner using OpenCV. Falls back to msgbox if OpenCV isn't available.
    try:
        import cv2
        import numpy as np
    except Exception:
        show_msgbox(text)
        return

    seconds = max(0.2, float(seconds))
    win = "TRIGGER"

    img = np.zeros((140, 560, 3), dtype=np.uint8)
    img[:] = (30, 30, 200)  # reddish background (BGR)
    cv2.putText(img, "trigger smth", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, img)
    cv2.waitKey(1)

    t0 = time.time()
    while time.time() - t0 < seconds:
        # Allow ESC to close early
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyWindow(win)


def main():
    p = argparse.ArgumentParser(description="Simple trigger popup/banner (no model work).")
    p.add_argument("--text", type=str, default="trigger smth", help="Text to display")
    p.add_argument("--seconds", type=float, default=2.0, help="Banner duration (banner mode)")
    p.add_argument("--mode", type=str, default="banner", choices=["banner", "msgbox"], help="Popup mode")
    args = p.parse_args()

    if args.mode == "msgbox":
        show_msgbox(args.text)
    else:
        show_banner(args.text, seconds=args.seconds)


if __name__ == "__main__":
    main()

import os
import time
import threading
import ctypes

import cv2
import mss
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# ---------------- Settings (env vars) ----------------
MODEL_ID = os.environ.get("DINO_MODEL_ID", "facebook/dinov3-vits16-pretrain-lvd1689m")
HF_TOKEN = os.environ.get("HF_TOKEN")

MONITOR_INDEX = int(os.environ.get("DINO_SCREEN_MONITOR", "1"))  # 1=primary monitor
SCREEN_SCALE = float(os.environ.get("DINO_SCREEN_SCALE", "0.5"))  # e.g. 0.5 for faster

TOKEN_STRIDE_PREFER = int(os.environ.get("DINO_TOKEN_STRIDE_PREFER", "16"))
SCROLL_K = int(os.environ.get("DINO_SCROLL_K", "16"))

THRESH = float(os.environ.get("DINO_TRIGGER_THRESH", "0.15"))
CONSEC = int(os.environ.get("DINO_TRIGGER_CONSEC", "3"))
COOLDOWN_SEC = float(os.environ.get("DINO_TRIGGER_COOLDOWN_SEC", "5.0"))

POPUP_MODE = os.environ.get("DINO_TRIGGER_MODE", "banner").strip().lower()  # "banner" or "msgbox"
POPUP_SECONDS = float(os.environ.get("DINO_TRIGGER_SECONDS", "2.0"))
EXIT_AFTER_TRIGGER = os.environ.get("DINO_TRIGGER_EXIT", "0").strip() not in ("0", "false", "False", "")


# ---------------- Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trust = bool(str(MODEL_ID).startswith("facebook/dinov3")) or os.path.isdir(str(MODEL_ID))
if HF_TOKEN:
    try:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=trust, token=HF_TOKEN).to(device).eval()
    except TypeError:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=trust, use_auth_token=HF_TOKEN).to(device).eval()
else:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=trust).to(device).eval()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)


def infer_stride_and_special_tokens(H: int, W: int, T: int, prefer_stride: int):
    stride_candidates = [prefer_stride, 32, 16, 14, 8]
    seen = set()
    stride_candidates = [s for s in stride_candidates if isinstance(s, int) and s > 0 and not (s in seen or seen.add(s))]

    for stride in stride_candidates:
        if H % stride != 0 or W % stride != 0:
            continue
        rows, cols = H // stride, W // stride
        n_patches = rows * cols
        for n_special in range(0, 33):
            if T - n_special == n_patches:
                return stride, n_special, rows, cols
    return prefer_stride, 1, H // prefer_stride, W // prefer_stride


def extract_Xn(model, tensor: torch.Tensor, prefer_stride: int):
    with torch.no_grad():
        out = model(pixel_values=tensor)
    hs = out.last_hidden_state.squeeze(0)  # (T, D) on device
    T, D = int(hs.shape[0]), int(hs.shape[1])
    H, W = int(tensor.shape[2]), int(tensor.shape[3])
    stride, n_special, rows, cols = infer_stride_and_special_tokens(H, W, T, prefer_stride=prefer_stride)
    n_patches = rows * cols
    X = hs[n_special : n_special + n_patches, :].reshape(n_patches, D)
    Xn = X / (torch.linalg.norm(X, dim=1, keepdim=True) + 1e-8)
    return Xn, rows, cols, stride


def dino_col_change(prev_Xn: torch.Tensor, curr_Xn: torch.Tensor, rows: int, cols: int, k_rows: int):
    if prev_Xn is None or curr_Xn is None:
        return float("nan")
    if prev_Xn.shape != curr_Xn.shape or prev_Xn.shape[0] != rows * cols:
        return float("nan")

    k_rows = max(0, int(k_rows))
    prev = prev_Xn.reshape(rows, cols, -1)
    curr = curr_Xn.reshape(rows, cols, -1)

    best = torch.full((rows, cols), float("-inf"), device=curr.device, dtype=curr.dtype)
    for delta in range(-k_rows, k_rows + 1):
        if delta >= 0:
            r_curr0, r_curr1 = 0, rows - delta
            r_prev0, r_prev1 = delta, rows
        else:
            r_curr0, r_curr1 = -delta, rows
            r_prev0, r_prev1 = 0, rows + delta

        if r_curr1 <= r_curr0:
            continue

        sim = (curr[r_curr0:r_curr1] * prev[r_prev0:r_prev1]).sum(dim=-1)
        best[r_curr0:r_curr1] = torch.maximum(best[r_curr0:r_curr1], sim)

    mean_best = float(best.mean().detach().cpu().item())
    return float(1.0 - mean_best)


def _show_msgbox(text: str):
    def _run():
        try:
            ctypes.windll.user32.MessageBoxW(0, text, "DINO trigger", 0x00001000)  # system modal
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def _show_banner(text: str, seconds: float):
    """
    Lightweight OpenCV banner window. Not guaranteed to be topmost, but non-blocking.
    """
    seconds = max(0.2, float(seconds))
    win = "TRIGGER"
    img = np.zeros((140, 520, 3), dtype=np.uint8)
    img[:] = (30, 30, 200)  # reddish
    cv2.putText(img, "trigger smth", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, img)
    cv2.waitKey(1)
    t0 = time.time()
    while time.time() - t0 < seconds:
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cv2.destroyWindow(win)


def trigger_popup(change_value: float):
    msg = f"dino_col_change={change_value:.3f} > thresh={THRESH:g}"
    if POPUP_MODE == "msgbox":
        _show_msgbox("trigger smth\n" + msg)
    else:
        _show_banner(msg, seconds=POPUP_SECONDS)


def main():
    prev_Xn = None
    prev_shape = None  # (rows, cols, stride)

    above = 0
    last_trigger_ts = 0.0

    with mss.mss() as sct:
        monitors = sct.monitors
        mon = monitors[MONITOR_INDEX] if 0 <= MONITOR_INDEX < len(monitors) else monitors[1]

        while True:
            frame = np.array(sct.grab(mon))[:, :, :3]  # BGR

            # Crop to safe multiple (32), then optional downscale (still multiple of 32)
            H0, W0 = frame.shape[:2]
            Hc = (H0 // 32) * 32
            Wc = (W0 // 32) * 32
            if Hc == 0 or Wc == 0:
                continue
            frame_proc = frame[:Hc, :Wc]
            if 0 < SCREEN_SCALE < 1.0:
                new_w = max(32, int(Wc * SCREEN_SCALE) // 32 * 32)
                new_h = max(32, int(Hc * SCREEN_SCALE) // 32 * 32)
                if new_w != Wc or new_h != Hc:
                    frame_proc = cv2.resize(frame_proc, (new_w, new_h), interpolation=cv2.INTER_AREA)

            tensor = preprocess(np.ascontiguousarray(frame_proc, dtype=np.uint8))
            curr_Xn, rows, cols, stride = extract_Xn(model, tensor, prefer_stride=TOKEN_STRIDE_PREFER)

            curr_shape = (int(rows), int(cols), int(stride))
            if prev_Xn is None or prev_shape != curr_shape:
                change = float("nan")
            else:
                change = dino_col_change(prev_Xn, curr_Xn, rows=rows, cols=cols, k_rows=SCROLL_K)

            prev_Xn = curr_Xn
            prev_shape = curr_shape

            if np.isfinite(change) and change > THRESH:
                above += 1
            else:
                above = 0

            now = time.time()
            if above >= max(1, CONSEC) and (now - last_trigger_ts) >= COOLDOWN_SEC:
                last_trigger_ts = now
                above = 0
                print(f"trigger smth (dino_col_change={change:.4f})", flush=True)
                trigger_popup(change)
                if EXIT_AFTER_TRIGGER:
                    break


if __name__ == "__main__":
    main()

