import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
import mss

# ---- Model loading (adapt to your setup) ----
from transformers import AutoModel

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
# MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hugging Face auth (for gated DINOv3 repos)
HF_TOKEN = os.environ.get("HF_TOKEN")
# trust_remote_code helps with facebook/dinov3 repos
if HF_TOKEN:
    try:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN).to(device).eval()
    except TypeError:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, use_auth_token=HF_TOKEN).to(device).eval()
else:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()

def infer_stride_and_special_tokens(H: int, W: int, T: int, prefer_stride: int):
    # Try preferred stride first, then common candidates.
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

# Default preferred token stride (ViT-S/16). For ConvNeXt the inferred stride is usually 32.
TOKEN_STRIDE_PREFER = 16

# ---- Normalization stats ----
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def preprocess(img_bgr):
    # Do NOT resize: keep original multiples of stride for proper patch mapping
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)
    return tensor, pil_img

def extract_patches(model, tensor, prefer_stride: int):
    with torch.no_grad():
        out = model(pixel_values=tensor)
    # last_hidden_state: (1, T, D) where T = [CLS] + patches
    hs = out.last_hidden_state.squeeze(0).cpu().numpy()  # (T, D)

    H, W = tensor.shape[2], tensor.shape[3]
    T, D = hs.shape
    stride, n_special, rows, cols = infer_stride_and_special_tokens(int(H), int(W), int(T), prefer_stride=int(prefer_stride))
    n_patches = rows * cols

    patches = hs[n_special:n_special + n_patches, :]  # drop special tokens
    patches = patches.reshape(rows, cols, D)

    X = patches.reshape(-1, D)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn, rows, cols, stride

# ---- Screen loop ----
selected_patch_vec = None
last_selected_frame = None
r_sel, c_sel = None, None  # initialize explicitly

# ---- Change tracking ----
prev_frame_proc = None
prev_Xn = None
prev_shape = None  # (rows, cols, stride)

CHANGE_HISTORY_MAX = int(os.environ.get("DINO_CHANGE_HISTORY", "240"))
change_hist_pixel = []
change_hist_dino = []
change_hist_dino_col = []
PLOT_YMAX = float(os.environ.get("DINO_PLOT_YMAX", "0.4"))
PLOT_TITLE_SCALE = float(os.environ.get("DINO_PLOT_TITLE_SCALE", "0.65"))
PLOT_LEGEND_SCALE = float(os.environ.get("DINO_PLOT_LEGEND_SCALE", "0.5"))
PLOT_FOOTER_SCALE = float(os.environ.get("DINO_PLOT_FOOTER_SCALE", "0.5"))

def _push_hist(hist, v: float):
    hist.append(float(v))
    if len(hist) > CHANGE_HISTORY_MAX:
        del hist[: len(hist) - CHANGE_HISTORY_MAX]

def compute_pixel_change(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    if prev_bgr.shape != curr_bgr.shape:
        return float("nan")
    # Mean absolute difference scaled to [0,1]
    diff = np.abs(curr_bgr.astype(np.int16) - prev_bgr.astype(np.int16)).mean()
    return float(diff / 255.0)

def compute_dino_change(prev_Xn: np.ndarray, curr_Xn: np.ndarray) -> float:
    if prev_Xn is None or curr_Xn is None or prev_Xn.shape != curr_Xn.shape:
        return float("nan")
    # Mean cosine similarity of corresponding patches (Xn are already L2-normalized)
    sim = float(np.mean(np.sum(prev_Xn * curr_Xn, axis=1)))
    # Convert similarity [-1,1] to a "change" score where 0 means identical
    return float(1.0 - sim)

def compute_dino_change_columnwise(
    prev_Xn: np.ndarray,
    curr_Xn: np.ndarray,
    rows: int,
    cols: int,
    k_rows: int = 16,
) -> float:
    """
    Scroll-resistant DINO change.
    For each patch at (r,c) in the current frame, find the best cosine match among patches
    (r+delta, c) in the previous frame for delta in [-k_rows, k_rows].

    Returns: 1 - mean(best_similarity).
    """
    if prev_Xn is None or curr_Xn is None or prev_Xn.shape != curr_Xn.shape:
        return float("nan")
    if rows <= 0 or cols <= 0:
        return float("nan")
    if prev_Xn.shape[0] != rows * cols:
        return float("nan")

    k_rows = max(0, int(k_rows))
    # Use torch for speed (vectorized dot products); run on same device as model if possible.
    dev = device
    prev = torch.from_numpy(prev_Xn).to(dev, dtype=torch.float32).reshape(rows, cols, -1)
    curr = torch.from_numpy(curr_Xn).to(dev, dtype=torch.float32).reshape(rows, cols, -1)

    best = torch.full((rows, cols), float("-inf"), device=dev, dtype=torch.float32)
    for delta in range(-k_rows, k_rows + 1):
        if delta >= 0:
            # curr[r] vs prev[r+delta]
            r_curr0, r_curr1 = 0, rows - delta
            r_prev0, r_prev1 = delta, rows
        else:
            # curr[r] vs prev[r+delta]  (delta negative)
            r_curr0, r_curr1 = -delta, rows
            r_prev0, r_prev1 = 0, rows + delta

        if r_curr1 <= r_curr0:
            continue

        sim = (curr[r_curr0:r_curr1] * prev[r_prev0:r_prev1]).sum(dim=-1)  # (overlap_rows, cols)
        best[r_curr0:r_curr1] = torch.maximum(best[r_curr0:r_curr1], sim)

    mean_best = float(best.mean().detach().cpu().item())
    return float(1.0 - mean_best)

def render_change_plot(h_pixel, h_dino, h_dino_col, size_wh):
    W, H = int(size_wh[0]), int(size_wh[1])
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Background + title
    cv2.putText(img, "Change over time", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, PLOT_TITLE_SCALE, (255, 255, 255), 2, cv2.LINE_AA)

    # Plot area
    x0, y0 = 10, 45
    x1, y1 = W - 10, H - 20
    if x1 <= x0 + 10 or y1 <= y0 + 10:
        return img
    cv2.rectangle(img, (x0, y0), (x1, y1), (120, 120, 120), 1)

    def _plot(series, color, label, y_scale_max=1.0):
        vals = [v for v in series if np.isfinite(v)]
        if len(vals) < 2:
            return
        # Use the last N points that fit the width
        N = min(len(series), max(2, x1 - x0))
        s = series[-N:]
        # Normalize y
        pts = []
        for i, v in enumerate(s):
            if not np.isfinite(v):
                continue
            t = i / float(max(1, N - 1))
            x = int(x0 + t * (x1 - x0))
            vv = max(0.0, min(float(v), y_scale_max))
            y = int(y1 - (vv / y_scale_max) * (y1 - y0))
            pts.append((x, y))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2)
        # Legend entry
        cv2.putText(img, label, (x0 + 8, y0 + 20 if label.endswith("pixel") else y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, PLOT_LEGEND_SCALE, color, 2, cv2.LINE_AA)

    # Pixel change is naturally [0,1]
    _plot(h_pixel, (80, 200, 255), "pixel", y_scale_max=PLOT_YMAX)
    # DINO change can exceed 1 if sim goes negative; clamp display to [0,1] for readability
    _plot(h_dino, (255, 180, 80), "dino", y_scale_max=PLOT_YMAX)
    _plot(h_dino_col, (140, 255, 140), "dino_col", y_scale_max=PLOT_YMAX)

    # Latest values
    last_p = next((v for v in reversed(h_pixel) if np.isfinite(v)), float("nan"))
    last_d = next((v for v in reversed(h_dino) if np.isfinite(v)), float("nan"))
    last_dc = next((v for v in reversed(h_dino_col) if np.isfinite(v)), float("nan"))
    cv2.putText(img, f"latest: pixel={last_p:.3f}  dino={last_d:.3f}  dino_col={last_dc:.3f}", (10, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, PLOT_FOOTER_SCALE, (220, 220, 220), 2, cv2.LINE_AA)

    return img

def mouse_callback(event, x, y, flags, param):
    global selected_patch_vec, last_selected_frame, r_sel, c_sel
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param.get('scale', 1.0)
        x_unscaled = int(x / max(scale, 1e-8))
        y_unscaled = int(y / max(scale, 1e-8))

        cols = param['cols']
        rows = param['Xn'].shape[0] // cols
        stride = int(param.get("stride", TOKEN_STRIDE_PREFER))

        # Click must be inside the patch grid area
        if x_unscaled < cols * stride and y_unscaled < rows * stride:
            r = y_unscaled // stride
            c = x_unscaled // stride
            idx = r * cols + c
            if 0 <= idx < param['Xn'].shape[0]:
                # Keep a normalized copy
                v = param['Xn'][idx].copy()
                v /= (np.linalg.norm(v) + 1e-8)
                selected_patch_vec = v
                last_selected_frame = param['frame'].copy()
                r_sel, c_sel = r, c

win_name = "Patch Cosine Similarity (3x2 Grid)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
WINDOW_MAX_W, WINDOW_MAX_H = 1600, 900
window_initialized = False

# Screen capture settings (primary monitor by default)
MONITOR_INDEX = int(os.environ.get("DINO_SCREEN_MONITOR", "1"))
# Downscale factor for processing (e.g. 0.5 = half-res). Use 1.0 for native.
SCREEN_SCALE = float(os.environ.get("DINO_SCREEN_SCALE", "0.1"))

pca_model = None
first_frame_pca_fitted = False

with mss.mss() as sct:
    monitors = sct.monitors
    mon = monitors[MONITOR_INDEX] if 0 <= MONITOR_INDEX < len(monitors) else monitors[1]

    while True:
        # mss returns BGRA; OpenCV expects BGR (we just drop alpha)
        frame = np.array(sct.grab(mon))[:, :, :3]

        # --- Ensure frame dims are multiples of a safe stride ---
        # ConvNeXt-style backbones commonly behave like stride 32; ViT-S/16 also divides 32.
        H0, W0 = frame.shape[:2]
        Hc = (H0 // 32) * 32
        Wc = (W0 // 32) * 32
        if Hc == 0 or Wc == 0:
            continue  # skip weird frames
        frame_proc = frame[:Hc, :Wc]  # crop to safe stride multiple

        # Optional downscale for speed (keep multiples of 32 so token grid stays consistent)
        if 0 < SCREEN_SCALE < 1.0:
            new_w = max(32, int(Wc * SCREEN_SCALE) // 32 * 32)
            new_h = max(32, int(Hc * SCREEN_SCALE) // 32 * 32)
            if new_w != Wc or new_h != Hc:
                frame_proc = cv2.resize(frame_proc, (new_w, new_h), interpolation=cv2.INTER_AREA)
                Hc, Wc = new_h, new_w

        # --- Features ---
        tensor, _ = preprocess(frame_proc)
        Xn, rows, cols, stride = extract_patches(model, tensor, TOKEN_STRIDE_PREFER)

        # --- Frame-to-frame change metrics ---
        curr_shape = (int(rows), int(cols), int(stride))
        if prev_frame_proc is None or prev_Xn is None or prev_shape != curr_shape:
            pixel_change = float("nan")
            dino_change = float("nan")
            dino_col_change = float("nan")
        else:
            pixel_change = compute_pixel_change(prev_frame_proc, frame_proc)
            dino_change = compute_dino_change(prev_Xn, Xn)
            k_rows = int(os.environ.get("DINO_SCROLL_K", "16"))
            dino_col_change = compute_dino_change_columnwise(prev_Xn, Xn, rows=rows, cols=cols, k_rows=k_rows)
        _push_hist(change_hist_pixel, pixel_change)
        _push_hist(change_hist_dino, dino_change)
        _push_hist(change_hist_dino_col, dino_col_change)

        prev_frame_proc = frame_proc
        prev_Xn = Xn
        prev_shape = curr_shape

        # --- Cosine overlay ---
        overlay_color = np.zeros_like(frame_proc)
        if selected_patch_vec is not None:
            qn = selected_patch_vec
            cos_map = Xn @ qn  # (rows*cols,)
            # Normalize to [0,1] for visualization
            cmin, cmax = 0.0, 1.0
            cos_map = (cos_map - cmin) / (cmax - cmin + 1e-8)
            cos_map_img = cos_map.reshape(rows, cols)
            overlay_small = cv2.resize(
                cos_map_img.astype(np.float32),
                (cols * stride, rows * stride),
                interpolation=cv2.INTER_NEAREST
            )
            overlay_color = cv2.applyColorMap((overlay_small * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

            # Blend (same size as frame_proc)
            blended = cv2.addWeighted(frame_proc, 0.5, overlay_color, 0.5, 0.0)
        else:
            blended = frame_proc

        # --- Last selected view with grid and red rectangle ---
        if last_selected_frame is not None and r_sel is not None and c_sel is not None:
            last_grid = last_selected_frame.copy()
            hl, wl = last_grid.shape[:2]
            rows_last = hl // stride
            cols_last = wl // stride
            # draw grid
            for rr in range(rows_last):
                y = rr * stride
                cv2.line(last_grid, (0, y), (wl, y), (200, 200, 200), 1)
            for cc in range(cols_last):
                x = cc * stride
                cv2.line(last_grid, (x, 0), (x, hl), (200, 200, 200), 1)
            # selected patch
            x0 = c_sel * stride
            y0 = r_sel * stride
            x1 = x0 + stride
            y1 = y0 + stride
            cv2.rectangle(last_grid, (x0, y0), (x1, y1), (0, 0, 255), 2)
            # match current size
            last_view = cv2.resize(last_grid, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        else:
            last_view = np.zeros_like(frame_proc)

        # --- PCA Visualization ---
        if not first_frame_pca_fitted and Xn.shape[1] >= 3:
            pca_model = PCA(n_components=3)
            pca_model.fit(Xn)
            first_frame_pca_fitted = True

        # If a patch is selected, refit PCA on features of the frame at selection
        if selected_patch_vec is not None and last_selected_frame is not None:
            tensor_sel, _ = preprocess(last_selected_frame[:Hc, :Wc])  # safe crop if needed
            Xn_sel, _, _, _ = extract_patches(model, tensor_sel, TOKEN_STRIDE_PREFER)
            if Xn_sel.shape[1] >= 3:
                pca_model = PCA(n_components=3)
                pca_model.fit(Xn_sel)

        if pca_model is not None and Xn.shape[1] >= 3 and Xn.shape[0] == rows * cols:
            pca_feats = pca_model.transform(Xn)
            lo = np.percentile(pca_feats, 1, axis=0, keepdims=True)
            hi = np.percentile(pca_feats, 99, axis=0, keepdims=True)
            pca_feats = (pca_feats - lo) / (hi - lo + 1e-8)
            pca_feats = np.clip(pca_feats, 0, 1)
            pca_small = (pca_feats.reshape(rows, cols, 3) * 255).astype(np.uint8)
            pca_img = cv2.resize(pca_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        else:
            pca_img = np.zeros_like(frame_proc)

        # --- Titles ---
        def add_title(img, text):
            out = img.copy()
            cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            return out

        frame_title   = add_title(frame_proc, "Original")
        blended_title = add_title(blended, "Original + Cosine Overlay")
        overlay_title = add_title(overlay_color, "Cosine Overlay Only")
        last_title    = add_title(last_view, "Last Selection (Grid + Red)")
        pca_title     = add_title(pca_img, "PCA Visualization")
        black_title   = add_title(np.zeros_like(frame_proc), "Empty")
        change_plot = render_change_plot(change_hist_pixel, change_hist_dino, change_hist_dino_col, (Wc, Hc))
        change_title = add_title(change_plot, "Change (pixel & DINO)")

        # --- 3x2 grid ---
        row1 = np.concatenate([frame_title, blended_title], axis=1)
        row2 = np.concatenate([overlay_title, last_title], axis=1)
        row3 = np.concatenate([pca_title, change_title], axis=1)
        grid = np.concatenate([row1, row2, row3], axis=0)

        # --- Fit to window once grid exists (fix: do NOT touch 'grid' before this) ---
        gh, gw = grid.shape[:2]

        # Read current window size (users can resize it)
        try:
            rect = cv2.getWindowImageRect(win_name)
            screen_w, screen_h = max(1, int(rect[2])), max(1, int(rect[3]))
        except Exception:
            screen_w, screen_h = WINDOW_MAX_W, WINDOW_MAX_H

        # Set a sensible initial window size that matches the grid aspect ratio
        if not window_initialized:
            init_scale = min(WINDOW_MAX_W / gw, WINDOW_MAX_H / gh)
            cv2.resizeWindow(win_name, max(1, int(gw * init_scale)), max(1, int(gh * init_scale)))
            window_initialized = True
            try:
                rect = cv2.getWindowImageRect(win_name)
                screen_w, screen_h = max(1, int(rect[2])), max(1, int(rect[3]))
            except Exception:
                pass

        # Uniform scaling => no horizontal/vertical distortion
        scale = min(screen_w / gw, screen_h / gh)
        if abs(scale - 1.0) > 1e-6:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
            grid = cv2.resize(
                grid,
                (max(1, int(gw * scale)), max(1, int(gh * scale))),
                interpolation=interp
            )

        # Register mouse callback ONCE per frame, after 'scale' is known
        cv2.setMouseCallback(
            win_name,
            mouse_callback,
            param={
                'cols': cols,
                'Xn': Xn,
                'frame': frame_proc,           # use the cropped/processed frame
                'scale': scale,
                'stride': int(stride),
            }
        )

        cv2.imshow(win_name, grid)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
cv2.destroyAllWindows()
