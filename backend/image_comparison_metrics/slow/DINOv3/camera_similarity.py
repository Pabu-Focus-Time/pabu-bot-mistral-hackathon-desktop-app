import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA

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

# ---- Camera loop ----
selected_patch_vec = None
last_selected_frame = None
r_sel, c_sel = None, None  # initialize explicitly

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

cap = cv2.VideoCapture(0)
win_name = "Patch Cosine Similarity (3x2 Grid)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
WINDOW_MAX_W, WINDOW_MAX_H = 1600, 900
window_initialized = False

pca_model = None
first_frame_pca_fitted = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Ensure frame dims are multiples of a safe stride ---
    # ConvNeXt-style backbones commonly behave like stride 32; ViT-S/16 also divides 32.
    H0, W0 = frame.shape[:2]
    Hc = (H0 // 32) * 32
    Wc = (W0 // 32) * 32
    if Hc == 0 or Wc == 0:
        continue  # skip weird frames
    frame_proc = frame[:Hc, :Wc]  # crop to safe stride multiple

    # --- Features ---
    tensor, _ = preprocess(frame_proc)
    Xn, rows, cols, stride = extract_patches(model, tensor, TOKEN_STRIDE_PREFER)

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

    # --- 3x2 grid ---
    row1 = np.concatenate([frame_title, blended_title], axis=1)
    row2 = np.concatenate([overlay_title, last_title], axis=1)
    row3 = np.concatenate([pca_title, black_title], axis=1)
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

cap.release()
cv2.destroyAllWindows()
