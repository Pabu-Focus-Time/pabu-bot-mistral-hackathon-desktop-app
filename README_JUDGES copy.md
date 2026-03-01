

## What the trigger system does

1. **Capture the screen** (fast):
   - Uses `mss` to grab frames from a selected monitor.
   - Optionally downsamples frames for speed while keeping dimensions aligned to the model’s stride.

2. **Compute DINOv3 patch embeddings** (semantic features):
   - Uses a DINOv3 vision model (via Hugging Face `transformers`) to produce a feature vector per spatial patch.
   - Runs on `cuda` when available (Windows/NVIDIA), otherwise on Apple Silicon `mps`, otherwise CPU.

3. **Compute a scroll‑resistant “screen change” statistic**:
   - Instead of comparing the exact same patch location across frames (which breaks during scrolling), it compares each patch to the best match *within its column* in the previous frame.

4. **Trigger logic**:
   - When the statistic is above a threshold for N consecutive frames (and not within a cooldown window), we fire an action.
   - Today that action is “spawn `screen_change_alarm.py`”, but the code is structured so it can call a backend function instead.

## How the trigger score is computed

### Step A — Get patch embeddings

For each frame `t`, the model outputs a sequence of tokens `last_hidden_state` with shape `(T, D)`.

- Some of those `T` tokens are special tokens (CLS / register tokens).
- The rest correspond to spatial patches, arranged into a grid of `R x C` patches.

The code infers:

- **stride** (effective token spacing, e.g. 16 for ViT‑S/16, often 32 for ConvNeXt), and
- **number of special tokens**

by matching `T - n_special = R * C` given the current image height/width.

Each patch embedding is L2‑normalized.



$$
\hat{x}_{t,i} = \frac{x_{t,i}}{\lVert x_{t,i}\rVert_2 + \epsilon}
$$

so cosine similarity becomes a dot product.

### Step B — Column-wise best match (scroll-resistant)

If the user scrolls vertically, content moves up/down. A strict “same index” comparison would report a large change even when the same content is still present but shifted.

To make this scroll‑resistant, we compare each patch in the current frame `t` to the **best matching patch in the previous frame `t-1`** *within the same column* and within a vertical window:

- Current patch: `x_hat[t,r,c]`
- Candidate patches in previous frame: `{ x_hat[t-1,r+delta,c] }` for `delta in [-K, K]`

Per patch we take the best cosine similarity (dot product since vectors are normalized).



$$
s^{col}_{t,r,c} = \max_{\Delta \in [-K, K]\ \mathrm{valid}} \hat{x}_{t,r,c}^{\top}\hat{x}_{t-1,r+\Delta,c}
$$

Then we average over the grid:



$$
\overline{s}^{col}_t = \frac{1}{RC}\sum_{r=1}^{R}\sum_{c=1}^{C} s^{col}_{t,r,c}
$$

Finally, we convert similarity into a “change” score:



$$
\mathrm{dino\_col\_change}_t = 1 - \overline{s}^{col}_t
$$

Interpretation:

- **Near 0**: current screen is very similar to the previous frame (possibly after a small scroll).
- **Larger values**: screen content changed more significantly.

The window size `K` is configurable (env var `DINO_SCROLL_K`). Larger `K` tolerates more scroll, but costs more compute.
