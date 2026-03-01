# Screen change trigger (MacBook Pro / Apple Silicon)

This guide is for running `screen_change_trigger.py` on a **MacBook Pro (M‑chip)** and adding your own backend logic where the trigger fires.

## What this does

- Captures your screen (via `mss`)
- Computes a **scroll‑resistant “screen changed” score**: `dino_col_change`
- Plots `dino_col_change` over time
- When `dino_col_change` stays above a threshold for N frames, it triggers an action

By default the action is to spawn `screen_change_alarm.py` which shows a small banner/notification. You will replace that with your backend call.

## Prereqs

- **Python**: 3.11 recommended
- **Hugging Face token**: you need access to gated `facebook/dinov3-*` models. 
- **macOS permission**: Terminal (or your IDE) must be allowed to record the screen

## 1) Create a virtual environment

From the repo root:

```bash
cd backend/image_comparison_metrics/slow/DINOv3
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

## 2) Install dependencies

This installs everything used by the DINOv3 scripts (including screen capture + OpenCV):

```bash
pip install -r requirements.txt
```

Notes:

- On Apple Silicon, PyTorch should use **MPS** automatically when available.
- If you hit any PyTorch install issues, follow the official PyTorch macOS install instructions for Apple Silicon and then re-run `pip install -r requirements.txt`.

## 3) Add your Hugging Face token

Create a `.env` file next to the scripts:

`backend/image_comparison_metrics/slow/DINOv3/.env`

with:

```env
HF_TOKEN=hf_your_token_here
```

## 4) Grant Screen Recording permission (macOS)

Go to:

- **System Settings → Privacy & Security → Screen Recording**

Enable it for the app you’re using to run the script (Terminal, iTerm, VS Code, etc).  
Then restart that app.

## 5) Run the trigger script

```bash
python screen_change_trigger.py
```

Press **ESC** to quit.

## Common knobs (environment variables)

Run with env vars to tune performance/behavior:

```bash
# faster (downscale)
export DINO_SCREEN_SCALE=0.5

# scroll resistance window (patch rows)
export DINO_SCROLL_K=16

# plot y-axis range
export DINO_PLOT_YMAX=0.4

# trigger condition
export DINO_TRIGGER_THRESH=0.15
export DINO_TRIGGER_CONSEC=1
export DINO_TRIGGER_COOLDOWN_SEC=5

# what alarm popup should do on mac
export DINO_TRIGGER_MODE=notify   # notify|banner|msgbox
export DINO_TRIGGER_SECONDS=2

python screen_change_trigger.py
```

## Where to add your backend logic

Open `screen_change_trigger.py` and find this block (around the bottom of the main loop):

```python
# AT THE MOMENT HERE IT JUST SPAWNS THE ALARM SCRIPT, BUT YOU CAN TRIGGER WHATEVER YOU WANT
spawn_alarm(change)
```

Replace `spawn_alarm(change)` with your backend call, e.g.:

- send a message to your backend process
- write to a socket
- call an internal Python function
- enqueue an event (Redis, etc.)

The variable `change` is the current `dino_col_change` value that exceeded your threshold.

## Quick mental model: what is `dino_col_change`?

It’s computed from DINOv3 patch embeddings, but instead of comparing patch (r,c) at time t to patch (r,c) at time t+1 (which breaks on scroll), it compares patch (r,c) at time t+1 to the **best match in the same column** in time t within ±K rows. This makes it **scroll‑resistant**.
