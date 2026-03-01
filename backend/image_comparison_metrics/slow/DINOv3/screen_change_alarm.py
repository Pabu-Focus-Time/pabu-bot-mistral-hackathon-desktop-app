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
