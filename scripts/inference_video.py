"""
inference_video.py
Frame-by-frame crowd density inference on MPS.
  - Overlays jet colormap density heatmap per frame
  - Displays predicted count in top-left corner
  - Plots running count-over-time graph alongside video
  - Exports clean MP4
Supports both real video files and image folder (pseudo-video).
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from src.model import build_model
from src.utils import load_config, load_checkpoint


# ── Preprocessing ─────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, width):
    h, w   = img_bgr.shape[:2]
    new_w  = width
    new_h  = int(h * new_w / w)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    img_t   = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
    return resized, img_t


def predict_frame(model, img_bgr, width, device):
    img_resized, img_t = preprocess(img_bgr, width)
    img_t = img_t.to(device)
    with torch.no_grad():
        out = model(img_t)
    dm   = out.squeeze().cpu().numpy()
    count = float(dm.sum())
    return img_resized, dm, count


# ── Heatmap overlay ────────────────────────────────────────────────────────────

def make_heatmap_overlay(img_bgr_resized, dm, alpha):
    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
    dm_up   = cv2.resize(
        dm,
        (img_bgr_resized.shape[1], img_bgr_resized.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    dm_norm  = dm_up / (dm_up.max() + 1e-8)
    cmap     = plt.get_cmap("jet")
    heatmap  = (cmap(dm_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    overlay  = cv2.addWeighted(img_bgr_resized, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay


# ── Count label on frame ───────────────────────────────────────────────────────

def draw_count_label(frame_bgr, count, font_scale, thickness):
    label  = f"Count: {int(count)}"
    margin = 12
    font   = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    # Background rectangle
    cv2.rectangle(
        frame_bgr,
        (margin - 6, margin - 6),
        (margin + tw + 6, margin + th + baseline + 6),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame_bgr, label,
        (margin, margin + th),
        font, font_scale,
        (0, 255, 0), thickness, cv2.LINE_AA
    )
    return frame_bgr


# ── Count graph panel ──────────────────────────────────────────────────────────

def render_count_graph(count_history, graph_w, graph_h, max_count):
    """Render the count-over-time graph as a BGR numpy array."""
    dpi    = 100
    fig_w  = graph_w / dpi
    fig_h  = graph_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    x = list(range(len(count_history)))
    ax.plot(x, count_history, color="#00cc66", linewidth=1.8)
    ax.fill_between(x, count_history, alpha=0.25, color="#00cc66")

    ax.set_xlim(max(0, len(count_history) - 120), len(count_history) + 1)
    ax.set_ylim(0, max(max_count * 1.2, 10))
    ax.set_xlabel("Frame", fontsize=8, color="white")
    ax.set_ylabel("Crowd Count", fontsize=8, color="white")
    ax.set_title("Count Over Time", fontsize=9,
                 color="white", fontweight="bold")
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#444444")
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Current count annotation
    if count_history:
        ax.annotate(
            f"{int(count_history[-1])}",
            xy=(len(count_history) - 1, count_history[-1]),
            fontsize=9, color="white", fontweight="bold",
            xytext=(5, 5), textcoords="offset points"
        )

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    buf    = canvas.buffer_rgba()
    graph_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(
        int(fig_h * dpi), int(fig_w * dpi), 4
    )
    graph_bgr = cv2.cvtColor(graph_rgba, cv2.COLOR_RGBA2BGR)
    graph_bgr = cv2.resize(graph_bgr, (graph_w, graph_h))
    plt.close(fig)
    return graph_bgr


# ── Frame source ───────────────────────────────────────────────────────────────

def get_frame_source(input_path, pseudo_fps):
    """
    Returns an iterator of BGR frames and the fps to use.
    Supports: .mp4/.avi (real video) or image folder (pseudo-video).
    """
    if os.path.isdir(input_path):
        # Pseudo-video from image folder
        exts  = (".jpg", ".jpeg", ".png")
        files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(exts)
        ])
        print(f"  [Pseudo-video] {len(files)} images @ {pseudo_fps} fps")

        def gen():
            for p in files:
                img = cv2.imread(p)
                if img is not None:
                    yield img

        return gen(), pseudo_fps, len(files)

    else:
        # Real video file
        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  [Video] {total} frames @ {fps:.1f} fps")

        def gen():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

        return gen(), fps, total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--part", choices=["A", "B"], default="A",
        help="Which checkpoint to use (A=dense, B=sparse)"
    )
    args = parser.parse_args()

    cfg        = load_config(args.config)
    device     = torch.device(cfg["inference"]["device"])
    inp_width  = cfg["inference"]["input_width"]
    alpha      = cfg["inference"]["heatmap_alpha"]
    font_scale = cfg["inference"]["font_scale"]
    font_thick = cfg["inference"]["font_thickness"]
    graph_w    = cfg["video"]["graph_width"]
    graph_h    = cfg["video"]["graph_height"]
    input_path = cfg["video"]["input_path"]
    output_path= cfg["video"]["output_path"]
    out_fps    = cfg["video"]["fps"]
    pseudo_fps = cfg["video"]["pseudo_video_fps"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n  Loading Part {args.part} checkpoint onto {device}...")
    ckpt_path = cfg["training"][f"part_{args.part}"]["best_checkpoint"]
    model     = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, device=str(device))
    model.eval()
    print(f"  Checkpoint loaded: {ckpt_path}")

    # ── Frame source ──────────────────────────────────────────────────────
    frame_gen, src_fps, total_frames = get_frame_source(input_path, pseudo_fps)

    # ── Determine output frame size ───────────────────────────────────────
    # We'll figure out H from first frame
    first_frame     = next(iter(frame_gen))
    frame_gen, _, _ = get_frame_source(input_path, pseudo_fps)  # reset

    img_resized, _, _ = predict_frame(model, first_frame, inp_width, device)
    frame_h = img_resized.shape[0]
    frame_w = inp_width

    # Output canvas: video frame (left) + graph panel (right)
    canvas_w = frame_w + graph_w
    canvas_h = max(frame_h, graph_h)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (canvas_w, canvas_h),
    )

    count_history = []
    max_count     = 1.0

    print(f"\n  Processing {total_frames} frames → {output_path}")
    print(f"  Canvas: {canvas_w}×{canvas_h}  |  device: {device}\n")

    for frame_bgr in tqdm(frame_gen, total=total_frames, ncols=80):
        img_resized, dm, count = predict_frame(
            model, frame_bgr, inp_width, device
        )

        count_history.append(count)
        max_count = max(max_count, count)

        # ── Build left panel: heatmap overlay + count label ───────────────
        overlay = make_heatmap_overlay(img_resized, dm, alpha)
        overlay = draw_count_label(overlay, count, font_scale, font_thick)

        # Pad to canvas_h if needed
        if overlay.shape[0] < canvas_h:
            pad = canvas_h - overlay.shape[0]
            overlay = cv2.copyMakeBorder(
                overlay, 0, pad, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # ── Build right panel: count graph ────────────────────────────────
        graph = render_count_graph(count_history, graph_w, canvas_h, max_count)

        # ── Composite and write ───────────────────────────────────────────
        canvas = np.concatenate([overlay, graph], axis=1)
        writer.write(canvas)

    writer.release()
    print(f"\n✅  Demo saved → {output_path}")
    print(f"    Total frames : {len(count_history)}")
    print(f"    Peak count   : {max_count:.1f}")
    print(f"    Mean count   : {np.mean(count_history):.1f}")


if __name__ == "__main__":
    main()