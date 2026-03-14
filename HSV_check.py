import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import colorsys

class NpyVideoClickViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NPY Video Click Viewer")
        self.root.geometry("1000x800")

        self.video = None
        self.current_frame_index = 0
        self.photo = None
        self.zoom = 0.4
        self.image_on_canvas = None
        self.clicked_points = {}  # frame_index -> list of (x, y, color_text)
        self.display_mode = tk.StringVar(value="rgb")

        top_bar = tk.Frame(root)
        top_bar.pack(fill=tk.X, pady=8)

        self.load_button = tk.Button(top_bar, text="Load NPY Video", command=self.load_npy)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.prev_button = tk.Button(top_bar, text="<< Prev", command=self.prev_frame, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(top_bar, text="Next >>", command=self.next_frame, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.zoom_out_button = tk.Button(top_bar, text="Zoom -", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)

        self.zoom_in_button = tk.Button(top_bar, text="Zoom +", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)

        self.fit_button = tk.Button(top_bar, text="Fit Window", command=self.fit_to_window)
        self.fit_button.pack(side=tk.LEFT, padx=5)

        self.mode_menu = tk.OptionMenu(top_bar, self.display_mode, "rgb", "hsv", "grayscale", command=self.on_mode_change)
        self.mode_menu.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(root, text="Load an .npy video to begin", font=("Arial", 12))
        self.info_label.pack(pady=4)

        self.frame_label = tk.Label(root, text="Frame: -", font=("Arial", 11))
        self.frame_label.pack(pady=4)

        canvas_frame = tk.Frame(root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="black",
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.root.bind("<Left>", lambda event: self.prev_frame())
        self.root.bind("<Right>", lambda event: self.next_frame())

    def load_npy(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select NPY video file",
            filetypes=[("NumPy files", "*.npy")]
        )
        if not file_path:
            return

        try:
            arr = np.load(file_path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load file:{exc}")
            return

        try:
            self.validate_video(arr)
        except Exception as exc:
            messagebox.showerror("Error", f"Unsupported video format:{exc}")
            return

        self.video = arr
        self.current_frame_index = 0
        self.fit_to_window()
        self.update_frame_display()
        self.info_label.config(text=f"Loaded: {file_path}")
        self.update_buttons()

    def validate_video(self, arr: np.ndarray) -> None:
        if arr.ndim != 4:
            raise ValueError(f"Expected shape (N, H, W, 3/4), got {arr.shape}")
        n, h, w, c = arr.shape
        if c not in (3, 4):
            raise ValueError(f"Expected 3 or 4 channels, got {c}")
        if n == 0:
            raise ValueError("Video contains zero frames")

    def bgr_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[2] == 3:
            return frame[:, :, [2, 1, 0]]
        return frame[:, :, [2, 1, 0, 3]]

    def rgb_to_hsv_image(self, rgb: np.ndarray) -> np.ndarray:
        rgb_norm = rgb[:, :, :3].astype(np.float32) / 255.0
        r = rgb_norm[:, :, 0]
        g = rgb_norm[:, :, 1]
        b = rgb_norm[:, :, 2]

        maxc = np.max(rgb_norm, axis=2)
        minc = np.min(rgb_norm, axis=2)
        v = maxc
        deltac = maxc - minc

        s = np.zeros_like(maxc)
        nonzero = maxc != 0
        s[nonzero] = deltac[nonzero] / maxc[nonzero]

        h = np.zeros_like(maxc)
        mask = deltac != 0
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc[mask] - r[mask]) / deltac[mask]
        gc[mask] = (maxc[mask] - g[mask]) / deltac[mask]
        bc[mask] = (maxc[mask] - b[mask]) / deltac[mask]

        rmask = mask & (r == maxc)
        gmask = mask & (g == maxc)
        bmask = mask & (b == maxc)

        h[rmask] = (bc[rmask] - gc[rmask])
        h[gmask] = 2.0 + (rc[gmask] - bc[gmask])
        h[bmask] = 4.0 + (gc[bmask] - rc[bmask])
        h = (h / 6.0) % 1.0

        hsv = np.stack([h * 255.0, s * 255.0, v * 255.0], axis=2).astype(np.uint8)
        return hsv

    def rgb_to_grayscale_image(self, rgb: np.ndarray) -> np.ndarray:
        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        )
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=2)

    def frame_to_image(self, frame: np.ndarray) -> Image.Image:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = self.normalize_to_uint8(frame)

        rgb = self.bgr_to_rgb(frame)
        display_frame = rgb[:, :, :3]
        return Image.fromarray(display_frame, mode="RGB")

    def pixel_text(self, pixel_value: list[int]) -> str:
        if len(pixel_value) < 3:
            return str(pixel_value)

        if len(pixel_value) == 3:
            b, g, r = pixel_value
            a = None
        else:
            b, g, r, a = pixel_value

        rgb = (r, g, b)
        mode_name = self.display_mode.get()

        if mode_name == "rgb":
            return f"RGB={rgb}" if a is None else f"RGBA={(r, g, b, a)}"

        if mode_name == "grayscale":
            gray = int(round(0.299 * r + 0.587 * g + 0.114 * b))
            return f"Gray={gray}"

        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        hsv = (int(round(h * 255)), int(round(s * 255)), int(round(v * 255)))
        return f"HSV={hsv}"

    @staticmethod
    def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        min_val = np.min(arr)
        max_val = np.max(arr)

        if max_val == min_val:
            return np.zeros(arr.shape, dtype=np.uint8)

        arr = (arr - min_val) / (max_val - min_val)
        arr = (255 * arr).clip(0, 255).astype(np.uint8)
        return arr

    def displayed_size(self) -> tuple[int, int]:
        if self.video is None:
            return 1, 1
        _, frame_h, frame_w, _ = self.video.shape
        display_w = max(1, int(frame_w * self.zoom))
        display_h = max(1, int(frame_h * self.zoom))
        return display_w, display_h

    def update_frame_display(self) -> None:
        if self.video is None:
            return

        frame = self.video[self.current_frame_index]
        img = self.frame_to_image(frame)
        display_w, display_h = self.displayed_size()
        img = img.resize((display_w, display_h), Image.Resampling.NEAREST)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, display_w, display_h))
        self.draw_clicked_points()
        self.frame_label.config(
            text=(
                f"Frame: {self.current_frame_index + 1}/{self.video.shape[0]}   "
                f"Zoom: {self.zoom:.2f}x   Display: {display_w}x{display_h}"
            )
        )

    def update_buttons(self) -> None:
        if self.video is None:
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            return

        self.prev_button.config(
            state=tk.NORMAL if self.current_frame_index > 0 else tk.DISABLED
        )
        self.next_button.config(
            state=tk.NORMAL if self.current_frame_index < self.video.shape[0] - 1 else tk.DISABLED
        )

    def prev_frame(self) -> None:
        if self.video is None or self.current_frame_index == 0:
            return
        self.current_frame_index -= 1
        self.update_frame_display()
        self.update_buttons()

    def next_frame(self) -> None:
        if self.video is None or self.current_frame_index >= self.video.shape[0] - 1:
            return
        self.current_frame_index += 1
        self.update_frame_display()
        self.update_buttons()

    def zoom_in(self) -> None:
        self.zoom *= 1.25
        self.update_frame_display()

    def zoom_out(self) -> None:
        self.zoom /= 1.25
        self.zoom = max(self.zoom, 0.05)
        self.update_frame_display()

    def fit_to_window(self) -> None:
        self.root.update_idletasks()
        if self.video is None:
            return
        _, frame_h, frame_w, _ = self.video.shape
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        zoom_x = canvas_w / frame_w
        zoom_y = canvas_h / frame_h
        self.zoom = min(zoom_x, zoom_y)
        self.zoom = max(self.zoom, 0.05)
        if self.video is not None:
            self.update_frame_display()

    def on_mode_change(self, _value: str = "") -> None:
        self.update_frame_display()

    def on_canvas_resize(self, event: tk.Event) -> None:
        if self.video is None:
            return

    def draw_clicked_points(self) -> None:
        if self.video is None:
            return

        points = self.clicked_points.get(self.current_frame_index, [])
        radius = max(3, int(6 * self.zoom))
        font_size = max(10, int(14 * self.zoom))

        for x, y, color_text in points:
            cx = x * self.zoom
            cy = y * self.zoom
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                outline="yellow",
                width=2,
            )
            self.canvas.create_line(cx - radius, cy, cx + radius, cy, fill="yellow", width=2)
            self.canvas.create_line(cx, cy - radius, cx, cy + radius, fill="yellow", width=2)

            label_x = cx + radius + 6
            label_y = cy - radius - 6

            text_id = self.canvas.create_text(
                label_x,
                label_y,
                text=color_text,
                anchor=tk.SW,
                fill="yellow",
                font=("Arial", font_size, "bold"),
            )
            bbox = self.canvas.bbox(text_id)
            if bbox is not None:
                pad = 2
                bg_id = self.canvas.create_rectangle(
                    bbox[0] - pad,
                    bbox[1] - pad,
                    bbox[2] + pad,
                    bbox[3] + pad,
                    fill="black",
                    outline="yellow",
                )
                self.canvas.tag_lower(bg_id, text_id)

    def on_click(self, event: tk.Event) -> None:
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        x = int(canvas_x / self.zoom)
        y = int(canvas_y / self.zoom)

        _, frame_h, frame_w, _ = self.video.shape
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))

        pixel_value = self.video[self.current_frame_index, y, x].tolist() if self.video is not None else None
        color_text = self.pixel_text(pixel_value)

        if self.current_frame_index not in self.clicked_points:
            self.clicked_points[self.current_frame_index] = []
        self.clicked_points[self.current_frame_index].append((x, y, color_text))
        self.update_frame_display()

        self.info_label.config(
            text=(
                f"Clicked pixel: frame={self.current_frame_index}, x={x}, y={y}, "
                f"{color_text}"
            )
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = NpyVideoClickViewer(root)
    root.mainloop()
