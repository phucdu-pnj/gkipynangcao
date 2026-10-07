from pathlib import Path
import threading

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


# ---------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------
app = FastAPI(title="GKIPython - Stock Animation")

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "abc-28.csv"
VIDEO_FILE = BASE_DIR / "dothi.mp4"

_generation_lock = threading.Lock()
_generation_started = False


def generate_video():
    """Đọc dữ liệu và tạo file dothi.mp4."""
    if VIDEO_FILE.exists() and VIDEO_FILE.stat().st_size > 0:
        return

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {CSV_FILE.name}. "
            "Hãy đặt abc-28.csv cùng thư mục với main.py."
        )

    if imageio_ffmpeg is None:
        raise RuntimeError(
            "Thiếu imageio-ffmpeg. Hãy cài thư viện này trong requirements.txt."
        )

    # Cho Matplotlib dùng ffmpeg được đóng gói cùng imageio-ffmpeg.
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    # ---------------------------------------------------------
    # 1. Đọc và chuẩn bị dữ liệu
    # ---------------------------------------------------------
    df = pd.read_csv(CSV_FILE, header=None)

    if df.shape != (365, 7):
        raise ValueError(
            f"Dữ liệu phải có 365 dòng và 7 cột; "
            f"hiện có {df.shape[0]} dòng, {df.shape[1]} cột."
        )

    df = df.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            "Dữ liệu giá không được chứa giá trị thiếu hoặc vô hạn."
        )

    day_count = len(df)
    label_x = day_count + 13

    candle_days = set()
    for start in range(0, day_count, 30):
        end = min(start + 30, day_count)
        candle_count = min(
            8, max(1, round((end - start) * 8 / 30))
        )
        bucket_width = (end - start) / candle_count

        candle_days.update(
            np.linspace(
                start + bucket_width / 2,
                end - bucket_width / 2,
                candle_count,
                dtype=int,
            )
        )

    # 7 cột tương ứng với các mức giá trong ngày
    price_cols = list(range(7))
    df["Open"] = df[0]
    df["Close"] = df[6]
    df["High"] = df[price_cols].max(axis=1)
    df["Low"] = df[price_cols].min(axis=1)

    # Giá tham chiếu, giá trần, giá sàn
    df["Ref_Price"] = df["Close"].shift(1)
    df.loc[0, "Ref_Price"] = df.loc[0, "Open"]
    df["Floor_Price"] = df["Ref_Price"] * 0.9
    df["Ceil_Price"] = df["Ref_Price"] * 1.1

    global_max = df["High"].max()
    global_min = df["Low"].min()

    # ---------------------------------------------------------
    # 2. Figure
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    fig.patch.set_facecolor("black")
    plt.subplots_adjust(
        left=0.11,
        right=0.95,
        bottom=0.1,
        top=0.99,
        hspace=0.08,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=12)

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.set_ylabel("VND", color="white", fontweight="bold")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )

    y_axis_max = np.ceil(global_max / 5000) * 5000

    for ax in (ax1, ax2):
        ax.set_ylim(0, y_axis_max)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))

    ax2.set_xlabel("Date", color="white", fontweight="bold")
    ax2.set_xticks(np.arange(0, day_count, 30))
    ax2.set_xlim(-5, day_count + 35)

    ax2.axhline(
        y=global_max,
        color="green",
        linestyle="--",
        lw=0.8,
        alpha=0.6,
    )
    ax2.axhline(
        y=global_min,
        color="red",
        linestyle="--",
        lw=0.8,
        alpha=0.6,
    )

    ax2.text(
        label_x,
        global_max,
        f"{int(global_max):,}",
        color="green",
        va="center",
        ha="right",
    )
    ax2.text(
        label_x,
        global_min,
        f"{int(global_min):,}",
        color="red",
        va="center",
        ha="right",
    )

    days_data = []
    lows_data = []
    highs_data = []

    line_current = ax2.axhline(
        y=0,
        color="brown",
        linestyle="-",
        lw=1.5,
        alpha=0.7,
    )

    text_current = ax2.text(
        label_x,
        0,
        "",
        color="red",
        va="center",
        ha="right",
    )

    fill_collection = None

    high_line, = ax2.plot(
        [], [], color="olivedrab", lw=1.4
    )
    low_line, = ax2.plot(
        [], [], color="darkorange", lw=1.4
    )

    def get_candle_color(close_p, ref_p, floor_p, ceil_p):
        if np.isclose(close_p, floor_p, atol=1e-2):
            return "dodgerblue"

        if np.isclose(close_p, ceil_p, atol=1e-2):
            return "mediumorchid"

        if np.isclose(close_p, ref_p, atol=1e-2):
            return "yellow"

        if close_p < ref_p:
            return "red"

        return "limegreen"

    def init():
        return []

    def animate(i):
        nonlocal fill_collection

        row = df.iloc[i]
        day = i

        o = row["Open"]
        c = row["Close"]
        h = row["High"]
        l = row["Low"]

        ref = row["Ref_Price"]
        floor = row["Floor_Price"]
        ceil = row["Ceil_Price"]

        color = get_candle_color(c, ref, floor, ceil)

        # Biểu đồ trên - candlestick
        if day in candle_days:
            ax1.plot(
                [day, day],
                [l, h],
                color=color,
                lw=1.8,
            )

            body_bottom = min(o, c)
            body_height = max(
                abs(o - c),
                (global_max - global_min) * 0.005,
            )

            rect = plt.Rectangle(
                (day - 1.1, body_bottom),
                2.2,
                body_height,
                facecolor=color,
                edgecolor=color,
            )

            ax1.add_patch(rect)

            ax1.plot(
                [day - 0.8, day + 0.8],
                [(o + c) / 2, (o + c) / 2],
                color="white",
                lw=0.9,
                alpha=0.8,
            )

        # Biểu đồ dưới - area band
        days_data.append(day)
        lows_data.append(l)
        highs_data.append(h)

        if fill_collection is not None:
            fill_collection.remove()

        fill_collection = ax2.fill_between(
            days_data,
            lows_data,
            highs_data,
            color="darkorange",
            alpha=0.45,
        )

        high_line.set_data(days_data, highs_data)
        low_line.set_data(days_data, lows_data)

        line_current.set_ydata([c, c])
        text_current.set_position((label_x, c))
        text_current.set_text(f"{int(c):,}")

        return []

    # ---------------------------------------------------------
    # 3. Tạo video
    # ---------------------------------------------------------
    playback_fps = max(1, day_count / 180)

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=day_count,
        init_func=init,
        interval=1000 / playback_fps,
        blit=False,
    )

    writer = animation.FFMpegWriter(
        fps=playback_fps,
        metadata={"title": "Stock Price Simulation"},
    )

    ani.save(str(VIDEO_FILE), writer=writer)
    plt.close(fig)


def ensure_video():
    """Chỉ tạo video khi cần, tránh làm Render timeout lúc import main.py."""
    global _generation_started

    if VIDEO_FILE.exists() and VIDEO_FILE.stat().st_size > 0:
        return

    with _generation_lock:
        if VIDEO_FILE.exists() and VIDEO_FILE.stat().st_size > 0:
            return

        _generation_started = True
        try:
            generate_video()
        finally:
            _generation_started = False


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    status = "Đã có video." if VIDEO_FILE.exists() else "Video chưa được tạo."

    return f"""
    <!doctype html>
    <html lang="vi">
    <head>
        <meta charset="utf-8">
        <title>GKIPython</title>
        <style>
            body {{
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 40px;
            }}
            a {{
                color: #66ccff;
                font-size: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Mô phỏng giá chứng khoán</h1>
        <p>{status}</p>
        <p><a href="/video">Xem / tải video dothi.mp4</a></p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "video_exists": VIDEO_FILE.exists(),
        "video_generating": _generation_started,
    }


@app.get("/video")
def video():
    ensure_video()

    return FileResponse(
        VIDEO_FILE,
        media_type="video/mp4",
        filename="dothi.mp4",
    )
