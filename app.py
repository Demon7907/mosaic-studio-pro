import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageChops, ImageFilter
import numpy as np
import random
import hashlib
from scipy.spatial import KDTree
from datetime import datetime
from io import BytesIO
import tempfile
import os
import uuid

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mosaic Studio Pro",
    page_icon="🖼️",
    layout="wide"
)

# ---------------- CUSTOM WEBSITE UI ----------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(168,85,247,0.18));
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.1rem;
        color: #cbd5e1;
        max-width: 780px;
    }

    .glass-card {
        padding: 1.4rem;
        border-radius: 20px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 1.2rem;
    }

    .step-title {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .muted-text {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 16px;
        background: rgba(30,41,59,0.8);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .metric-card h3 {
        color: #ffffff;
        margin: 0;
        font-size: 1.5rem;
    }

    .metric-card p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
    }

    div.stButton > button {
        border-radius: 14px;
        height: 3rem;
        font-weight: 700;
        border: none;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        box-shadow: 0 10px 20px rgba(37,99,235,0.25);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #6d28d9);
        color: white;
        border: none;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #0f172a);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    [data-testid="stFileUploader"] {
        border-radius: 18px;
    }

    [data-testid="stStatusWidget"] {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        background-color: rgba(15, 23, 42, 0.96) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        padding: 22px 44px !important;
        border-radius: 18px !important;
        z-index: 99999 !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.55) !important;
    }

    [data-testid="stStatusWidget"] label {
        font-size: 1.1rem !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

Image.MAX_IMAGE_PIXELS = None


# ---------------- CORE FUNCTIONS ----------------

@st.cache_data(show_spinner=False)
def process_tile_library(file_items, tile_size):
    processed_tiles = {}

    for item in file_items:
        try:
            item.seek(0)
            sample_bytes = item.read(8192)
            file_hash_prefix = hashlib.md5(sample_bytes).hexdigest()

            file_id = f"{item.name}_{item.size}_{file_hash_prefix}"

            if file_id in processed_tiles:
                continue

            item.seek(0)
            img = Image.open(item).convert("RGB")

            tile = ImageOps.fit(
                img,
                (tile_size, tile_size),
                Image.Resampling.LANCZOS
            )

            avg_color = np.array(tile).mean(axis=(0, 1))
            stddev = ImageStat.Stat(tile.convert("L")).stddev[0]

            processed_tiles[file_id] = {
                "img": tile,
                "color": avg_color,
                "stddev": stddev
            }

        except Exception:
            continue

    return list(processed_tiles.values())


@st.cache_resource(show_spinner=False)
def build_kdtree(color_array):
    return KDTree(color_array)


def apply_luminosity_blend(mosaic_img, target_img):
    mosaic_rgb = mosaic_img.convert("RGB")
    target_rgb = target_img.convert("RGB")

    multiplied = ImageChops.multiply(mosaic_rgb, target_rgb)

    return Image.blend(mosaic_rgb, multiplied, alpha=0.6)


# ---------------- HEADER ----------------

st.markdown(
    """
    <div class="hero">
        <h1>🖼️ Mosaic Studio Pro</h1>
        <p>
            Create high-resolution photo mosaics from your own image library.
            Designed for portraits, posters, gifts, wallpapers, and print-ready artwork.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## ⚙️ Studio Controls")
    st.caption("Adjust quality, density, blending, and export settings.")

    tile_res = st.select_slider(
        "Tile Resolution (px)",
        options=[16, 32, 64, 128],
        value=64,
        help="Size of each tiny photo tile."
    )

    density = st.slider(
        "Grid Density Across",
        min_value=40,
        max_value=300,
        value=150,
        help="Higher value gives more detail but creates a larger file."
    )

    st.divider()

    st.markdown("### ✨ Sharpening & Blending")

    target_sharpness = st.slider(
        "Pre-Sharpen Main Photo",
        min_value=0,
        max_value=300,
        value=150,
        help="Improves facial edges before mosaic generation."
    )

    random_k = st.slider(
        "Texture Variety",
        min_value=1,
        max_value=10,
        value=2,
        help="Lower value gives better color accuracy. Higher value gives more variety."
    )

    blend_mode = st.radio(
        "Blending Method",
        ["Luminosity Multiply (Sharp)", "Alpha Overlay"]
    )

    alpha_mix = st.slider(
        "Overlay Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        help="Adds original image visibility over the mosaic."
    )

    export_fmt = st.selectbox(
        "Export Format",
        ["JPEG", "PNG", "TIFF"]
    )


# ---------------- LOAD TILE LIBRARY ----------------

st.markdown(
    """
    <div class="glass-card">
        <div class="step-title">Step 1 · Upload Your Tile Images</div>
        <div class="muted-text">
            Select multiple images from your device. These photos will form the small mosaic tiles.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

files_to_process = []
file_hash = None

uploaded = st.file_uploader(
    "Upload tile photos (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded:
    files_to_process = uploaded

    file_hash = hashlib.md5(
        "".join([
            f"{f.name}{f.size}{f.type}"
            for f in uploaded
        ]).encode()
    ).hexdigest()

    st.success(f"✅ {len(files_to_process)} images uploaded successfully!")

else:
    st.info("👆 Upload at least 20–50 images for best mosaic quality.")


# ---------------- RESET SESSION ----------------

if "current_hash" not in st.session_state or st.session_state.current_hash != file_hash:
    st.session_state.current_hash = file_hash

    for key in ["top_picks", "active_target", "tiles", "built_res"]:
        if key in st.session_state:
            del st.session_state[key]


# ---------------- PROCESS TILE LIBRARY ----------------

if files_to_process:
    if "tiles" not in st.session_state or st.session_state.get("built_res") != tile_res:
        with st.spinner("Analyzing your photo library..."):
            st.session_state.tiles = process_tile_library(files_to_process, tile_res)
            st.session_state.built_res = tile_res

    tiles = st.session_state.tiles

    if not tiles:
        st.error("No valid images found.")

    else:
        st.success(f"✅ Processed {len(tiles)} usable tile images.")

        if "top_picks" not in st.session_state:
            scored = sorted(
                tiles,
                key=lambda x: x["stddev"],
                reverse=True
            )
            st.session_state.top_picks = scored[:3]

        selection_container = st.expander(
            "Step 2 · Choose Your Main Portrait",
            expanded=("active_target" not in st.session_state)
        )

        with selection_container:
            cols = st.columns(3)

            for i, pick in enumerate(st.session_state.top_picks):
                with cols[i]:
                    st.image(pick["img"], use_container_width=True)

                    if st.button(f"Use Photo #{i + 1}", key=f"pick_{i}"):
                        st.session_state.active_target = pick["img"]
                        st.rerun()

            st.divider()

            custom_target_file = st.file_uploader(
                "🎯 Or upload a specific main photo",
                type=["jpg", "jpeg", "png", "webp"],
                key="custom_target"
            )

            if custom_target_file:
                custom_img = Image.open(custom_target_file).convert("RGB")

                col_preview, col_btn = st.columns([1, 4])

                with col_preview:
                    st.image(custom_img, use_container_width=True)

                with col_btn:
                    if st.button("✅ Set as Main Portrait", type="primary"):
                        st.session_state.active_target = custom_img
                        st.rerun()


# ---------------- GENERATOR ----------------

if "active_target" in st.session_state and "tiles" in st.session_state:
    st.divider()

    st.markdown(
        """
        <div class="glass-card">
            <div class="step-title">Step 3 · Generate Your Mosaic</div>
            <div class="muted-text">
                Preview the selected portrait, check output size, then generate the final master file.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_tgt, col_info = st.columns([1, 5])

    preview_target = st.session_state.active_target.copy()

    if target_sharpness > 0:
        preview_target = preview_target.filter(
            ImageFilter.UnsharpMask(
                radius=2,
                percent=target_sharpness,
                threshold=3
            )
        )

    with col_tgt:
        st.caption("Active Portrait Preview")
        st.image(preview_target, use_container_width=True)

    with col_info:
        st.success("Ready to generate mosaic.")

        target = st.session_state.active_target.convert("RGB")
        w, h = target.size

        grid_h_preview = max(1, int(density * (h / w)))
        full_w_preview = density * tile_res
        full_h_preview = grid_h_preview * tile_res

        estimated_mp = (full_w_preview * full_h_preview) / 1_000_000

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{full_w_preview}</h3>
                    <p>Width px</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{full_h_preview}</h3>
                    <p>Height px</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{estimated_mp:.1f}</h3>
                    <p>Megapixels</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        if estimated_mp > 100:
            st.warning(
                "⚠️ This is a very large output. Generation, preview, and download may use high RAM."
            )

        generate = st.button(
            "🚀 Generate High-Density Mosaic",
            type="primary",
            use_container_width=True
        )

    if generate:
        tiles = st.session_state.tiles

        tile_colors = np.array([t["color"] for t in tiles])
        tree = build_kdtree(tile_colors)

        target = st.session_state.active_target.convert("RGB")

        w, h = target.size

        grid_h = max(1, int(density * (h / w)))

        full_w = density * tile_res
        full_h = grid_h * tile_res

        target_res = target.resize(
            (full_w, full_h),
            Image.Resampling.LANCZOS
        )

        if target_sharpness > 0:
            target_res = target_res.filter(
                ImageFilter.UnsharpMask(
                    radius=2,
                    percent=target_sharpness,
                    threshold=3
                )
            )

        target_rgb = np.array(target_res)

        target_blocks = target_rgb.reshape(
            grid_h,
            tile_res,
            density,
            tile_res,
            3
        ).mean(axis=(1, 3))

        placed_indices = np.zeros((grid_h, density), dtype=int)

        temp_dir = tempfile.gettempdir()
        memmap_path = os.path.join(
            temp_dir,
            f"mosaic_engine_cache_{uuid.uuid4().hex}.dat"
        )

        canvas_mem = np.memmap(
            memmap_path,
            dtype="uint8",
            mode="w+",
            shape=(full_h, full_w, 3)
        )

        status_text = st.empty()
        progress_bar = st.empty()

        status_text.info(
            f"Rendering {full_w} × {full_h} mosaic directly to temporary storage..."
        )

        pb = progress_bar.progress(0)

        for y in range(grid_h):
            for x in range(density):
                reg_color = target_blocks[y, x]

                _, idxs = tree.query(
                    reg_color,
                    k=min(random_k + 4, len(tiles))
                )

                idxs = np.atleast_1d(idxs)

                target_box = (
                    x * tile_res,
                    y * tile_res,
                    (x + 1) * tile_res,
                    (y + 1) * tile_res
                )

                target_crop = target_res.crop(target_box)

                neighbors = set()

                for dy, dx in [(0, -1), (-1, -1), (-1, 0), (-1, 1)]:
                    ny = y + dy
                    nx = x + dx

                    if 0 <= ny < grid_h and 0 <= nx < density:
                        neighbors.add(placed_indices[ny, nx])

                candidates = [i for i in idxs if i not in neighbors]

                if not candidates:
                    candidates = [idxs[0]]

                best_idx = int(random.choice(candidates[:random_k]))
                placed_indices[y, x] = best_idx

                raw_tile = tiles[best_idx]["img"]

                if raw_tile.size != target_crop.size:
                    raw_tile = raw_tile.resize(
                        target_crop.size,
                        Image.Resampling.LANCZOS
                    )

                if blend_mode == "Luminosity Multiply (Sharp)":
                    blended = apply_luminosity_blend(raw_tile, target_crop)

                    if alpha_mix > 0:
                        final_tile = Image.blend(
                            blended,
                            target_crop,
                            alpha=alpha_mix
                        )
                    else:
                        final_tile = blended

                else:
                    final_tile = Image.blend(
                        raw_tile,
                        target_crop,
                        alpha=alpha_mix
                    )

                canvas_mem[
                    y * tile_res:(y + 1) * tile_res,
                    x * tile_res:(x + 1) * tile_res
                ] = np.array(final_tile)

            canvas_mem.flush()

            if y % max(1, grid_h // 20) == 0 or y == grid_h - 1:
                pb.progress((y + 1) / grid_h)

        status_text.empty()
        progress_bar.empty()

        final_output = Image.fromarray(np.array(canvas_mem).copy())

        st.markdown(
            """
            <div class="glass-card">
                <div class="step-title">Final Output</div>
                <div class="muted-text">
                    Preview your completed mosaic and download the high-resolution master file.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        preview_image = final_output.copy()
        preview_image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)

        st.image(
            preview_image,
            caption="Web Preview - downscaled for browser",
            use_container_width=True
        )

        st.subheader("🔍 1:1 Detail Preview")

        cx = final_output.width // 2
        cy = final_output.height // 2

        sz = min(
            400,
            final_output.width // 2,
            final_output.height // 2
        )

        crop_img = final_output.crop(
            (cx - sz, cy - sz, cx + sz, cy + sz)
        )

        st.image(
            crop_img,
            caption="Central Detail from Master File"
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = export_fmt.lower()

        if export_fmt == "JPEG":
            final_output = final_output.convert("RGB")
            ext = "jpg"

        buf = BytesIO()

        save_kwargs = {
            "format": export_fmt
        }

        if export_fmt == "JPEG":
            save_kwargs["quality"] = 95
            save_kwargs["optimize"] = True

        final_output.save(buf, **save_kwargs)

        st.download_button(
            label="📥 Download High-Res Master File",
            data=buf.getvalue(),
            file_name=f"mosaic_{ts}.{ext}",
            mime=f"image/{ext}"
        )

        del final_output
        del canvas_mem

        try:
            os.remove(memmap_path)
        except Exception:
            pass