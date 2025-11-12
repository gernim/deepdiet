# src/build_indexes.py
from pathlib import Path
import os, sys
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
DATA = REPO_ROOT / "data" / "nutrition5k_dataset"
SPLITS = DATA / "dish_ids" / "splits"
META1 = DATA / "metadata" / "dish_metadata_cafe1.csv"
META2 = DATA / "metadata" / "dish_metadata_cafe2.csv"
OVER  = DATA / "imagery" / "realsense_overhead"
SIDE  = DATA / "imagery" / "side_angles"
INDEX_DIR = REPO_ROOT / "indexes"

def _dbg(msg): print(msg, file=sys.stderr)
_dbg(f"[build_indexes] DATA root: {DATA}")
_dbg(f"[build_indexes] SPLITS dir: {SPLITS}")

OUT = Path("indexes"); OUT.mkdir(exist_ok=True, parents=True)

def read_splits():
    def _read(p): return [l.strip() for l in open(p) if l.strip()]
    return {
        "train": set(_read(SPLITS/"train.txt")),
        "test":  set(_read(SPLITS/"test.txt")),
    }

def read_dish_meta():
    """
    Robustly read Nutrition5k dish metadata.
    Tries headered parse; if columns aren’t found, falls back to headerless
    and treats first 6 columns as: dish_id, cal, mass, fat, carb, protein.
    """
    def _robust_read(p: Path):
        if not p.exists():
            return None
        for kwargs in (
            dict(engine="python", on_bad_lines="skip", dtype=str),
            dict(engine="python", sep=",", on_bad_lines="skip", dtype=str),
            dict(engine="python", sep=",", quotechar='"', on_bad_lines="skip", dtype=str),
        ):
            try:
                return pd.read_csv(p, **kwargs)
            except Exception:
                continue
        return None

    df1 = _robust_read(META1)
    df2 = _robust_read(META2)
    frames = [d for d in (df1, df2) if d is not None]
    if not frames:
        print(f"[build_indexes] Dish metadata not found at {META1} / {META2}", file=sys.stderr)
        return None

    df = pd.concat(frames, ignore_index=True)

    # Try headered parse
    colmap = {c.lower(): c for c in df.columns if isinstance(c, str)}

    def pick(*cands):
        for cand in cands:
            key = str(cand).lower()
            if key in colmap:
                return colmap[key]
        return None

    col_dish = pick("dish_id", "id", "dish")
    col_cal  = pick("total_calories", "calories_total", "total_kcal", "kcal")
    col_mass = pick("total_mass", "mass_total", "mass_g", "total_mass_g")
    col_fat  = pick("total_fat", "fat_total", "fat_g")
    col_carb = pick("total_carb", "carb_total", "carbs_total", "carb_g", "carbs_g")
    col_prot = pick("total_protein", "protein_total", "protein_g")

    header_ok = all(c is not None for c in [col_dish, col_cal, col_mass, col_fat, col_carb, col_prot])

    if header_ok:
        df = df[[col_dish, col_cal, col_mass, col_fat, col_carb, col_prot]].rename(columns={
            col_dish: "dish_id",
            col_cal:  "cal",
            col_mass: "mass",
            col_fat:  "fat",
            col_carb: "carb",
            col_prot: "protein",
        })
    else:
        # Headerless fallback
        def _read_no_header(p: Path):
            try:
                return pd.read_csv(p, header=None, engine="python", on_bad_lines="skip")
            except Exception:
                return None

        frames_nh = []
        for p in (META1, META2):
            if p.exists():
                d = _read_no_header(p)
                if d is not None and not d.empty:
                    frames_nh.append(d)
        if not frames_nh:
            print("[build_indexes] Could not parse dish metadata (no-header fallback failed).", file=sys.stderr)
            return None
        df = pd.concat(frames_nh, ignore_index=True)
        if df.shape[1] < 6:
            print(f"[build_indexes] Dish metadata has only {df.shape[1]} columns; expected ≥ 6.", file=sys.stderr)
            return None
        df = df.iloc[:, :6]
        df.columns = ["dish_id", "cal", "mass", "fat", "carb", "protein"]

    for k in ["cal", "mass", "fat", "carb", "protein"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["dish_id", "cal", "mass", "fat", "carb", "protein"]).reset_index(drop=True)
    return df

def side_frames_index(dish_ids, meta):
    rows = []
    for d in dish_ids:
        fdir = SIDE/d/"frames_sampled5"
        if not fdir.is_dir(): continue
        for img in sorted(fdir.glob("*.jpeg")):
            rows.append({"dish_id": d, "image": str(img.relative_to(DATA))})
    df_img = pd.DataFrame(rows)
    df = df_img.merge(meta, on="dish_id", how="inner")
    return df

def overhead_index(dish_ids, meta):
    rows = []
    for d in dish_ids:
        ddir = OVER/d
        rgb = ddir/"rgb.png"
        depth = ddir/"depth_raw.png"  # 16-bit, units=10000 per meter
        if rgb.is_file() and depth.is_file():
            rows.append({
                "dish_id": d,
                "rgb": str(rgb.relative_to(DATA)),
                "depth_raw": str(depth.relative_to(DATA))
            })
    df_img = pd.DataFrame(rows)
    df = df_img.merge(meta, on="dish_id", how="inner")
    return df

if __name__ == "__main__":
    splits = read_splits()
    meta = read_dish_meta()
    if meta is None:
        print("[build_indexes] No dish metadata available; writing empty indexes.", file=sys.stderr)
        for split in ["train", "test"]:
            pd.DataFrame().to_csv(OUT / f"side_frames_{split}.csv", index=False)
            pd.DataFrame().to_csv(OUT / f"overhead_{split}.csv", index=False)
        sys.exit(0)

    for split in ["train","test"]:
        sf = side_frames_index(splits[split], meta)
        oh = overhead_index(splits[split], meta)
        sf.to_csv(OUT/f"side_frames_{split}.csv", index=False)
        oh.to_csv(OUT/f"overhead_{split}.csv", index=False)
        print(split, len(sf), len(oh))