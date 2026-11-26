"""Script chia_du_lieu.py
Thuc hien:
1. Chia tap train / val / test theo ty le 70 / 15 / 15 voi seed = 42.
2. Chia THEO NHOM (nhom) va phan tang theo nguon / trai de chong ro ri du lieu.
3. Cat anh tu ANH GOC theo dung quy cach muc 4.3 cua plan_data_posture.md (ham cat_vuong).
4. Bo sung khoang 15% anh train voi hop xe dich ngau nhien +-5..10%.
5. Luu split.csv va xuat anh vao data/data_posture_final_ok_final/{train,val,test}/{dung,khong_dung}/*.jpg.
"""

import os
import cv2
import glob
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

NHAN_CSV = BASE_DIR / "nhan.csv"
SPLIT_CSV = BASE_DIR / "split.csv"


def cat_vuong(khung, hop, kich_thuoc=224, noi=0.10, canh_trung_gian=None):
    """
    Quy cach cat anh train (bat buoc theo muc 4.3):
    - khung: anh goc BGR; hop: (x1, y1, x2, y2) pixel.
    - RGB_9579: noi=0, canh_trung_gian=280.
    """
    x1, y1, x2, y2 = map(float, hop)
    w, h = x2 - x1, y2 - y1
    canh = max(w, h) * (1 + 2 * noi)                              # buoc 2: noi 10% moi ben (canh * 1.2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(round(cx - canh / 2)), int(round(cy - canh / 2))
    c, d = a + int(round(canh)), b + int(round(canh))
    H, W = khung.shape[:2]
    dem = (max(0, -b), max(0, d - H), max(0, -a), max(0, c - W))  # tren, duoi, trai, phai
    vung = khung[max(0, b):min(H, d), max(0, a):min(W, c)]
    if any(dem):                                                   # buoc 3: dem phan chieu BORDER_REFLECT_101
        vung = cv2.copyMakeBorder(vung, *dem, cv2.BORDER_REFLECT_101)
    if h > w:                                                      # buoc 4: xoay 90 do chieu kim dong ho neu hop doc
        vung = cv2.rotate(vung, cv2.ROTATE_90_CLOCKWISE)
    if canh_trung_gian and vung.shape[0] > canh_trung_gian:        # buoc 5 (RGB_9579): thu nho truoc ve 280x280
        vung = cv2.resize(vung, (canh_trung_gian, canh_trung_gian), interpolation=cv2.INTER_AREA)
    nd = cv2.INTER_AREA if vung.shape[0] > kich_thuoc else cv2.INTER_LINEAR
    return cv2.resize(vung, (kich_thuoc, kich_thuoc), interpolation=nd)  # buoc 6: ve 224x224


def load_original_image_map():
    """Tạo từ điển ánh xạ đường dẫn ảnh gốc."""
    pigdetect_map = {Path(p).name: p for p in glob.glob(str(PROJECT_ROOT / "data/data_pig_det_final/images/*/*.*"))}
    pigtrack_map = {(Path(p).parent.parent.name, Path(p).name): p for p in glob.glob(str(PROJECT_ROOT / "data_goc/pigtrack/pigtrack*/img1/*.jpg"))}
    rgb_map = {Path(p).name: p for p in glob.glob(str(PROJECT_ROOT / "data/data_cat_lung_tinh_kg/RGB_9579/*/*/*.*"))}
    return pigdetect_map, pigtrack_map, rgb_map


def split_groups(sub_df, seed=42):
    """
    Chia các nhóm (nhom) thành train (70%), val (15%), test (15%)
    dựa trên thuật toán cân bằng tải (greedy bin-packing).
    """
    r = np.random.RandomState(seed)
    nhom_stats = []
    for nhom, grp in sub_df.groupby("nhom"):
        nhom_stats.append({
            "nhom": nhom,
            "total": len(grp),
        })
    nhom_df = pd.DataFrame(nhom_stats)
    nhom_df = nhom_df.sample(frac=1.0, random_state=r).reset_index(drop=True)

    total = len(sub_df)
    target_train = 0.70 * total
    target_val = 0.15 * total
    target_test = 0.15 * total

    counts = {"train": 0, "val": 0, "test": 0}
    splits = {}

    for _, row in nhom_df.iterrows():
        nh = row["nhom"]
        cnt = row["total"]

        deficits = {
            "train": (target_train - counts["train"]) / max(target_train, 1),
            "val": (target_val - counts["val"]) / max(target_val, 1),
            "test": (target_test - counts["test"]) / max(target_test, 1),
        }
        best_split = max(deficits.keys(), key=lambda s: deficits[s])
        splits[nh] = best_split
        counts[best_split] += cnt

    return splits


def main():
    print("=" * 80)
    print("BẮT ĐẦU CHIA TẬP VÀ CẮT ẢNH (chia_du_lieu.py)")
    print("=" * 80)

    if not NHAN_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy: {NHAN_CSV}")

    df = pd.read_csv(NHAN_CSV)
    print(f"Tổng số mẫu trong nhan.csv: {len(df)}")

    # 1. Phân chia tập theo nhóm với seed = 42
    group_splits = {}

    # Phân tầng theo trại cho PigDetect
    for farm in ["alameer", "bergamini", "danuma", "psota"]:
        farm_df = df[(df["nguon"] == "pigdetect") & (df["trai"] == farm)]
        if len(farm_df) > 0:
            res = split_groups(farm_df, seed=42)
            group_splits.update(res)

    # Phân tầng cho PigTrack theo chuỗi video (mỗi chuỗi là 1 nhóm)
    pt_df = df[df["nguon"] == "pigtrack"]
    if len(pt_df) > 0:
        res = split_groups(pt_df, seed=42)
        group_splits.update(res)

    # Nguồn RGB_9579 (nếu có)
    rgb_df = df[df["nguon"] == "rgb_9579"]
    if len(rgb_df) > 0:
        res = split_groups(rgb_df, seed=42)
        group_splits.update(res)

    df["tap"] = df["nhom"].map(group_splits)

    print("\nKết quả chia tập tổng quan:")
    print(df["tap"].value_counts().to_string())
    print("\nTỷ lệ %:")
    print((df["tap"].value_counts(normalize=True) * 100).round(1).to_string())

    print("\nPhân bố nhãn theo từng tập:")
    print(pd.crosstab(df["tap"], df["nhan"], margins=True).to_string())

    # 2. Chuẩn bị thư mục xuất ảnh theo chuẩn YOLO classification:
    # data_posture_final_ok_final/{train,val,test}/{dung,khong_dung}/
    splits = ["train", "val", "test"]
    classes = ["dung", "khong_dung"]

    for sp in splits:
        for cls in classes:
            dir_path = BASE_DIR / sp / cls
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)

    # 3. Nạp bản đồ ảnh gốc và thực hiện cắt ảnh
    print("\n--- BẮT ĐẦU CẮT ẢNH TỪ ẢNH GỐC (cat_vuong) ---")
    pigdetect_map, pigtrack_map, rgb_map = load_original_image_map()

    img_cache = {}
    split_records = []
    success_count = 0

    for idx, row in df.iterrows():
        nguon = row["nguon"]
        trai = row["trai"]
        anh_goc = row["anh_goc"]
        nhan = row["nhan"]
        tap = row["tap"]
        hop = (row["x1"], row["y1"], row["x2"], row["y2"])
        file_name = row["file"]

        if nguon == "pigdetect":
            p = pigdetect_map.get(anh_goc)
        elif nguon == "pigtrack":
            p = pigtrack_map.get((trai, anh_goc))
        elif nguon == "rgb_9579":
            p = rgb_map.get(anh_goc)
        else:
            p = None

        if p is None or not os.path.exists(p):
            print(f"[CẢNH BÁO] Không tìm thấy ảnh gốc: {nguon}/{trai}/{anh_goc}")
            continue

        if p not in img_cache:
            if len(img_cache) > 40:
                img_cache.pop(next(iter(img_cache)))
            img_bgr = cv2.imread(p)
            img_cache[p] = img_bgr
        else:
            img_bgr = img_cache[p]

        if img_bgr is None:
            continue

        noi = 0.0 if nguon == "rgb_9579" else 0.10
        canh_tg = 280 if nguon == "rgb_9579" else None
        crop = cat_vuong(img_bgr, hop, kich_thuoc=224, noi=noi, canh_trung_gian=canh_tg)

        out_path = BASE_DIR / tap / nhan / file_name
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        success_count += 1

        rec = row.to_dict()
        rec["tap"] = tap
        rec["file_luu"] = file_name
        rec["loai_hop"] = "chuan"
        split_records.append(rec)

    print(f"Đã cắt và lưu thành công {success_count} ảnh tiêu chuẩn.")

    # 4. Bổ sung khoảng 15% ảnh train với hộp xê dịch ngẫu nhiên +-5..10%
    print("\n--- BỔ SUNG 15% ẢNH TRAIN VỚI HỘP XÊ DỊCH (JITTERED BOX) ---")
    train_indices = [i for i, r in enumerate(split_records) if r["tap"] == "train"]
    rng = np.random.RandomState(42)
    n_jitter = int(round(len(train_indices) * 0.15))
    selected_jitter_indices = rng.choice(train_indices, size=n_jitter, replace=False)
    print(f"Tổng số ảnh train ban đầu: {len(train_indices)}")
    print(f"Số ảnh train được xê dịch hộp (15%): {n_jitter}")

    jitter_count = 0
    for idx_in_rec in selected_jitter_indices:
        r = split_records[idx_in_rec]
        nguon = r["nguon"]
        trai = r["trai"]
        anh_goc = r["anh_goc"]
        nhan = r["nhan"]
        tap = r["tap"]
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        w = x2 - x1
        h = y2 - y1

        # Xê dịch ngẫu nhiên +-5% đến +-10% theo cả 2 trục
        shift_x = rng.uniform(0.05, 0.10) * w * rng.choice([-1, 1])
        shift_y = rng.uniform(0.05, 0.10) * h * rng.choice([-1, 1])

        hop_shift = (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)

        if nguon == "pigdetect":
            p = pigdetect_map.get(anh_goc)
        elif nguon == "pigtrack":
            p = pigtrack_map.get((trai, anh_goc))
        elif nguon == "rgb_9579":
            p = rgb_map.get(anh_goc)
        else:
            p = None

        if p not in img_cache:
            img_bgr = cv2.imread(p)
            img_cache[p] = img_bgr
        else:
            img_bgr = img_cache[p]

        if img_bgr is None:
            continue

        noi = 0.0 if nguon == "rgb_9579" else 0.10
        canh_tg = 280 if nguon == "rgb_9579" else None
        crop_jitter = cat_vuong(img_bgr, hop_shift, kich_thuoc=224, noi=noi, canh_trung_gian=canh_tg)

        # Đặt tên file phái sinh: <stem>_jitter.jpg
        orig_file = r["file"]
        stem, ext = os.path.splitext(orig_file)
        jitter_file = f"{stem}_jitter{ext}"

        out_path = BASE_DIR / tap / nhan / jitter_file
        cv2.imwrite(str(out_path), crop_jitter, [cv2.IMWRITE_JPEG_QUALITY, 95])
        jitter_count += 1

        rec = dict(r)
        rec["file_luu"] = jitter_file
        rec["loai_hop"] = "xe_dich"
        rec["x1"] = hop_shift[0]
        rec["y1"] = hop_shift[1]
        rec["x2"] = hop_shift[2]
        rec["y2"] = hop_shift[3]
        rec["hop"] = f"({int(round(hop_shift[0]))}, {int(round(hop_shift[1]))}, {int(round(hop_shift[2]))}, {int(round(hop_shift[3]))})"
        split_records.append(rec)

    print(f"Đã tạo và lưu thành công {jitter_count} ảnh train xê dịch hộp.")

    # 5. Lưu toàn bộ split.csv
    df_split = pd.DataFrame(split_records)
    df_split.to_csv(SPLIT_CSV, index=False)
    print(f"\nĐã xuất file log chia tập: {SPLIT_CSV} ({len(df_split)} dòng)")

    # Thống kê tổng kết số lượng file trong từng thư mục
    print("\n" + "=" * 80)
    print("TỔNG KẾT DỮ LIỆU ĐÃ CẮT VÀO CẤU TRÚC YOLO CLASSIFICATION")
    print("=" * 80)
    for sp in splits:
        for cls in classes:
            folder = BASE_DIR / sp / cls
            count = len(list(folder.glob("*.jpg")))
            print(f"  {sp:5s} / {cls:10s}: {count:5d} ảnh")

    print(f"\nTổng ảnh toàn bộ bộ dữ liệu: {len(df_split)} ảnh.")
    print("=" * 80)


if __name__ == "__main__":
    main()
