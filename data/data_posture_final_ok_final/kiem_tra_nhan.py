"""Script kiem_tra_nhan.py
Kiem tra chat luong du lieu va nhan sau khi nhap tu Roboflow:
1. Kiem tra can bang lop dung / khong_dung theo nguon va cac ca (chen chuc, sat mep, ban dem).
2. Kiem tra ty le RGB_9579 trong lop dung (muc tieu <= 15%).
3. Kiem tra tinh hop le cua toa do hop bounding box.
4. Kiem tra trung lap anh / gan trung lap bang pHash giua cac nhom khac nhau.
"""

import os
import cv2
import glob
import itertools
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

NHAN_CSV = BASE_DIR / "nhan.csv"


def cat_vuong(khung, hop, kich_thuoc=224, noi=0.10, canh_trung_gian=None):
    """Ham cat vuong chuan theo muc 4.3 cua plan_data_posture.md."""
    x1, y1, x2, y2 = map(float, hop)
    w, h = x2 - x1, y2 - y1
    canh = max(w, h) * (1 + 2 * noi)                              # buoc 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(round(cx - canh / 2)), int(round(cy - canh / 2))
    c, d = a + int(round(canh)), b + int(round(canh))
    H, W = khung.shape[:2]
    dem = (max(0, -b), max(0, d - H), max(0, -a), max(0, c - W))  # tren, duoi, trai, phai
    vung = khung[max(0, b):min(H, d), max(0, a):min(W, c)]
    if any(dem):                                                   # buoc 3
        vung = cv2.copyMakeBorder(vung, *dem, cv2.BORDER_REFLECT_101)
    if h > w:                                                      # buoc 4
        vung = cv2.rotate(vung, cv2.ROTATE_90_CLOCKWISE)
    if canh_trung_gian and vung.shape[0] > canh_trung_gian:        # buoc 5 (RGB_9579)
        vung = cv2.resize(vung, (canh_trung_gian, canh_trung_gian), interpolation=cv2.INTER_AREA)
    nd = cv2.INTER_AREA if vung.shape[0] > kich_thuoc else cv2.INTER_LINEAR
    return cv2.resize(vung, (kich_thuoc, kich_thuoc), interpolation=nd)  # buoc 6


def load_original_image_map():
    """Anh xa tu anh goc sang duong dan file thuc te tren o dia."""
    pigdetect_map = {Path(p).name: p for p in glob.glob(str(PROJECT_ROOT / "data/data_pig_det_final/images/*/*.*"))}
    pigtrack_map = {(Path(p).parent.parent.name, Path(p).name): p for p in glob.glob(str(PROJECT_ROOT / "data_goc/pigtrack/pigtrack*/img1/*.jpg"))}
    rgb_map = {Path(p).name: p for p in glob.glob(str(PROJECT_ROOT / "data/data_cat_lung_tinh_kg/RGB_9579/*/*/*.*"))}
    return pigdetect_map, pigtrack_map, rgb_map


def main():
    print("=" * 80)
    print("KIỂM TRA DỮ LIỆU VÀ NHÃN (kiem_tra_nhan.py)")
    print("=" * 80)

    if not NHAN_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {NHAN_CSV}. Vui lòng chạy nhap_roboflow.py trước.")

    df = pd.read_csv(NHAN_CSV)
    print(f"Tổng số mẫu trong nhan.csv: {len(df)}")

    # 1. Kiểm tra hộp bounding box hợp lệ
    print("\n--- [1] KIỂM TRA TÍNH HỢP LỆ CỦA BOUNDING BOX ---")
    invalid_box = df[(df["x1"] >= df["x2"]) | (df["y1"] >= df["y2"])]
    if len(invalid_box) > 0:
        print(f"[CẢNH BÁO] Có {len(invalid_box)} hộp có toạ độ không hợp lệ (x1 >= x2 hoặc y1 >= y2)!")
    else:
        print("Tất cả toạ độ bounding box (x1 < x2, y1 < y2) đều HỢP LỆ 100%.")

    # 2. Kiểm tra tỷ lệ RGB_9579 trong lớp dung (mục tiêu <= 15%)
    print("\n--- [2] KIỂM TRA TỶ LỆ RGB_9579 TRONG LỚP 'DUNG' ---")
    dung_total = (df["nhan"] == "dung").sum()
    rgb_dung = ((df["nhan"] == "dung") & (df["nguon"] == "rgb_9579")).sum()
    ty_le_rgb_dung = (rgb_dung / dung_total * 100) if dung_total > 0 else 0
    print(f"Tổng số mẫu lớp 'dung': {dung_total}")
    print(f"Số mẫu RGB_9579 trong lớp 'dung': {rgb_dung}")
    print(f"Tỷ lệ RGB_9579 / 'dung': {ty_le_rgb_dung:.2f}% (Mục tiêu: <= 15%)")
    if ty_le_rgb_dung <= 15.0:
        print("-> ĐẠT CHỈ TIÊU (<= 15%).")
    else:
        print("-> [CẢNH BÁO] VƯỢT CHỈ TIÊU 15%!")

    # 3. Thống kê cân bằng lớp theo nguồn và trại
    print("\n--- [3] CÂN BẰNG LỚP THEO NGUỒN VÀ TRẠI ---")
    ct_nguon = pd.crosstab(df["nguon"], df["nhan"], margins=True)
    ct_nguon["% dung"] = (ct_nguon["dung"] / ct_nguon["All"] * 100).round(1)
    print(ct_nguon.to_string())

    print("\nChi tiết theo trại / chuỗi:")
    ct_trai = pd.crosstab([df["nguon"], df["trai"]], df["nhan"], margins=True)
    ct_trai["% dung"] = (ct_trai["dung"] / ct_trai["All"] * 100).round(1)
    print(ct_trai.to_string())

    # 4. Thống kê cân bằng lớp theo các ca đặc thù (chen chúc, sát mép, ban đêm, tag khó)
    print("\n--- [4] THỐNG KÊ CÁC CA ĐẶC THÙ (CHỈ TIÊU >= 100 MẪU MỖI LỚP) ---")
    cases = {
        "chen_chuc (IoU >= 0.3)": (df["muc_chen"] == "chen_chuc") | (df["iou_max"] >= 0.3),
        "sat_mep_tuong (sat_mep == 1)": df["sat_mep"] == 1,
        "ban_dem (dem == 1)": df["dem"] == 1,
        "tag_kho (kho == 1)": df["kho"] == 1,
    }

    case_rows = []
    for case_name, mask in cases.items():
        sub = df[mask]
        n_dung = (sub["nhan"] == "dung").sum()
        n_khong_dung = (sub["nhan"] == "khong_dung").sum()
        n_total = len(sub)
        dat = "ĐẠT" if (n_dung >= 100 and n_khong_dung >= 100) else "CHƯA ĐỦ 100/LỚP"
        case_rows.append({
            "Ca": case_name,
            "dung": n_dung,
            "khong_dung": n_khong_dung,
            "Tổng": n_total,
            "Đánh giá": dat,
        })
    df_cases = pd.DataFrame(case_rows)
    print(df_cases.to_string(index=False))

    # 5. Kiểm tra trùng lặp ảnh / gần trùng lặp (pHash) giữa các nhóm (nhom)
    print("\n--- [5] KIỂM TRA TRÙNG LẶP ẢNH (pHash) GIỮA CÁC NHÓM KHÁC NHAU ---")
    print("Đang nạp ảnh gốc và tính pHash cho từng mẫu...")
    pigdetect_map, pigtrack_map, rgb_map = load_original_image_map()

    hashes = []
    sample_records = []
    img_cache = {}

    for idx, row in df.iterrows():
        nguon = row["nguon"]
        trai = row["trai"]
        anh_goc = row["anh_goc"]
        nhom = row["nhom"]
        hop = (row["x1"], row["y1"], row["x2"], row["y2"])

        # Tìm đường dẫn ảnh gốc
        if nguon == "pigdetect":
            p = pigdetect_map.get(anh_goc)
        elif nguon == "pigtrack":
            p = pigtrack_map.get((trai, anh_goc))
        elif nguon == "rgb_9579":
            p = rgb_map.get(anh_goc)
        else:
            p = None

        if p is None or not os.path.exists(p):
            continue

        if p not in img_cache:
            # Giới hạn cache tối đa 50 ảnh để tránh tốn RAM
            if len(img_cache) > 50:
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

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ph = imagehash.phash(pil_img)
        hashes.append(ph)
        sample_records.append((idx, row["file"], nhom, row["nhan"]))

    print(f"Đã tính pHash thành công cho {len(hashes)} ảnh.")

    # So sánh khoảng cách Hamming giữa các cặp khác nhóm
    # Để tối ưu O(N^2), so sánh trực tiếp qua hash bit
    hash_matrix = np.array([h.hash.flatten() for h in hashes], dtype=bool)
    n = len(hash_matrix)
    print("Đang kiểm tra trùng lặp qua ma trận pHash Hamming distance...")

    cross_group_duplicates = []
    # Quét theo từng block 500 dòng để tối ưu bộ nhớ
    block_size = 500
    for i_start in range(0, n, block_size):
        i_end = min(n, i_start + block_size)
        sub_hashes = hash_matrix[i_start:i_end]
        # XOR bitwise để tính Hamming distance: (A != B).sum(axis=1)
        # sub_hashes shape: (K, 64), hash_matrix shape: (N, 64)
        for sub_idx, i in enumerate(range(i_start, i_end)):
            row_bits = sub_hashes[sub_idx]
            diffs = np.count_nonzero(row_bits != hash_matrix[i + 1:], axis=1)
            # Ngưỡng gần trùng lặp pHash: hamming distance <= 2 (rất giống nhau)
            close_indices = np.where(diffs <= 2)[0] + (i + 1)
            for j in close_indices:
                if sample_records[i][2] != sample_records[j][2]: # khác nhóm
                    cross_group_duplicates.append((
                        sample_records[i][1], sample_records[i][2],
                        sample_records[j][1], sample_records[j][2],
                        diffs[j - (i + 1)],
                    ))

    print(f"Số cặp ảnh gần trùng (Hamming <= 2) KHÁC NHÓM: {len(cross_group_duplicates)}")
    if cross_group_duplicates:
        print("[LƯU Ý] Danh sách các cặp ảnh khác nhóm có độ tương đồng rất cao:")
        for r1, nhom1, r2, nhom2, dist in cross_group_duplicates[:10]:
            print(f"  - ({nhom1}) {r1} <-> ({nhom2}) {r2} (dist={dist})")
    else:
        print("-> KHÔNG có cặp ảnh gần trùng lặp nào nằm ở các nhóm khác nhau. ĐẠT CHỐNG RÒ RỈ HOÀN TOÀN!")

    print("\n" + "=" * 80)
    print("HOÀN THÀNH KIỂM TRA NHÃN VÀ DỮ LIỆU.")
    print("=" * 80)


if __name__ == "__main__":
    main()
