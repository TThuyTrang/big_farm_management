"""Script nhap_roboflow.py
Doc du lieu xuat tu Roboflow (_roboflow/{train,valid,test}/{dung,khong_dung}),
ghep nhan voi ung_vien.csv va xuat ra nhan.csv.
"""

import os
import re
import glob
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

ROBOFLOW_DIR = BASE_DIR / "_roboflow"
UNG_VIEN_CSV = BASE_DIR / "ung_vien.csv"
OUT_NHAN_CSV = BASE_DIR / "nhan.csv"


def norm_str(s: str) -> str:
    """Chuan hoa ten file de ghep khop 1-1 khong loi giua Roboflow va ung_vien.csv."""
    s = re.sub(r"_jpg\.rf\.[0-9a-fA-F]+", "", s)
    s = re.sub(r"\.rf\.[0-9a-fA-F]+", "", s)
    s = re.sub(r"-+", "-", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()


def main():
    print("=" * 80)
    print("BẮT ĐẦU NHẬP DỮ LIỆU TỪ ROBOFLOW (nhap_roboflow.py)")
    print("=" * 80)

    if not UNG_VIEN_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {UNG_VIEN_CSV}")

    df_ung_vien = pd.read_csv(UNG_VIEN_CSV)
    print(f"Tổng số ứng viên trong {UNG_VIEN_CSV.name}: {len(df_ung_vien)}")

    # Xây dựng bảng tra cứu từ key chuẩn hoá sang index trong df_ung_vien
    norm_to_idx = {}
    for idx, row in df_ung_vien.iterrows():
        k = norm_str(row["file"])
        if k in norm_to_idx:
            print(f"[CẢNH BÁO] Trùng key chuẩn hoá: {k}")
        norm_to_idx[k] = idx

    # Quét toàn bộ ảnh trong _roboflow/{train,valid,test}/{dung,khong_dung}/
    splits = ["train", "valid", "test"]
    classes = ["dung", "khong_dung"]

    matched_records = []
    unmatched_rf = []
    rf_total_images = 0

    for split in splits:
        for cls in classes:
            folder = ROBOFLOW_DIR / split / cls
            if not folder.exists():
                continue
            img_paths = list(folder.glob("*.jpg"))
            for p in img_paths:
                rf_total_images += 1
                fname = p.name
                k = norm_str(fname)
                if k in norm_to_idx:
                    orig_idx = norm_to_idx[k]
                    matched_records.append((orig_idx, cls, fname))
                else:
                    unmatched_rf.append((str(p), fname))

    print(f"Tổng số ảnh có nhãn tìm thấy trong Roboflow: {rf_total_images}")
    print(f"Số ảnh ghép khớp thành công với ung_vien.csv: {len(matched_records)}")
    if unmatched_rf:
        print(f"[CẢNH BÁO] Có {len(unmatched_rf)} ảnh từ Roboflow không khớp ung_vien.csv:")
        for up, un in unmatched_rf[:5]:
            print(f"  - {un}")

    # Tạo DataFrame nhan.csv
    kept_indices = [idx for idx, cls, rf_fname in matched_records]
    df_nhan = df_ung_vien.iloc[kept_indices].copy().reset_index(drop=True)
    df_nhan["nhan"] = [cls for idx, cls, rf_fname in matched_records]
    # Tag 'kho': mặc định 0 nếu không có trong metadata Roboflow
    df_nhan["kho"] = 0
    df_nhan["hop"] = df_nhan.apply(
        lambda r: f"({int(r['x1'])}, {int(r['y1'])}, {int(r['x2'])}, {int(r['y2'])})",
        axis=1,
    )

    # Đảm bảo các cột theo đúng chuẩn plan_data_posture.md:
    # file, nguon, trai, anh_goc, hop, id, nhom, dem, muc_chen, sat_mep, nhan, kho
    # (đồng thời giữ lại x1, y1, x2, y2, iou_max để chia_du_lieu.py dùng thuận tiện)
    std_cols = [
        "file",
        "nguon",
        "trai",
        "anh_goc",
        "x1",
        "y1",
        "x2",
        "y2",
        "hop",
        "id",
        "nhom",
        "dem",
        "iou_max",
        "muc_chen",
        "sat_mep",
        "nhan",
        "kho",
    ]
    df_nhan = df_nhan[std_cols]
    df_nhan.to_csv(OUT_NHAN_CSV, index=False)
    print(f"\nĐã xuất file nhãn: {OUT_NHAN_CSV} ({len(df_nhan)} dòng)")

    # Thống kê số ảnh giữ lại và bị loại theo từng nguồn / trại
    print("\n" + "=" * 80)
    print("THỐNG KÊ GIỮ LẠI VÀ BỊ LOẠI THEO NGUỒN VÀ TRẠI")
    print("=" * 80)

    kept_set = set(kept_indices)
    df_ung_vien["trang_thai"] = [
        "giu_lai" if i in kept_set else "bi_loai" for i in range(len(df_ung_vien))
    ]

    # Thống kê theo nguồn
    print("\n[1] Thống kê theo NGUỒN:")
    stat_nguon = df_ung_vien.groupby(["nguon", "trang_thai"]).size().unstack(fill_value=0)
    if "giu_lai" not in stat_nguon.columns:
        stat_nguon["giu_lai"] = 0
    if "bi_loai" not in stat_nguon.columns:
        stat_nguon["bi_loai"] = 0
    stat_nguon["tong_ung_vien"] = stat_nguon["giu_lai"] + stat_nguon["bi_loai"]
    stat_nguon["ty_le_giu_%"] = (
        stat_nguon["giu_lai"] / stat_nguon["tong_ung_vien"] * 100
    ).round(1)
    print(stat_nguon.to_string())

    # Thống kê theo trại (trên toàn bộ hoặc từng nguồn)
    print("\n[2] Thống kê theo TRẠI / CHUỖI:")
    stat_trai = df_ung_vien.groupby(["nguon", "trai", "trang_thai"]).size().unstack(fill_value=0)
    if "giu_lai" not in stat_trai.columns:
        stat_trai["giu_lai"] = 0
    if "bi_loai" not in stat_trai.columns:
        stat_trai["bi_loai"] = 0
    stat_trai["tong"] = stat_trai["giu_lai"] + stat_trai["bi_loai"]
    stat_trai["ty_le_giu_%"] = (stat_trai["giu_lai"] / stat_trai["tong"] * 100).round(1)
    print(stat_trai.to_string())

    # Thống kê phân bố lớp trong số ảnh giữ lại
    print("\n[3] Phân bố nhãn trong số ảnh giữ lại:")
    print(df_nhan["nhan"].value_counts().to_string())
    print("\nPhân bố nhãn theo nguồn:")
    print(pd.crosstab(df_nhan["nguon"], df_nhan["nhan"], margins=True).to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
