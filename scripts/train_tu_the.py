"""Script train_tu_the.py
Huan luyen va danh gia mo hinh phan loai tu the lon (YOLO11n-cls).
Tuan thu 100% quy cach ky thuat theo plan_data_posture.md:
- Du lieu: data/data_posture_final_ok_final/{train,val,test}
- Mo hinh: yolo11n-cls.pt (imgsz=224, epochs=50, patience=10, auto-detect device)
- Luu trong so tot nhat: models/Pig_Posture/pig_posture.pt
- Danh gia chi tiet tren tap test:
  + Accuracy test tong the va Accuracy test khong tinh RGB_9579.
  + Accuracy rieng cho tung ca: chen chuc, sat mep/tuong, ban dem, tag kho.
  + Ty le khong_dung bi doan nham thanh dung o nguong P(dung) >= 0.8 (muc tieu <= 2%).
  + Recall lop dung o nguong 0.8 (muc tieu >= 0.70).
  + Liet ke danh sach anh val/test bi model doan nguoc nhan voi do tin cay >= 0.9.
  + Xuat bao cao danh gia ra models/Pig_Posture/danh_gia_tu_the.md.
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train và đánh giá mô hình tư thế lợn YOLO11n-cls.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data/data_posture_final_ok_final"),
        help="Đường dẫn đến thư mục dữ liệu phân loại.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-cls.pt",
        help="Trọng số khởi tạo mô hình (yolo11n-cls.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số epoch tối đa (mặc định: 50).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience dừng sớm (mặc định: 10).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Kích thước ảnh đầu vào (mặc định: 224).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (mặc định: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Thiết bị chạy (auto-detect: '0' nếu có CUDA, else 'cpu').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Số worker nạp dữ liệu.",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=str(PROJECT_ROOT / "models/Pig_Posture/pig_posture.pt"),
        help="Đường dẫn lưu trọng số tốt nhất.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Chỉ đánh giá mô hình đã có tại output_model mà không train lại.",
    )
    return parser.parse_args()


def evaluate_split(model, split_dir, split_name, df_meta, dung_idx, khong_dung_idx):
    """Đánh giá mô hình trên một tập (val hoặc test) và ghi nhận xác suất dự đoán."""
    records = []
    classes = ["dung", "khong_dung"]

    meta_lookup = {}
    for _, row in df_meta.iterrows():
        # Tra cứu theo file_luu hoặc file
        if "file_luu" in row and pd.notna(row["file_luu"]):
            meta_lookup[row["file_luu"]] = row
        meta_lookup[row["file"]] = row

    for cls in classes:
        folder = Path(split_dir) / cls
        if not folder.exists():
            continue
        img_paths = list(folder.glob("*.jpg"))
        for p in img_paths:
            fname = p.name
            meta = meta_lookup.get(fname, None)

            # Dự đoán với mô hình
            res = model.predict(source=str(p), imgsz=224, verbose=False)[0]
            probs = res.probs.data.cpu().numpy()

            p_dung = float(probs[dung_idx])
            p_khong_dung = float(probs[khong_dung_idx])
            pred_cls = "dung" if p_dung >= 0.5 else "khong_dung"
            conf = max(p_dung, p_khong_dung)

            rec = {
                "file": fname,
                "split": split_name,
                "true_label": cls,
                "pred_label": pred_cls,
                "p_dung": p_dung,
                "p_khong_dung": p_khong_dung,
                "confidence": conf,
                "is_correct": int(pred_cls == cls),
            }

            if meta is not None:
                rec["nguon"] = meta.get("nguon", "unknown")
                rec["trai"] = meta.get("trai", "unknown")
                rec["nhom"] = meta.get("nhom", "unknown")
                rec["dem"] = int(meta.get("dem", 0))
                rec["sat_mep"] = int(meta.get("sat_mep", 0))
                rec["muc_chen"] = meta.get("muc_chen", "")
                rec["iou_max"] = float(meta.get("iou_max", 0.0))
                rec["kho"] = int(meta.get("kho", 0))
            else:
                rec["nguon"] = "unknown"
                rec["trai"] = "unknown"
                rec["nhom"] = "unknown"
                rec["dem"] = 0
                rec["sat_mep"] = 0
                rec["muc_chen"] = ""
                rec["iou_max"] = 0.0
                rec["kho"] = 0

            records.append(rec)

    return pd.DataFrame(records)


def main():
    args = parse_args()
    data_dir = Path(args.data)
    out_model_path = Path(args.output_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Tự nhận diện thiết bị
    if not args.device:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cpu":
        num_threads = min(12, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"Thiết lập PyTorch CPU threads: {num_threads}")

    print("=" * 80)
    print("QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH TƯ THẾ (train_tu_the.py)")
    print("=" * 80)
    print(f"Thư mục dữ liệu : {data_dir}")
    print(f"Thiết bị        : {device} (CUDA: {torch.cuda.is_available()})")
    print(f"Epochs          : {args.epochs} | Patience: {args.patience} | Imgsz: {args.imgsz} | Batch: {args.batch}")
    print(f"File model đích : {out_model_path}")
    print("=" * 80)

    # 1. Huấn luyện mô hình
    if not args.eval_only:
        print("\n--- BẮT ĐẦU HUẤN LUYỆN YOLO11n-cls ---")
        model = YOLO(args.model)
        train_results = model.train(
            data=str(data_dir),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=str(PROJECT_ROOT / "runs/pig_posture"),
            name="train",
            exist_ok=True,
            cache=True,
            plots=True,
            verbose=True,
        )

        # Lưu best.pt ra out_model_path
        best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy(best_pt, out_model_path)
            print(f"\n-> Đã lưu trọng số tốt nhất vào: {out_model_path}")
        else:
            print(f"[CẢNH BÁO] Không tìm thấy file {best_pt}, kiểm tra runs directory.")
    else:
        print(f"\n[CHẾ ĐỘ CHỈ ĐÁNH GIÁ] Nạp mô hình từ: {out_model_path}")

    # 2. Đánh giá chi tiết trên tập test và val
    print("\n" + "=" * 80)
    print("BẮT ĐẦU ĐÁNH GIÁ CHI TIẾT MÔ HÌNH (THEO MỤC 5 PLAN_DATA_POSTURE.MD)")
    print("=" * 80)

    if not out_model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {out_model_path}")

    eval_model = YOLO(str(out_model_path))
    class_names = eval_model.names
    print(f"Class mapping: {class_names}")

    dung_idx = [k for k, v in class_names.items() if v == "dung"][0]
    khong_dung_idx = [k for k, v in class_names.items() if v == "khong_dung"][0]

    # Nạp metadata từ split.csv
    split_csv = data_dir / "split.csv"
    if split_csv.exists():
        df_meta = pd.read_csv(split_csv)
    else:
        df_meta = pd.DataFrame()

    # Đánh giá tập test
    test_dir = data_dir / "test"
    df_test_res = evaluate_split(eval_model, test_dir, "test", df_meta, dung_idx, khong_dung_idx)
    print(f"Đã dự đoán xong trên tập TEST ({len(df_test_res)} ảnh).")

    # Đánh giá tập val
    val_dir = data_dir / "val"
    df_val_res = evaluate_split(eval_model, val_dir, "val", df_meta, dung_idx, khong_dung_idx)
    print(f"Đã dự đoán xong trên tập VAL ({len(df_val_res)} ảnh).")

    # 3. Tính toán các chỉ số theo bảng mục 5
    # (1) Accuracy test tổng thể
    total_test = len(df_test_res)
    correct_test = df_test_res["is_correct"].sum()
    acc_total = correct_test / total_test if total_test > 0 else 0.0

    # (2) Accuracy test không tính RGB_9579
    df_test_no_rgb = df_test_res[df_test_res["nguon"] != "rgb_9579"]
    acc_no_rgb = df_test_no_rgb["is_correct"].sum() / len(df_test_no_rgb) if len(df_test_no_rgb) > 0 else 0.0

    # (3) Accuracy riêng cho từng ca:
    # - chen chúc: muc_chen == 'chen_chuc' hoặc iou_max >= 0.3
    sub_chen = df_test_res[(df_test_res["muc_chen"] == "chen_chuc") | (df_test_res["iou_max"] >= 0.3)]
    acc_chen = sub_chen["is_correct"].sum() / len(sub_chen) if len(sub_chen) > 0 else 0.0

    # - sát mép / tường: sat_mep == 1
    sub_mep = df_test_res[df_test_res["sat_mep"] == 1]
    acc_mep = sub_mep["is_correct"].sum() / len(sub_mep) if len(sub_mep) > 0 else 0.0

    # - ban đêm: dem == 1
    sub_dem = df_test_res[df_test_res["dem"] == 1]
    acc_dem = sub_dem["is_correct"].sum() / len(sub_dem) if len(sub_dem) > 0 else 0.0

    # - tag kho: kho == 1
    sub_kho = df_test_res[df_test_res["kho"] == 1]
    acc_kho = sub_kho["is_correct"].sum() / len(sub_kho) if len(sub_kho) > 0 else 0.0

    # (4) Tỷ lệ con không đứng (khong_dung) bị đoán nhầm thành dung ở ngưỡng P(đứng) >= 0.8
    # Mục tiêu: <= 2%
    true_khong_dung = df_test_res[df_test_res["true_label"] == "khong_dung"]
    khong_dung_nham_08 = (true_khong_dung["p_dung"] >= 0.8).sum()
    ty_le_nham_08 = khong_dung_nham_08 / len(true_khong_dung) if len(true_khong_dung) > 0 else 0.0

    # (5) Recall lớp dung ở ngưỡng 0.8 (PigDetect + PigTrack / không tính RGB_9579)
    # Mục tiêu: >= 0.70
    true_dung_non_rgb = df_test_res[(df_test_res["true_label"] == "dung") & (df_test_res["nguon"] != "rgb_9579")]
    dung_dat_08 = (true_dung_non_rgb["p_dung"] >= 0.8).sum()
    recall_dung_08 = dung_dat_08 / len(true_dung_non_rgb) if len(true_dung_non_rgb) > 0 else 0.0

    # In bảng kết quả so sánh với mục tiêu
    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH VÀ SO SÁNH VỚI MỤC TIÊU")
    print("=" * 80)
    print(f"{'Chỉ số':<45} | {'Bộ cũ':<12} | {'Mục tiêu bộ mới':<18} | {'Kết quả đạt được':<15}")
    print("-" * 95)
    print(f"{'Accuracy test (tất cả)':<45} | {'0.909':<12} | {'>= 0.92':<18} | {acc_total:.4f} ({correct_test}/{total_test})")
    print(f"{'Accuracy test không tính RGB_9579':<45} | {'0.887':<12} | {'>= 0.90':<18} | {acc_no_rgb:.4f} ({df_test_no_rgb['is_correct'].sum()}/{len(df_test_no_rgb)})")
    print(f"{'Accuracy riêng ca chen chúc':<45} | {'Chưa đo':<12} | {'>= 0.85':<18} | {acc_chen:.4f} ({sub_chen['is_correct'].sum()}/{len(sub_chen)})")
    print(f"{'Accuracy riêng ca sát mép/tường':<45} | {'Chưa đo':<12} | {'>= 0.85':<18} | {acc_mep:.4f} ({sub_mep['is_correct'].sum()}/{len(sub_mep)})")
    print(f"{'Accuracy riêng ca ban đêm':<45} | {'Chưa đo':<12} | {'>= 0.85':<18} | {acc_dem:.4f} ({sub_dem['is_correct'].sum()}/{len(sub_dem)})")
    if len(sub_kho) > 0:
        print(f"{'Accuracy riêng ca tag kho':<45} | {'Chưa đo':<12} | {'>= 0.85':<18} | {acc_kho:.4f} ({sub_kho['is_correct'].sum()}/{len(sub_kho)})")
    else:
        print(f"{'Accuracy riêng ca tag kho':<45} | {'Chưa đo':<12} | {'>= 0.85':<18} | N/A (0 mẫu)")
    print(f"{'Không đứng bị nhầm P(đứng) >= 0.8':<45} | {'3.4%':<12} | {'<= 2%':<18} | {ty_le_nham_08 * 100:.2f}% ({khong_dung_nham_08}/{len(true_khong_dung)})")
    print(f"{'Recall dung ở ngưỡng 0.8 (không RGB)':<45} | {'0.625':<12} | {'>= 0.70':<18} | {recall_dung_08:.4f} ({dung_dat_08}/{len(true_dung_non_rgb)})")
    print("=" * 80)

    # 4. Liệt kê danh sách ảnh val / test bị đoán ngược với độ tin cậy >= 0.9
    print("\n--- SOÁT LỖI NHÃN: ẢNH VAL/TEST BỊ ĐOÁN NGƯỢC NHÃN VỚI CONFIDENCE >= 0.90 ---")
    df_all_eval = pd.concat([df_val_res, df_test_res], ignore_index=True)
    high_conf_errors = df_all_eval[(df_all_eval["is_correct"] == 0) & (df_all_eval["confidence"] >= 0.90)]

    print(f"Tổng số ảnh bị đoán ngược với độ tin cậy >= 0.90: {len(high_conf_errors)}")
    if len(high_conf_errors) > 0:
        print(high_conf_errors[["split", "file", "true_label", "pred_label", "confidence", "nguon", "trai"]].to_string())
        # Lưu ra CSV để phục vụ soát lỗi
        err_csv = out_model_path.parent / "soat_loi_nhan_conf90.csv"
        high_conf_errors.to_csv(err_csv, index=False)
        print(f"-> Đã lưu danh sách soát lỗi ra: {err_csv}")
    else:
        print("-> Không có ảnh nào bị đoán sai với độ tin cậy >= 0.90!")

    # 5. Xuất báo cáo Markdown chi tiết
    md_report_path = out_model_path.parent / "danh_gia_tu_the.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo đánh giá mô hình tư thế lợn (Pig Posture)\n\n")
        f.write(f"- **Mô hình**: `YOLO11n-cls`\n")
        f.write(f"- **Trọng số lưu tại**: `{out_model_path}`\n")
        f.write(f"- **Tập dữ liệu**: `{data_dir}`\n")
        f.write(f"- **Tổng số ảnh Test**: {total_test} ảnh ({len(df_test_res[df_test_res['true_label']=='dung'])} dung, {len(df_test_res[df_test_res['true_label']=='khong_dung'])} khong_dung)\n")
        f.write(f"- **Tổng số ảnh Val**: {len(df_val_res)} ảnh\n\n")

        f.write("## 1. Bảng so sánh chỉ số với mục tiêu (plan_data_posture.md)\n\n")
        f.write("| Chỉ số | Bộ cũ | Mục tiêu bộ mới | Kết quả bộ mới | Đánh giá |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Accuracy test (tất cả) | 0,909 (186 ảnh) | ≥ 0,92 (khoảng 300+ ảnh) | **{acc_total:.4f}** ({correct_test}/{total_test}) | {'ĐẠT' if acc_total >= 0.92 else 'CHƯA ĐẠT'} |\n")
        f.write(f"| Accuracy test không tính RGB_9579 | 0,887 | ≥ 0,90 | **{acc_no_rgb:.4f}** ({df_test_no_rgb['is_correct'].sum()}/{len(df_test_no_rgb)}) | {'ĐẠT' if acc_no_rgb >= 0.90 else 'CHƯA ĐẠT'} |\n")
        f.write(f"| Accuracy riêng ca chen chúc (IoU ≥ 0.3) | Chưa đo | ≥ 0,85 | **{acc_chen:.4f}** ({sub_chen['is_correct'].sum()}/{len(sub_chen)}) | {'ĐẠT' if acc_chen >= 0.85 else 'CHƯA ĐẠT'} |\n")
        f.write(f"| Accuracy riêng ca sát mép/tường | Chưa đo | ≥ 0,85 | **{acc_mep:.4f}** ({sub_mep['is_correct'].sum()}/{len(sub_mep)}) | {'ĐẠT' if acc_mep >= 0.85 else 'CHƯA ĐẠT'} |\n")
        f.write(f"| Accuracy riêng ca ban đêm | Chưa đo | ≥ 0,85 | **{acc_dem:.4f}** ({sub_dem['is_correct'].sum()}/{len(sub_dem)}) | {'ĐẠT' if acc_dem >= 0.85 else 'CHƯA ĐẠT'} |\n")
        f.write(f"| Con không đứng bị lọt qua ở ngưỡng P(đứng) ≥ 0,8 | 3,4 % | ≤ 2 % | **{ty_le_nham_08 * 100:.2f} %** ({khong_dung_nham_08}/{len(true_khong_dung)}) | {'ĐẠT' if ty_le_nham_08 <= 0.02 else 'CHƯA ĐẠT'} |\n")
        f.write(f"| Recall dung ở ngưỡng 0,8 (không RGB) | 0,625 | ≥ 0,70 | **{recall_dung_08:.4f}** ({dung_dat_08}/{len(true_dung_non_rgb)}) | {'ĐẠT' if recall_dung_08 >= 0.70 else 'CHƯA ĐẠT'} |\n\n")

        f.write("## 2. Danh sách ảnh Val/Test bị đoán ngược nhãn với Confidence ≥ 0.90\n\n")
        if len(high_conf_errors) > 0:
            f.write(f"Tìm thấy **{len(high_conf_errors)}** ảnh có độ tự tin cao nhưng ngược nhãn thật:\n\n")
            f.write("| Tập | Tên file | Nhãn thật | Dự đoán | P(đứng) | P(không đứng) | Nguồn | Trại |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for _, r in high_conf_errors.iterrows():
                f.write(f"| {r['split']} | `{r['file']}` | `{r['true_label']}` | `{r['pred_label']}` | {r['p_dung']:.4f} | {r['p_khong_dung']:.4f} | {r['nguon']} | {r['trai']} |\n")
        else:
            f.write("Không phát hiện trường hợp nào bị đoán ngược với độ tin cậy ≥ 0.90.\n")

    print(f"\n-> Đã xuất báo cáo chi tiết ra: {md_report_path}")
    print("=" * 80)
    print("HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH.")
    print("=" * 80)


if __name__ == "__main__":
    main()
