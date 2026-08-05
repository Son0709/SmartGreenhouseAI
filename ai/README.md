# AI Module — Smart Greenhouse (Kaggle-first)

Toàn bộ notebook trong `ai/notebooks/` được viết để chạy trên **Kaggle**, không chạy local. Dữ liệu thật (raw/interim/processed) không sống trong repo — chỉ sống trong `/kaggle/working/` của mỗi notebook run, và được lưu lại giữa các notebook thông qua Kaggle Dataset output/input (Save Version → attach làm input cho notebook sau).

Repo chỉ giữ lại phần tái tạo được: notebook, script, manifest (`sources.csv`, `class_mapping.yaml`...).

## Cấu trúc

```text
ai/
├── notebooks/
│   ├── 00_data_acquisition_ripeness.ipynb   # tải + đóng băng dataset nhánh độ chín (đã có)
│   ├── 01_build_tomato_ripeness_v1.ipynb    # audit + class mapping + convert + dedup + split → dataset hoàn chỉnh (đã có)
│   ├── 01b_dataset_report_ripeness.ipynb    # đọc lại Kaggle Dataset đã build, thống kê + ảnh ví dụ (đã có)
│   ├── 00_data_acquisition_leaf.ipynb       # tải bộ bệnh lá qua Roboflow API + Zenodo (đã có)
│   ├── 01_build_tomato_leaf_disease_v1.ipynb # class mapping + negative sample + dedup + split → dataset hoàn chỉnh (đã có)
│   ├── 01b_dataset_report_leaf.ipynb        # đọc lại Kaggle Dataset đã build, thống kê + ảnh ví dụ (đã có)
│   ├── 02_train_ripeness_baseline.ipynb     # train YOLOv8n baseline + đánh giá test/domain gap (đã có)
│   └── 03_train_leaf_baseline.ipynb         # (chưa tạo)
├── datasets/
│   ├── manifests/        # sources.csv, class_mapping.yaml, licenses.md — version hóa trong git
│   └── (raw/interim/processed chỉ tồn tại trên Kaggle, xem .gitignore)
└── scripts/               # tiện ích dùng chung, import vào notebook khi cần
```

## Thứ tự chạy trên Kaggle

1. **`00_data_acquisition_ripeness.ipynb`** — tải Laboro, AgRobTomato, TomatoPlantfactoryDataset, OpenField-BD (curl trực tiếp) + attach Kaggle Dataset `enalis/tomatoes-dataset`. Bật **Internet: ON** trong Settings trước khi chạy. Save Version để lấy output làm input cho bước 2.
2. **`01_build_tomato_ripeness_v1.ipynb`** — Add Input output của bước 1. Khám phá cấu trúc thật, liệt kê tên class gốc, convert COCO/VOC/YOLO → YOLO 3-class thống nhất, phát hiện trùng ảnh, chia split chống leakage, sinh `tomato_ripeness_v1` hoàn chỉnh + manifest. Save Version để có Kaggle Dataset dùng cho training.
3. **`01b_dataset_report_ripeness.ipynb`** (tùy chọn nhưng khuyến nghị) — Add Input output đã Save Version của bước 2, đọc lại (không build lại) để in thống kê số ảnh/box theo class-split, kích thước ảnh, biểu đồ phân bố class, ảnh mẫu kèm bounding box, và toàn bộ báo cáo/license gốc. Dùng để xác nhận nhanh dataset trước khi train.
4. **`00_data_acquisition_leaf.ipynb`** — cần Kaggle Secret `ROBOFLOW_API_KEY` (Add-ons → Secrets, attach vào notebook). Liệt kê version thật của project Roboflow (không đoán số version) rồi tải về `raw/leaf_roboflow/`; tải thêm `raw/leaf_zenodo/` (chỉ để audit). Save Version để lấy output làm input cho bước 5.
5. **`01_build_tomato_leaf_disease_v1.ipynb`** — Add Input output của bước 4. Remap 4 bệnh mục tiêu, chuyển class `Healthy` thành negative sample (label rỗng), loại ảnh chỉ có bệnh ngoài phạm vi MVP, dedup (kể cả theo tên file gốc Roboflow — dữ liệu nguồn đã bị augment sẵn) + chia split, audit chéo SHA-256 với `leaf_zenodo` (chưa gộp — theo tài liệu, chỉ gộp sau khi xác nhận chất lượng/trùng lặp). Save Version để có Kaggle Dataset `tomato_leaf_disease_v1` dùng cho training.
6. **`01b_dataset_report_leaf.ipynb`** (tùy chọn nhưng khuyến nghị) — Add Input output đã Save Version của bước 5, đọc lại (không build lại) để in thống kê số ảnh/box theo class-split, số ảnh negative, kích thước ảnh, biểu đồ, ảnh mẫu (cả dương lẫn negative) và toàn bộ báo cáo/audit gốc. Dùng để xác nhận nhanh dataset trước khi train.
7. **`02_train_ripeness_baseline.ipynb`** — cần **Accelerator: GPU** (P100/T4x2). Add Input Kaggle Dataset `tomato_ripeness_v1`. Tự vá lại `path:` trong `data.yaml` (đường dẫn cũ từ session build không còn tồn tại), train YOLOv8n (imgsz 640, epochs 100, batch 16, patience 20, seed 42), đánh giá riêng trên `test` và `test_outdomain_openfield` để đo domain gap, lưu `best.pt` + train config + metrics + ảnh dự đoán mẫu.
8. `03_train_leaf_baseline.ipynb` (chưa tạo) — cấu hình tương tự cho `tomato_leaf_disease_v1`, đánh giá trên `test` (không có tập ngoài miền riêng cho nhánh này).

## Nguồn tài liệu tham chiếu

- `TONG_HOP_YEU_CAU_VA_DE_XUAT_AI_SMART_GREENHOUSE.docx` — yêu cầu hệ thống, class mapping đề xuất, luồng inference/tích hợp.
- `HUONG_DAN_XAY_DUNG_DATASET_SMART_GREENHOUSE_AI.md` — quy trình xây dataset chi tiết (13 bước), checklist, Definition of Done.
- `TONG_HOP_DATASET_DO_CHIN_CA_CHUA_PUBLIC.docx` — danh sách + link tải các dataset độ chín quả.

## Quy tắc

- Không remap class chỉ dựa vào tên — xem `manifests/class_mapping.yaml` khi được tạo.
- `openfield_bd` license chưa xác nhận → chỉ dùng test ngoài miền, không đưa vào train/merge.
- `leaf_zenodo` chưa gộp vào `tomato_leaf_disease_v1` — chỉ audit (xem `manifests/zenodo_audit.md`) cho tới khi xác nhận class mapping/chất lượng label/mức trùng lặp.
- Ảnh lá khỏe (`Healthy`) không tạo class riêng — giữ làm negative sample (label rỗng); ảnh chỉ có bệnh ngoài phạm vi MVP (không phải Healthy) bị loại hẳn thay vì bị coi nhầm là negative.
- Mọi dataset gộp phải giữ trường `source_dataset` để đánh giá theo từng nguồn và chia split chống leakage theo video/cây/ngày.