# AI Module — Smart Greenhouse (Kaggle-first)

Toàn bộ notebook trong `ai/notebooks/` được viết để chạy trên **Kaggle**, không chạy local. Dữ liệu thật (raw/interim/processed) không sống trong repo — chỉ sống trong `/kaggle/working/` của mỗi notebook run, và được lưu lại giữa các notebook thông qua Kaggle Dataset output/input (Save Version → attach làm input cho notebook sau).

Repo chỉ giữ lại phần tái tạo được: notebook, script, manifest (`sources.csv`, `class_mapping.yaml`...).

## Cấu trúc

```text
ai/
├── notebooks/
│   ├── 00_data_acquisition_ripeness.ipynb   # tải + đóng băng dataset nhánh độ chín (đã có)
│   ├── 01_ripeness_dataset_audit.ipynb      # audit + class mapping + split (chưa tạo)
│   ├── 00_data_acquisition_leaf.ipynb       # tải bộ bệnh lá qua Roboflow API (chưa tạo)
│   ├── 01_leaf_dataset_audit.ipynb          # (chưa tạo)
│   ├── 02_ripeness_baseline_train.ipynb     # (chưa tạo)
│   └── 03_leaf_baseline_train.ipynb         # (chưa tạo)
├── datasets/
│   ├── manifests/        # sources.csv, class_mapping.yaml, licenses.md — version hóa trong git
│   └── (raw/interim/processed chỉ tồn tại trên Kaggle, xem .gitignore)
└── scripts/               # tiện ích dùng chung, import vào notebook khi cần
```

## Thứ tự chạy trên Kaggle

1. **`00_data_acquisition_ripeness.ipynb`** — tải Laboro, AgRobTomato, TomatoPlantfactoryDataset, OpenField-BD (curl trực tiếp) + attach Kaggle Dataset `enalis/tomatoes-dataset`. Bật **Internet: ON** trong Settings trước khi chạy.
2. `01_ripeness_dataset_audit.ipynb` (kế tiếp) — audit trực quan, `class_mapping.yaml`, convert VOC→YOLO, chống leakage, chia split → tạo `tomato_ripeness_v1`.
3. Song song/sau đó: nhánh bệnh lá qua Roboflow API (đã có API key) — `00_data_acquisition_leaf.ipynb`.
4. Baseline training riêng cho từng nhánh (YOLO nano, imgsz 640, epochs ~100, patience 20, seed 42) theo cấu hình trong hai tài liệu gốc.

## Nguồn tài liệu tham chiếu

- `TONG_HOP_YEU_CAU_VA_DE_XUAT_AI_SMART_GREENHOUSE.docx` — yêu cầu hệ thống, class mapping đề xuất, luồng inference/tích hợp.
- `HUONG_DAN_XAY_DUNG_DATASET_SMART_GREENHOUSE_AI.md` — quy trình xây dataset chi tiết (13 bước), checklist, Definition of Done.
- `TONG_HOP_DATASET_DO_CHIN_CA_CHUA_PUBLIC.docx` — danh sách + link tải các dataset độ chín quả.

## Quy tắc

- Không remap class chỉ dựa vào tên — xem `manifests/class_mapping.yaml` khi được tạo.
- `openfield_bd` license chưa xác nhận → chỉ dùng test ngoài miền, không đưa vào train/merge.
- Mọi dataset gộp phải giữ trường `source_dataset` để đánh giá theo từng nguồn và chia split chống leakage theo video/cây/ngày.