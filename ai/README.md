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
│   ├── 01c_review_negative_leaf.ipynb       # lọc ảnh negative nghi ngờ nhãn sai bằng từ khóa + chính model đã train (đã có)
│   ├── 02_train_ripeness_baseline.ipynb     # train YOLOv8n baseline + đánh giá test/domain gap (đã có)
│   ├── 02b_review_domain_gap_ripeness.ipynb # điều tra domain gap trên test_outdomain_openfield: lỗi nhãn hay domain shift thật (đã có)
│   ├── 02c_train_ripeness_augmented.ipynb   # train lại với augmentation màu/ánh sáng tăng cường, so trực tiếp với baseline (đã có)
│   └── 03_train_leaf_baseline.ipynb         # train YOLOv8n baseline + tỷ lệ báo động giả trên ảnh negative (đã có)
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
5. **`01_build_tomato_leaf_disease_v1.ipynb`** — Add Input output của bước 4. Remap 4 bệnh mục tiêu, chuyển class `Healthy` thành negative sample (label rỗng), loại ảnh chỉ có bệnh ngoài phạm vi MVP, dedup (kể cả theo tên file gốc Roboflow — dữ liệu nguồn đã bị augment sẵn) + chia split, audit chéo SHA-256 với `leaf_zenodo` (chưa gộp — theo tài liệu, chỉ gộp sau khi xác nhận chất lượng/trùng lặp). Save Version để có Kaggle Dataset `tomato_leaf_disease_v1` dùng cho training. **Nếu Add Input thêm output của `01c_review_negative_leaf.ipynb`** (bước 9, chạy ở lần build sau), tự động loại các ảnh negative đã xác nhận gắn nhãn Healthy sai (`review_candidates_negative.csv`, chỉ áp dụng nhóm ưu tiên cao nhất — nghi ngờ bởi cả tên file lẫn model) sang `rejected_images.csv`.
6. **`01b_dataset_report_leaf.ipynb`** (tùy chọn nhưng khuyến nghị) — Add Input output đã Save Version của bước 5, đọc lại (không build lại) để in thống kê số ảnh/box theo class-split, số ảnh negative, kích thước ảnh, biểu đồ, ảnh mẫu (cả dương lẫn negative) và toàn bộ báo cáo/audit gốc. Dùng để xác nhận nhanh dataset trước khi train.
7. **`02_train_ripeness_baseline.ipynb`** — cần **Accelerator: GPU** (P100/T4x2). Add Input Kaggle Dataset `tomato_ripeness_v1`. Tự vá lại `path:` trong `data.yaml` (đường dẫn cũ từ session build không còn tồn tại), train YOLOv8n (imgsz 640, epochs 100, batch 16, patience 20, seed 42), đánh giá riêng trên `test` và `test_outdomain_openfield` để đo domain gap, lưu `best.pt` + train config + metrics + ảnh dự đoán mẫu.
8. **`03_train_leaf_baseline.ipynb`** — cấu hình tương tự cho `tomato_leaf_disease_v1` (GPU, cùng vá `data.yaml`), đánh giá trên `test` (không có tập ngoài miền riêng cho nhánh này) + tính riêng tỷ lệ ảnh negative (lá khỏe) bị báo nhầm có bệnh (false positive trigger rate).
8b. **`02b_review_domain_gap_ripeness.ipynb`** (tùy chọn, chạy khi domain gap ở bước 7 lớn) — Add Input dataset `tomato_ripeness_v1` + output đã Save Version của bước 7 (chứa `best.pt`). So khớp dự đoán với ground truth theo IoU trên `test_outdomain_openfield`, dựng ma trận nhầm lẫn, xem trực quan ảnh lỗi nhiều nhất (GT vs dự đoán), so sánh thống kê màu/độ sáng — để phân biệt domain gap là do lỗi nhãn (có thể sửa) hay domain shift thật (cần dữ liệu camera thật, không sửa được bằng gắn lại nhãn).
8c. **`02c_train_ripeness_augmented.ipynb`** (chạy sau khi 8b xác nhận domain shift thật) — Add Input dataset `tomato_ripeness_v1` (không cần model cũ). Train lại YOLOv8n với `hsv_s`/`hsv_v`/`hsv_h` cao hơn mặc định, so trực tiếp với số liệu baseline (đã lưu cứng trong notebook) trên cả `test` và `test_outdomain_openfield`, tách riêng xem tỷ lệ nhầm hướng "xanh hơn thực tế" và số box báo thừa có cải thiện không. Chỉ xử lý nguyên nhân màu sắc/ánh sáng — không xử lý được nguyên nhân báo thừa trên nền lạ (cần dữ liệu nền thật, không dùng `test_outdomain_openfield` để tạo hard negative vì sẽ phá vỡ tính độc lập của tập test).
9. **`01c_review_negative_leaf.ipynb`** (tùy chọn, chạy khi tỷ lệ báo động giả ở bước 8 cao) — Add Input cả dataset `tomato_leaf_disease_v1` lẫn output đã Save Version của bước 8 (chứa `best.pt`). Lọc ảnh negative nghi ngờ bị gắn nhãn Healthy sai bằng 2 lớp: từ khóa trong tên file gốc + chính model baseline tự báo có bệnh, gộp thành shortlist ưu tiên để xem lại bằng mắt thay vì rà thủ công toàn bộ. Chỉ chẩn đoán, không tự sửa dataset. Sau khi Save Version, quay lại bước 5 (Add Input thêm output của bước này) để tự động loại các ảnh đã xác nhận khỏi tập negative, rồi train lại từ bước 8.

## Trạng thái hiện tại (sau baseline + 1 vòng cải thiện dữ liệu)

**Nhánh độ chín (`tomato_ripeness_v1`, YOLOv8n baseline):** đạt cả 3 mục tiêu MVP tham khảo trên `test` cùng miền (Precision 0,826 / Recall 0,827 / mAP@0.5 0,878). Domain gap trên `test_outdomain_openfield` rất lớn (mAP@0.5 rơi còn 0,473) — `02b_review_domain_gap_ripeness.ipynb` đã điều tra và **xác nhận đây là domain shift thật, không phải lỗi nhãn** (class mapping của `openfield_bd` đã xác nhận đúng qua `data.yaml` thật từ trước):
- Ma trận nhầm lẫn lệch hệ thống: 93% lỗi phân loại đi theo hướng đánh giá quả *xanh hơn* thực tế (`fruit_turning`→`fruit_green_unripe` chiếm 533/835 lỗi).
- Model báo thừa (false positive) nhiều hơn cả số quả thật: 3.231 box thừa / 2.802 box GT trên 604 ảnh (~5,4 box thừa/ảnh) — xem ảnh trực quan cho thấy model "ảo giác" ra quả trên nền đất/đá ngoài đồng, một loại nền chưa từng xuất hiện trong dữ liệu train (toàn bộ nguồn trong miền đều chụp trong nhà kính).
- Bằng chứng định lượng độc lập: chênh lệch độ bão hòa màu 44,5/255 giữa trong miền và `openfield_bd`.
- Lưu ý: `openfield_bd` bản thân là ảnh **ngoài đồng** (không phải nhà kính) — domain gap đo được ở đây phản ánh độ nhạy cảm với thay đổi môi trường nói chung, chưa chắc đại diện chính xác gap thực tế khi triển khai trong nhà kính.

**Nhánh bệnh lá (`tomato_leaf_disease_v1`, YOLOv8n baseline):** đạt cả 3 mục tiêu MVP thoải mái (Precision 0,918 / Recall 0,872 / mAP@0.5 0,937) sau 1 vòng cải thiện dữ liệu. `01c_review_negative_leaf.ipynb` phát hiện một lô ảnh liên tiếp trong dữ liệu gốc Roboflow bị gắn nhãn Healthy sai (`Leaf-Mold-509`–`524`, `Late-Blight-650`–`654`); loại 62 ảnh đã xác nhận và train lại giúp `leaf_mold` Recall tăng từ 0,666 → 0,774, tỷ lệ báo động giả trên ảnh lá khỏe giảm từ 24,6% → 19,7%.

## Đề xuất bước tiếp theo

1. **Ưu tiên cao nhất — thu thập ảnh thật từ camera IMX179** trong đúng môi trường nhà kính triển khai (cả quả lẫn lá). Đây là hành động ngoài phạm vi notebook (cần phần cứng thật), nhưng là nút thắt quan trọng nhất: domain shift đã xác nhận ở nhánh độ chín không thể sửa bằng cách xử lý lại dữ liệu public hiện có — chỉ dữ liệu thật mới thu hẹp được gap này. Ảnh thật cũng nên dùng làm test set thực tế thay thế dần cho `test_outdomain_openfield`.
2. ~~Trong lúc chờ dữ liệu thật: thử tăng cường augmentation màu/ánh sáng~~ — **đã làm** ở `02c_train_ripeness_augmented.ipynb` (`hsv_s` 0,7→0,9, `hsv_v` 0,4→0,6, `hsv_h` 0,015→0,02). Phần "hard negative nền đất" trong đề xuất gốc **không khả thi** mà không có dữ liệu mới: nguồn ảnh nền duy nhất hiện có là chính `test_outdomain_openfield`, dùng nó để tạo hard negative sẽ phá vỡ tính độc lập của tập test — để lại cho khi có ảnh thật (mục 1). *(Điền kết quả cụ thể vào đây sau khi chạy `02c` trên Kaggle.)*
3. **Dọn nốt các mục còn treo:**
   - `review_required.csv` của AgRobTomato (116 box "reddish" chưa gán nhãn) — nhánh độ chín.
   - Quyết định gộp `leaf_zenodo` hay không, dựa trên `zenodo_audit.md` đã có sẵn số liệu.
   - 26 ảnh negative nhóm ưu tiên 1 (`priority == 1`, chỉ 1 trong 2 lớp lọc nghi ngờ) trong `01c_review_negative_leaf.ipynb` chưa được xem xét — nhóm ưu tiên 2 đã xử lý xong.
   - Class-id của `tomato_plantfactory` vẫn là giả định (đã xác nhận trực quan nhưng chưa có `classes.txt` gốc để xác minh chính thức).
4. **Sau khi có dữ liệu thật hoặc dọn xong các mục trên:** train "phiên bản chính thức" theo đúng quy trình mục 9 của tài liệu gốc — không chỉ tăng epoch mà phải dựa trên cải thiện dữ liệu/nhãn đã kiểm chứng.
5. **Xa hơn (sau khi cả 2 baseline ổn định — đã đạt điều kiện này):** có thể thử nghiệm gộp 2 nhánh thành 1 model đa lớp (9 class: 4 bệnh lá + 3 độ chín + 2 class mở rộng), theo đúng lộ trình đề xuất trong tài liệu, chỉ khi có nhu cầu đơn giản hóa triển khai.

## Nguồn tài liệu tham chiếu

- `TONG_HOP_YEU_CAU_VA_DE_XUAT_AI_SMART_GREENHOUSE.docx` — yêu cầu hệ thống, class mapping đề xuất, luồng inference/tích hợp.
- `HUONG_DAN_XAY_DUNG_DATASET_SMART_GREENHOUSE_AI.md` — quy trình xây dataset chi tiết (13 bước), checklist, Definition of Done.
- `TONG_HOP_DATASET_DO_CHIN_CA_CHUA_PUBLIC.docx` — danh sách + link tải các dataset độ chín quả.

## Quy tắc

- Không remap class chỉ dựa vào tên — xem `manifests/class_mapping.yaml` khi được tạo.
- `openfield_bd` license chưa xác nhận → chỉ dùng test ngoài miền, không đưa vào train/merge.
- `leaf_zenodo` chưa gộp vào `tomato_leaf_disease_v1` — chỉ audit (xem `manifests/zenodo_audit.md`) cho tới khi xác nhận class mapping/chất lượng label/mức trùng lặp.
- Ảnh lá khỏe (`Healthy`) không tạo class riêng — giữ làm negative sample (label rỗng); ảnh chỉ có bệnh ngoài phạm vi MVP (không phải Healthy) bị loại hẳn thay vì bị coi nhầm là negative.
- Nhãn negative không mặc định đúng — `01c_review_negative_leaf.ipynb` từng phát hiện một lô ảnh liên tiếp (`Leaf-Mold-509`–`524`, `Late-Blight-650`–`654`) bị gắn nhãn Healthy sai trong dữ liệu gốc Roboflow; chỉ loại các ảnh đã xác nhận cả 2 lớp lọc (tên file + model), không tự động loại chỉ vì model nghi ngờ một mình.
- Mọi dataset gộp phải giữ trường `source_dataset` để đánh giá theo từng nguồn và chia split chống leakage theo video/cây/ngày.