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
│   ├── 02d_review_reddish_agrob.ipynb       # xem xét 116 box "reddish" AgRobTomato, đề xuất class theo Hue thực đo (đã có)
│   ├── 03_train_leaf_baseline.ipynb         # train YOLOv8n baseline + tỷ lệ báo động giả trên ảnh negative (đã có)
│   └── 01d_review_leaf_remaining.ipynb      # audit chất lượng leaf_zenodo + xem 26 ảnh negative ưu tiên 1 (đã có)
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
10. **`02d_review_reddish_agrob.ipynb`** (dọn mục treo, nhánh độ chín) — Add Input output đã Save Version của bước 1 (`raw/agrob_tomato`) + bước 2 (`review_required.csv`). Tính Hue trung bình từng box "reddish" thật, so với tâm màu Hue của 2 class đã biết chắc (`breaking`→`fruit_turning`, `riped`→`fruit_ripe`) để đề xuất gán class theo khoảng cách màu gần nhất thay vì đoán, kèm xác nhận trực quan trước khi tin. Sinh `reddish_class_decision.csv` — chưa tự áp dụng vào dataset.
11. **`01d_review_leaf_remaining.ipynb`** (dọn 2 mục treo, nhánh bệnh lá) — Add Input output của bước 4 (`raw/leaf_zenodo`), bước 5 mới nhất, và bước 9 (`negative_images_full_scan.csv`). Phần A: audit trực quan chất lượng label `leaf_zenodo` theo từng class, đối chiếu 4 tiêu chí gộp. Phần B: xem 26 ảnh negative nhóm ưu tiên 1 (chỉ 1 trong 2 lớp lọc nghi ngờ ở bước 9) mà nhóm ưu tiên 2 chưa xử lý. Chỉ chẩn đoán, không tự sửa dataset.

## Trạng thái hiện tại (sau baseline + 1 vòng cải thiện dữ liệu)

**Nhánh độ chín (`tomato_ripeness_v1`, YOLOv8n baseline):** đạt cả 3 mục tiêu MVP tham khảo trên `test` cùng miền (Precision 0,826 / Recall 0,827 / mAP@0.5 0,878). Domain gap trên `test_outdomain_openfield` rất lớn (mAP@0.5 rơi còn 0,473) — `02b_review_domain_gap_ripeness.ipynb` đã điều tra và **xác nhận đây là domain shift thật, không phải lỗi nhãn** (class mapping của `openfield_bd` đã xác nhận đúng qua `data.yaml` thật từ trước):
- Ma trận nhầm lẫn lệch hệ thống: 93% lỗi phân loại đi theo hướng đánh giá quả *xanh hơn* thực tế (`fruit_turning`→`fruit_green_unripe` chiếm 533/835 lỗi).
- Model báo thừa (false positive) nhiều hơn cả số quả thật: 3.231 box thừa / 2.802 box GT trên 604 ảnh (~5,4 box thừa/ảnh) — xem ảnh trực quan cho thấy model "ảo giác" ra quả trên nền đất/đá ngoài đồng, một loại nền chưa từng xuất hiện trong dữ liệu train (toàn bộ nguồn trong miền đều chụp trong nhà kính).
- Bằng chứng định lượng độc lập: chênh lệch độ bão hòa màu 44,5/255 giữa trong miền và `openfield_bd`.
- Lưu ý: `openfield_bd` bản thân là ảnh **ngoài đồng** (không phải nhà kính) — domain gap đo được ở đây phản ánh độ nhạy cảm với thay đổi môi trường nói chung, chưa chắc đại diện chính xác gap thực tế khi triển khai trong nhà kính.

**Cập nhật — `02c_train_ripeness_augmented.ipynb` (tăng `hsv_s` 0,7→0,9, `hsv_v` 0,4→0,6, `hsv_h` 0,015→0,02):** kết quả tốt hơn đáng kể so với dự đoán ban đầu (dự đoán sai: tưởng augmentation màu chỉ giúp phần nhầm lẫn độ chín, không đụng tới phần báo thừa — thực tế giúp cả hai).
- `test` (cùng miền): gần như không đổi (mAP@0.5 0,878→0,876) — không đánh đổi hiệu năng.
- `test_outdomain_openfield`: mAP@0.5 **0,473 → 0,573** (+0,0995), mAP@0.5:0.95 0,430→0,515.
- Đối chiếu IoU chi tiết: box báo thừa (`extra`) giảm gần một nửa **3.231 → 1.673**; lỗi phân loại (`misclassified`) giảm 835→775; lỗi "đánh giá xanh hơn thực tế" giảm 777→644 (nhưng lỗi ngược lại tăng 58→131 — bớt thiên lệch 1 chiều nhưng chưa hết).
- **Khuyến nghị: dùng model này (`models/tomato_ripeness_yolov8n_augmented.pt`) thay cho baseline gốc** làm điểm khởi đầu cho các bước tiếp theo — cải thiện thật, không có đánh đổi đáng kể.

**Cập nhật — `02d_review_reddish_agrob.ipynb` (116 ảnh / 184 box "reddish" của AgRobTomato):** đo Hue trung bình thật từng box, so với tâm màu Hue của 2 class đã biết đúng (`breaking`=64,0° → `fruit_turning`, `riped`=54,9° → `fruit_ripe`; `unriped`=89,6° tách biệt rõ, xác nhận "reddish" chắc chắn không phải quả xanh). Kết quả: **130 box (71%) → `fruit_ripe`**, **54 box (29%) → `fruit_turning`**, dựa trên khoảng cách Hue gần nhất. Ảnh mẫu xác nhận hướng phân loại hợp lý (nhóm `turning` ngả cam/vàng, nhóm `ripe` ngả đỏ đậm hơn), dù 2 class tham chiếu tự thân cũng chồng lấn Hue đáng kể nên ranh giới không tuyệt đối. Đã lưu `reddish_class_decision.csv`, sẽ áp dụng vào `01_build_tomato_ripeness_v1.ipynb`.

**Nhánh bệnh lá (`tomato_leaf_disease_v1`, YOLOv8n baseline):** đạt cả 3 mục tiêu MVP thoải mái (Precision 0,918 / Recall 0,872 / mAP@0.5 0,937) sau 1 vòng cải thiện dữ liệu. `01c_review_negative_leaf.ipynb` phát hiện một lô ảnh liên tiếp trong dữ liệu gốc Roboflow bị gắn nhãn Healthy sai (`Leaf-Mold-509`–`524`, `Late-Blight-650`–`654`); loại 62 ảnh đã xác nhận và train lại giúp `leaf_mold` Recall tăng từ 0,666 → 0,774, tỷ lệ báo động giả trên ảnh lá khỏe giảm từ 24,6% → 19,7%.

## Đề xuất bước tiếp theo

1. **Ưu tiên cao nhất — thu thập ảnh thật từ camera IMX179** trong đúng môi trường nhà kính triển khai (cả quả lẫn lá). Đây là hành động ngoài phạm vi notebook (cần phần cứng thật), nhưng là nút thắt quan trọng nhất: domain shift đã xác nhận ở nhánh độ chín không thể sửa bằng cách xử lý lại dữ liệu public hiện có — chỉ dữ liệu thật mới thu hẹp được gap này. Ảnh thật cũng nên dùng làm test set thực tế thay thế dần cho `test_outdomain_openfield`.
2. ~~Trong lúc chờ dữ liệu thật: thử tăng cường augmentation màu/ánh sáng~~ — **Hoàn thành, kết quả tốt hơn kỳ vọng.** `02c_train_ripeness_augmented.ipynb` (`hsv_s` 0,7→0,9, `hsv_v` 0,4→0,6, `hsv_h` 0,015→0,02) giúp mAP@0.5 trên `test_outdomain_openfield` tăng từ 0,473 lên 0,573, và bất ngờ nhất là số box báo thừa trên nền lạ giảm gần một nửa (3.231→1.673) dù ban đầu dự đoán augmentation màu sẽ không đụng tới vấn đề này — không đánh đổi hiệu năng trên `test` cùng miền. Chi tiết xem mục "Trạng thái hiện tại" ở trên. Phần "hard negative nền đất" trong đề xuất gốc **không khả thi** mà không có dữ liệu mới: nguồn ảnh nền duy nhất hiện có là chính `test_outdomain_openfield`, dùng nó để tạo hard negative sẽ phá vỡ tính độc lập của tập test — để lại cho khi có ảnh thật (mục 1).
3. **Dọn nốt các mục còn treo — đã xong cả 4/4:**
   - ~~`review_required.csv` của AgRobTomato (116 box "reddish" chưa gán nhãn)~~ — **Đã có quyết định**, xem mục "Trạng thái hiện tại". `02d_review_reddish_agrob.ipynb` đo Hue thật của 184 box, đề xuất 130→`fruit_ripe`, 54→`fruit_turning` dựa trên khoảng cách tới tâm màu của 2 class đã biết đúng. Sẽ áp dụng vào `01_build_tomato_ripeness_v1.ipynb`.
   - ~~Quyết định gộp `leaf_zenodo`~~ — **Đã quyết định: KHÔNG gộp**, xem mục "Quy tắc" bên dưới (box không cùng quy ước với `leaf_roboflow`).
   - ~~26 ảnh negative nhóm ưu tiên 1~~ — **Đã xem, giữ nguyên.** Khác hẳn nhóm ưu tiên 2 (đã xác nhận nhãn sai rõ ràng), 26 ảnh này đều là lá xanh khỏe mạnh thật, model chỉ gắn cờ với confidence thấp (phần lớn 0,25–0,5 so với 0,95+ của nhóm ưu tiên 2) — không đủ bằng chứng để loại, không cần sửa dataset.
   - ~~Class-id của `tomato_plantfactory` vẫn là giả định~~ — **Coi là đã giải quyết.** Không có `classes.txt` gốc để xác minh chính thức (không tồn tại trong dataset gốc), nhưng đã xác nhận trực quan qua ảnh audit thật (`XAC_NHAN_tomato_plantfactory` ở `01_build_tomato_ripeness_v1.ipynb`): quả đỏ → `fruit_ripe`, quả xanh/nhạt → `fruit_green_unripe`, khớp đúng giả định `{0: green, 1: red}`. Đây là mức xác minh cao nhất có thể đạt được khi nguồn không cung cấp file khai báo class.
4. **Sau khi có dữ liệu thật hoặc dọn xong các mục trên:** train "phiên bản chính thức" theo đúng quy trình mục 9 của tài liệu gốc — không chỉ tăng epoch mà phải dựa trên cải thiện dữ liệu/nhãn đã kiểm chứng.
5. **Xa hơn (sau khi cả 2 baseline ổn định — đã đạt điều kiện này):** có thể thử nghiệm gộp 2 nhánh thành 1 model đa lớp (9 class: 4 bệnh lá + 3 độ chín + 2 class mở rộng), theo đúng lộ trình đề xuất trong tài liệu, chỉ khi có nhu cầu đơn giản hóa triển khai.

## Nguồn tài liệu tham chiếu

- `TONG_HOP_YEU_CAU_VA_DE_XUAT_AI_SMART_GREENHOUSE.docx` — yêu cầu hệ thống, class mapping đề xuất, luồng inference/tích hợp.
- `HUONG_DAN_XAY_DUNG_DATASET_SMART_GREENHOUSE_AI.md` — quy trình xây dataset chi tiết (13 bước), checklist, Definition of Done.
- `TONG_HOP_DATASET_DO_CHIN_CA_CHUA_PUBLIC.docx` — danh sách + link tải các dataset độ chín quả.

## Quy tắc

- Không remap class chỉ dựa vào tên — xem `manifests/class_mapping.yaml` khi được tạo.
- `openfield_bd` license chưa xác nhận → chỉ dùng test ngoài miền, không đưa vào train/merge.
- **`leaf_zenodo` — quyết định cuối: KHÔNG gộp vào `tomato_leaf_disease_v1`.** `01d_review_leaf_remaining.ipynb` Phần A phát hiện box của `leaf_zenodo` bao trùm gần như toàn bộ khung ảnh (quy ước "cả ảnh có bệnh X"), khác hẳn quy ước khoanh vùng cục bộ của `leaf_roboflow` — gộp thẳng sẽ dạy model 2 kiểu box không nhất quán, giảm chất lượng định vị. Chỉ cân nhắc lại nếu có công sức re-annotate riêng.
- Ảnh lá khỏe (`Healthy`) không tạo class riêng — giữ làm negative sample (label rỗng); ảnh chỉ có bệnh ngoài phạm vi MVP (không phải Healthy) bị loại hẳn thay vì bị coi nhầm là negative.
- Nhãn negative không mặc định đúng — `01c_review_negative_leaf.ipynb` từng phát hiện một lô ảnh liên tiếp (`Leaf-Mold-509`–`524`, `Late-Blight-650`–`654`) bị gắn nhãn Healthy sai trong dữ liệu gốc Roboflow; chỉ loại các ảnh đã xác nhận cả 2 lớp lọc (tên file + model), không tự động loại chỉ vì model nghi ngờ một mình.
- Mọi dataset gộp phải giữ trường `source_dataset` để đánh giá theo từng nguồn và chia split chống leakage theo video/cây/ngày.