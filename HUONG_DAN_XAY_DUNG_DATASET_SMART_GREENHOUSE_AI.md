# HƯỚNG DẪN XÂY DỰNG DATASET CHO SMART GREENHOUSE AI

**Dự án:** Smart Greenhouse IoT + AI cho cây cà chua  
**Mục tiêu:** Hoàn thiện dataset dùng cho YOLO để phát hiện bệnh lá, phát hiện quả và phân loại độ chín; chuẩn bị dữ liệu để triển khai trên Raspberry Pi với camera IMX179.  
**Phiên bản tài liệu:** 1.0  
**Ngày cập nhật:** 05/08/2026

---

## 1. Quyết định kỹ thuật ban đầu

Đội ngũ AI **không nên gộp toàn bộ dataset và train ngay**.

Hướng triển khai an toàn và dễ kiểm soát nhất:

1. Xây dựng riêng `tomato_leaf_disease_v1` cho bệnh lá.
2. Xây dựng riêng `tomato_ripeness_v1` cho quả và ba mức độ chín.
3. Audit chất lượng từng nguồn trước khi remap hoặc gộp class.
4. Phát hiện ảnh trùng và chia dữ liệu theo nguồn/video/cây/buổi chụp để chống data leakage.
5. Train hai model baseline riêng.
6. Thu thập ảnh từ camera IMX179 để tạo tập test thực tế và fine-tuning.
7. Chỉ thử model đa lớp hoặc sáu mức chín sau khi hai baseline ổn định.

### Phạm vi MVP đề xuất

#### Bệnh lá

```yaml
0: leaf_early_blight
1: leaf_late_blight
2: leaf_mold
3: leaf_septoria_spot
```

#### Độ chín quả phiên bản 1

```yaml
0: fruit_green_unripe
1: fruit_turning
2: fruit_ripe
```

#### Phạm vi mở rộng

```yaml
3: fruit_overripe
4: fruit_damaged
```

Hai class mở rộng chỉ được đưa vào detection khi có đủ bounding box và tiêu chí gắn nhãn thống nhất.

---

# 2. Danh mục dataset bệnh lá

## 2.1. Tomato Leaf Disease — Roboflow

**Đường dẫn:**  
https://universe.roboflow.com/universitas-atma-jaya/tomato-leaf-disease-rxcft

**Loại bài toán:** Object Detection  
**Quy mô công khai:** khoảng 8.439 ảnh gốc  
**Số class:** 11  
**Giấy phép:** CC BY 4.0

### Các class công khai

- Healthy
- Bacterial Spot
- Early Blight
- Iron Deficiency
- Late Blight
- Leaf Mold
- Leaf Miner
- Mosaic Virus
- Septoria
- Spider Mites
- Yellow Leaf Curl Virus

### Ánh xạ vào dự án

| Class nguồn | Class chuẩn | Hành động |
|---|---|---|
| Early Blight | `leaf_early_blight` | Giữ |
| Late Blight | `leaf_late_blight` | Giữ |
| Leaf Mold | `leaf_mold` | Giữ |
| Septoria | `leaf_septoria_spot` | Giữ |
| Healthy | Không tạo class | Chuyển thành negative sample |
| Các class còn lại | Ngoài MVP | Loại khỏi baseline hoặc giữ cho phiên bản sau |

### Vai trò đề xuất

Nguồn chính cho `tomato_leaf_disease_v1`.

### Điểm mạnh

- Đúng định dạng Object Detection.
- Có đủ bốn bệnh lá đề xuất cho MVP.
- Quy mô tương đối lớn.
- Có license rõ ràng.
- Có thể export sang định dạng YOLO.

### Rủi ro cần kiểm tra

- Một số phiên bản Roboflow có nhiều ảnh do augmentation, không phải ảnh gốc độc lập.
- Có thể mất cân bằng class.
- Class `Healthy` không phù hợp với thiết kế detection của dự án.
- Cần xác định bounding box khoanh vùng triệu chứng hay toàn bộ lá.
- Ảnh public có thể khác môi trường camera nhà kính thực tế.

### Việc đội AI phải làm

- Tải phiên bản ít hoặc chưa augmentation.
- Lưu dataset gốc bất biến.
- Audit ít nhất 100 ảnh ngẫu nhiên.
- Kiểm tra 30–50 bounding box mỗi class mục tiêu.
- Xóa box `Healthy`; giữ ảnh làm negative sample khi phù hợp.
- Thống kê số object theo từng bệnh.
- Kiểm tra ảnh trùng và near-duplicate.
- Chia lại train/validation/test.

---

## 2.2. Tomato Leaf Disease Detection — Zenodo

**Đường dẫn:**  
https://zenodo.org/records/20004230

**Loại bài toán:** Object Detection  
**Định dạng công bố:** YOLOv8 ZIP  
**Nền tảng:** Zenodo

### Vai trò đề xuất

Nguồn bổ sung cho dataset bệnh lá.

### Thông tin phải xác minh sau khi tải

- Tổng số ảnh.
- Danh sách class trong `data.yaml`.
- Số object của từng class.
- License.
- Nguồn ảnh gốc.
- Dataset có augmentation hay chưa.
- Tỷ lệ train/validation/test hiện có.
- Chất lượng bounding box.
- Mức độ trùng lặp với bộ Roboflow phía trên.

### Quyết định sử dụng

Chưa gộp ngay. Dataset chỉ được đưa vào `tomato_leaf_disease_v1` khi:

- License phù hợp.
- Class ánh xạ rõ ràng.
- Label đạt chất lượng.
- Không có trùng lặp nghiêm trọng.
- Không gây leakage giữa các split.

---

# 3. Danh mục dataset quả và độ chín

## 3.1. Laboro Tomato

**Trang chính thức:**  
https://github.com/laboroai/LaboroTomato

**Tải ZIP trực tiếp:**  
https://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip

**Loại bài toán:** Object Detection + Instance Segmentation  
**Quy mô:** 804 ảnh  
**Tổng bounding box:** 9.777  
**Số nhãn:** 6  
**Giấy phép:** CC BY-NC-SA 4.0

### Các class

- `b_fully_ripened`
- `b_half_ripened`
- `b_green`
- `l_fully_ripened`
- `l_half_ripened`
- `l_green`

Sáu nhãn là tổ hợp giữa kích thước quả và ba mức độ chín, không phải sáu mức chín riêng biệt.

### Ánh xạ đề xuất

```yaml
b_green: fruit_green_unripe
l_green: fruit_green_unripe

b_half_ripened: fruit_turning
l_half_ripened: fruit_turning

b_fully_ripened: fruit_ripe
l_fully_ripened: fruit_ripe
```

### Vai trò đề xuất

Nguồn nền chính cho `tomato_ripeness_v1`.

### Điểm mạnh

- Số bounding box rất lớn so với số ảnh.
- Ảnh trong môi trường nhà kính.
- Có nhiều quả trong cùng khung hình.
- Có trường hợp che khuất.
- Có cả bounding box và mask.

### Rủi ro

- License có điều kiện phi thương mại và chia sẻ tương tự.
- Cần xác minh chính xác ý nghĩa tiền tố `b_` và `l_`.
- Không có `fruit_overripe` và `fruit_damaged`.
- Dễ bỏ sót nhãn ở ảnh có mật độ quả cao.

### Việc đội AI phải làm

- Gộp class theo độ chín.
- Giữ thông tin nguồn trong manifest.
- Kiểm tra ảnh có nhiều quả và quả bị che khuất.
- Kiểm tra object bị bỏ sót.
- Xem lại điều khoản trước khi phát hành lại dataset phái sinh.

---

## 3.2. Tomato Ripeness Detection — Morpheus/Roboflow

**Đường dẫn do dự án cung cấp:**  
https://universe.roboflow.com/morpheus-4kkqr/tomato-ripeness-detection-tjysz

### Vai trò đề xuất

Nguồn bổ sung cho ba mức độ chín.

### Thông tin cần audit trực tiếp

- Số ảnh gốc.
- Danh sách class chính thức.
- Số object từng class.
- License.
- Phiên bản dataset.
- Dataset đã augmentation hay chưa.
- Cách khoanh vùng quả.
- Ảnh có phải bối cảnh nhà kính hay nền sạch.
- Mức độ trùng với Laboro và các bộ Roboflow khác.

### Quyết định sử dụng

Không đưa vào bản merged cho đến khi hoàn thành `dataset audit report`. Nếu class là Green/Yellow-Orange/Red hoặc tương đương thì có thể ánh xạ vào ba class chuẩn sau khi kiểm tra tiêu chí hình ảnh.

---

## 3.3. Tomato Maturity Detection (IEEE)

**Đường dẫn:**  
https://universe.roboflow.com/helmiubayastudent/tomato-maturity-detection-ieee

**Loại bài toán:** Object Detection  
**Quy mô:** 1.625 ảnh  
**Số nhãn:** 6  
**Giấy phép:** CC BY 4.0

### Các class

- green
- breaker
- turning
- pink
- light red
- red

### Vai trò đề xuất

Nguồn chính cho phiên bản sáu mức chín trong tương lai.

### Lý do chưa dùng ngay cho v1

- Phạm vi chi tiết hơn baseline ba class.
- Không thể gộp tự động các class trung gian chỉ dựa vào tên.
- Cần kiểm tra tính nhất quán giữa breaker, turning, pink và light red.
- Cần audit ảnh trùng, nguồn ảnh và mất cân bằng class.

### Việc đội AI phải làm

- Lưu riêng trong `raw/tomato_maturity_ieee/`.
- Audit ít nhất 100 ảnh.
- Thống kê object theo class.
- Xây dựng guideline màu cho sáu mức chín.
- Chỉ dùng cho `tomato_ripeness_v2_6classes`.

---

## 3.4. TomatoPlantfactoryDataset

**Trang tải chính thức:**  
https://data.mendeley.com/datasets/8h3s6jkyff/3

**GitHub hướng dẫn:**  
https://github.com/veveup/Tomato-Plant-Factory-Dataset

**Loại bài toán:** Object Detection  
**Quy mô:** 520 ảnh  
**Tổng số quả:** 9.112  
**Định dạng:** YOLO + Pascal VOC  
**Giấy phép:** CC BY 4.0

### Các class

- green
- red

### Phân bố công bố

- Green: 5.996 đối tượng.
- Red: 3.116 đối tượng.

### Ánh xạ đề xuất

```yaml
green: fruit_green_unripe
red: fruit_ripe
```

### Vai trò đề xuất

Nguồn bổ sung để tăng:

- Khả năng phát hiện quả xanh và quả đỏ.
- Khả năng xử lý nhiều quả trong ảnh.
- Khả năng xử lý che khuất.
- Sự đa dạng về ánh sáng, góc nhìn, khoảng cách và blur.

### Hạn chế

- Không có class `fruit_turning`.
- Không được tự suy ra các mức trung gian.
- Cần kiểm tra trùng ảnh với nguồn khác.

---

## 3.5. AgRobTomato Dataset

**Trang Zenodo:**  
https://zenodo.org/records/5596799

**Tải ZIP:**  
https://zenodo.org/records/5596799/files/Dataset-Greenhouse_Tomato_AgRob.zip?download=1

**Loại bài toán:** Object Detection  
**Quy mô:** 449 ảnh  
**Độ phân giải:** 1280 × 720  
**Định dạng:** Pascal VOC  
**Số nhãn:** 4

### Các class

- Unriped
- Breaking Stage
- Reddish
- Riped

### Ánh xạ ban đầu

```yaml
Unriped: fruit_green_unripe
Breaking Stage: fruit_turning
Reddish: REVIEW_REQUIRED
Riped: fruit_ripe
```

`Reddish` phải được kiểm tra trực quan trước khi remap.

### Vai trò đề xuất

Nguồn bổ sung ảnh robot di chuyển trong hàng nhà kính.

### Rủi ro

- Số ảnh không lớn.
- Các frame có thể lấy từ cùng video.
- Chia ngẫu nhiên từng ảnh dễ gây data leakage.
- Cần chuyển Pascal VOC sang YOLO.

### Việc đội AI phải làm

- Xác định nhóm video hoặc phiên ghi hình.
- Chia split theo video/ngày.
- Chuyển annotation sang YOLO.
- Kiểm tra class `Reddish`.
- Dùng như nguồn bổ sung, không làm dataset duy nhất.

---

## 3.6. OpenField-BD Tomato Maturity

**Trang Zenodo:**  
https://zenodo.org/records/20176021

**Tải dữ liệu:**  
https://zenodo.org/records/20176021/files/OF-Tomato-BD.rar?download=1

**Loại bài toán:** Object Detection  
**Quy mô:** 600 ảnh  
**Dung lượng:** khoảng 2,4 GB  
**Số nhãn:** 3  
**License:** Chưa hiển thị rõ trong tài liệu nguồn đã kiểm tra

### Các class

- green
- half-ripe
- fully-ripe

### Ánh xạ dự kiến

```yaml
green: fruit_green_unripe
half-ripe: fruit_turning
fully-ripe: fruit_ripe
```

### Vai trò đề xuất

- Tập test ngoài miền.
- Nguồn bổ sung để tăng robustness.
- Kiểm tra model trong nền tự nhiên phức tạp.

### Hạn chế

- Môi trường đồng ruộng khác camera cố định trong nhà kính.
- License chưa rõ.
- Không nên đưa ngay vào dataset phát hành lại.

### Quyết định sử dụng

Ưu tiên dùng như tập test ngoài miền. Chỉ thêm vào train sau khi xác minh license và đánh giá tác động tới model nhà kính.

---

## 3.7. Tomatoes Dataset — Kaggle

**Đường dẫn:**  
https://www.kaggle.com/datasets/enalis/tomatoes-dataset

**Loại bài toán:** Image Classification  
**Quy mô:** 7.226 ảnh  
**Số nhãn:** 4  
**Giấy phép:** CC0 Public Domain

### Các class

- Unripe
- Ripe
- Old
- Damaged

### Vai trò đề xuất

Nguồn chính cho classifier phụ đánh giá tình trạng quả.

### Pipeline phù hợp

```text
YOLO phát hiện quả
→ crop từng quả
→ classifier:
   Unripe / Ripe / Old / Damaged
```

### Hạn chế

- Không có bounding box.
- Không thể đưa trực tiếp vào YOLO Object Detection.
- Muốn dùng một model YOLO duy nhất thì phải gắn box thủ công.

### Hai phương án sử dụng

#### Phương án ưu tiên

- Train classifier riêng.
- Nhận crop từ model detection.
- Đánh giá Old/Damaged trên từng quả.

#### Phương án một model YOLO

- Chọn ảnh phù hợp.
- Vẽ bounding box.
- Chuẩn hóa thành:
  - `fruit_green_unripe`
  - `fruit_ripe`
  - `fruit_overripe`
  - `fruit_damaged`

---

## 3.8. NCHU Tomato Maturity Recognition Dataset

**Đường dẫn đăng ký:**  
https://aidata.nchu.edu.tw/smarter/en/dataset/smarter_08_pmml02_0_gsdf_20240820_htry_jtyrkt

**Loại bài toán:** Object Detection  
**Định dạng:** YOLO  
**Số nhãn:** 6  
**Quyền sử dụng:** Cần đăng ký và được chấp thuận

### Các class

- Breaker
- Green
- Lightred
- Pink
- Red
- Turning

### Vai trò đề xuất

Nguồn bổ sung có điều kiện cho model sáu mức chín.

### Yêu cầu

- Không tải lên GitHub nếu điều khoản không cho phép.
- Không phân phối lại.
- Chỉ dùng sau khi được cấp quyền.
- Lưu bằng chứng chấp thuận cùng tài liệu dự án.

---

# 4. Bảng ưu tiên sử dụng

| Dataset | Vai trò | Mức ưu tiên |
|---|---|---:|
| Tomato Leaf Disease — Roboflow | Nguồn chính bệnh lá | Cao |
| Tomato Leaf Disease Detection — Zenodo | Bổ sung bệnh lá | Có điều kiện |
| Laboro Tomato | Nguồn chính độ chín ba class | Cao |
| TomatoPlantfactoryDataset | Bổ sung green/red và che khuất | Cao |
| AgRobTomato | Bổ sung ảnh robot nhà kính | Cao |
| Tomato Ripeness Detection — Morpheus | Bổ sung độ chín | Cần audit |
| OpenField-BD | Test ngoài miền/tăng robustness | Trung bình |
| Tomato Maturity Detection IEEE | Nguồn sáu mức chín | Giai đoạn sau |
| Kaggle Tomatoes Dataset | Classifier Old/Damaged | Cao cho model phụ |
| NCHU | Bổ sung sáu mức chín | Chỉ khi được cấp quyền |

---

# 5. Cấu trúc thư mục chuẩn

```text
ai/
├── datasets/
│   ├── raw/
│   │   ├── leaf_roboflow/
│   │   ├── leaf_zenodo/
│   │   ├── laboro_tomato/
│   │   ├── fruit_morpheus/
│   │   ├── tomato_maturity_ieee/
│   │   ├── tomato_plantfactory/
│   │   ├── agrob_tomato/
│   │   ├── openfield_bd/
│   │   ├── kaggle_tomato_states/
│   │   ├── nchu_maturity/
│   │   └── camera_imx179/
│   ├── interim/
│   │   ├── leaf_normalized/
│   │   ├── fruit_normalized/
│   │   ├── duplicate_groups/
│   │   ├── review_required/
│   │   └── rejected/
│   ├── processed/
│   │   ├── tomato_leaf_disease_v1/
│   │   └── tomato_ripeness_v1/
│   ├── manifests/
│   │   ├── sources.csv
│   │   ├── licenses.md
│   │   ├── class_mapping.yaml
│   │   ├── duplicate_report.csv
│   │   ├── rejected_images.csv
│   │   └── split_manifest.csv
│   └── README.md
├── scripts/
│   ├── inventory_dataset.py
│   ├── validate_yolo_labels.py
│   ├── voc_to_yolo.py
│   ├── remap_classes.py
│   ├── find_exact_duplicates.py
│   ├── find_near_duplicates.py
│   ├── split_by_group.py
│   └── generate_dataset_report.py
└── notebooks/
    ├── 01_leaf_dataset_audit.ipynb
    ├── 02_ripeness_dataset_audit.ipynb
    ├── 03_leaf_baseline.ipynb
    ├── 04_ripeness_baseline.ipynb
    └── 05_camera_domain_gap.ipynb
```

---

# 6. Quy trình xử lý dataset chi tiết

## Bước 1 — Tải và đóng băng dữ liệu gốc

Yêu cầu:

- Mỗi dataset nằm trong một thư mục riêng.
- Không chỉnh sửa trực tiếp dữ liệu trong `raw/`.
- Giữ file ZIP/RAR gốc.
- Tạo checksum SHA-256.
- Lưu README và license.
- Ghi ngày tải và phiên bản.

### Mẫu `sources.csv`

```csv
dataset_id,name,url,version,download_date,license,task,notes
leaf_roboflow,Tomato Leaf Disease,https://universe.roboflow.com/universitas-atma-jaya/tomato-leaf-disease-rxcft,,2026-08-05,CC BY 4.0,detection,
laboro_tomato,Laboro Tomato,https://github.com/laboroai/LaboroTomato,,2026-08-05,CC BY-NC-SA 4.0,detection+segmentation,
```

---

## Bước 2 — Kiểm kê tự động

Với mỗi dataset, thống kê:

- Tổng số ảnh.
- Tổng số label.
- Ảnh không có label.
- Label không có ảnh.
- Số object từng class.
- Kích thước và tỷ lệ ảnh.
- Bounding box quá nhỏ hoặc quá lớn.
- Box vượt ngoài ảnh.
- Box có width/height bằng 0.
- File ảnh hỏng.
- Ảnh negative.
- Số ảnh theo nguồn/video/cây/ngày.

Đầu ra:

- `dataset_inventory.csv`
- `class_distribution.csv`
- `image_size_distribution.csv`
- `label_error_report.csv`
- `dataset_audit.md`

---

## Bước 3 — Audit trực quan

Mỗi dataset cần kiểm tra tối thiểu:

- 100 ảnh ngẫu nhiên.
- 30–50 object mỗi class.
- Ảnh có nhiều object nhất.
- Ảnh có box nhỏ nhất.
- Ảnh thiếu sáng.
- Ảnh ngược sáng.
- Ảnh mờ.
- Ảnh che khuất.
- Ảnh nền phức tạp.
- Ảnh không có đối tượng.

### Câu hỏi audit

- Bounding box có đúng vị trí không?
- Class có đúng không?
- Có bỏ sót object không?
- Có box trùng không?
- Quy tắc khoanh vùng có nhất quán không?
- Bệnh lá được khoanh vùng triệu chứng hay toàn bộ lá?
- Quả được khoanh toàn bộ quả hay chỉ vùng màu?
- Có ảnh được gắn nhãn dựa trên suy đoán không?

---

## Bước 4 — Chuẩn hóa ontology

Tạo một file duy nhất:

```text
ai/datasets/manifests/class_mapping.yaml
```

### Mẫu

```yaml
leaf_roboflow:
  Early Blight: leaf_early_blight
  Late Blight: leaf_late_blight
  Leaf Mold: leaf_mold
  Septoria: leaf_septoria_spot
  Healthy: NEGATIVE_SAMPLE

laboro_tomato:
  b_green: fruit_green_unripe
  l_green: fruit_green_unripe
  b_half_ripened: fruit_turning
  l_half_ripened: fruit_turning
  b_fully_ripened: fruit_ripe
  l_fully_ripened: fruit_ripe

tomato_plantfactory:
  green: fruit_green_unripe
  red: fruit_ripe

agrob_tomato:
  Unriped: fruit_green_unripe
  Breaking Stage: fruit_turning
  Reddish: REVIEW_REQUIRED
  Riped: fruit_ripe
```

### Nguyên tắc

- Không remap class chỉ dựa vào tên.
- Class chưa chắc chắn phải để `REVIEW_REQUIRED`.
- Class ngoài MVP phải được ghi rõ `DROP` hoặc `FUTURE`.
- Mọi thay đổi ontology phải được version hóa.

---

## Bước 5 — Chuyển đổi annotation

Dataset Pascal VOC cần chuyển sang YOLO.

### Kiểm tra sau chuyển đổi

- Class ID đúng.
- Tọa độ chuẩn hóa nằm trong `[0, 1]`.
- Không có box width/height bằng 0.
- Không mất object.
- Số lượng object trước và sau chuyển đổi bằng nhau.
- Render ngẫu nhiên ảnh sau chuyển đổi để so sánh.

---

## Bước 6 — Phát hiện ảnh trùng

### Mức 1: Trùng chính xác

- SHA-256.
- MD5 chỉ dùng bổ sung.

### Mức 2: Gần trùng

- pHash.
- dHash.
- SSIM hoặc image embedding khi cần.

### Cần phát hiện

- Ảnh resize.
- Ảnh crop nhẹ.
- Ảnh đổi sáng.
- Frame liền nhau trong video.
- Ảnh augmentation từ cùng ảnh gốc.
- Ảnh được fork giữa các dataset public.

### Quy tắc

Toàn bộ ảnh thuộc cùng một `duplicate_group_id` phải nằm trong cùng một split.

---

## Bước 7 — Negative samples

### Cho bệnh lá

- Lá khỏe.
- Lá già tự nhiên.
- Lá bị côn trùng cắn nhưng không thuộc class mục tiêu.
- Lá thiếu dinh dưỡng nếu không thuộc class mục tiêu.
- Nền nhà kính không có bệnh.
- Ảnh không có cây.

### Cho quả

- Nền nhà kính không có quả.
- Lá và thân.
- Vật màu đỏ, xanh hoặc vàng không phải quả cà chua.
- Ảnh camera không có quả.
- Quả quá nhỏ hoặc quá mờ theo ngưỡng thống nhất.

Negative sample có thể có file label rỗng hoặc không có file label, nhưng pipeline phải dùng một quy ước thống nhất.

---

## Bước 8 — Chia train/validation/test chống leakage

Không chia ngẫu nhiên từng ảnh nếu chúng thuộc:

- Cùng video.
- Cùng cây.
- Cùng ngày chụp.
- Cùng zone.
- Cùng phiên camera.
- Cùng ảnh gốc trước augmentation.
- Cùng nhóm duplicate.

### Trường nhóm đề xuất

```text
source_dataset
video_id
plant_id
zone_id
capture_date
capture_session
duplicate_group_id
```

### Tỷ lệ tham khảo

- Train: 75%
- Validation: 15%
- Test: 10%

### Tập test phải có

- Ảnh public chưa dùng để tuning.
- Ảnh camera IMX179.
- Ảnh nhà kính thực tế.
- Ảnh khó, che khuất, thiếu sáng.
- Ảnh negative.
- Một phần dữ liệu ngoài miền để kiểm tra robustness.

---

## Bước 9 — Cân bằng class

Đánh giá theo số object, không chỉ số ảnh.

### Thực hiện

- Thống kê object từng class.
- Tính tỷ lệ class lớn nhất/class nhỏ nhất.
- Bổ sung ảnh thật cho class thiếu.
- Không lạm dụng nhân bản ảnh.
- Chỉ augmentation trên train.
- Không augmentation validation và test.
- Báo cáo Precision/Recall/AP riêng từng class.

---

## Bước 10 — Thu thập ảnh camera IMX179

Đây là bước bắt buộc để giảm domain gap.

### Điều kiện chụp

- Sáng, trưa, chiều, tối.
- Đèn LED bật và tắt.
- Camera gần và xa.
- Góc thẳng và góc lệch.
- Lá/quả bị che.
- Nhiều quả trong ảnh.
- Ảnh không có quả.
- Lá khỏe.
- Lá bệnh nhẹ, vừa và nặng.
- Quả xanh, turning và ripe.
- Ảnh mờ hoặc lỗi camera.
- Nền nhà kính thực tế.

### Cách sử dụng

1. Tạo test set độc lập.
2. Fine-tune model.
3. Đánh giá domain gap.
4. Chọn confidence threshold thực tế.
5. Đánh giá hiệu năng trên Raspberry Pi.

---

## Bước 11 — Tạo dataset phiên bản 1

### `tomato_leaf_disease_v1`

```yaml
names:
  0: leaf_early_blight
  1: leaf_late_blight
  2: leaf_mold
  3: leaf_septoria_spot
```

### `tomato_ripeness_v1`

```yaml
names:
  0: fruit_green_unripe
  1: fruit_turning
  2: fruit_ripe
```

### Mỗi phiên bản phải chứa

- `data.yaml`
- `README.md`
- `dataset_report.md`
- `split_manifest.csv`
- `class_distribution.csv`
- `source_distribution.csv`
- `licenses.md`
- `class_mapping.yaml`
- `duplicate_report.csv`
- `rejected_images.csv`

---

## Bước 12 — Train baseline

Train riêng:

1. Model bệnh lá.
2. Model quả và độ chín.

### Cấu hình tham khảo

```yaml
model: YOLO nano
imgsz: 640
epochs: 100
batch: 8 hoặc 16
patience: 20
pretrained: true
seed: 42
```

Không tăng epoch trước khi phân tích:

- False positive.
- False negative.
- Class confusion.
- Lỗi label.
- Source bias.
- Domain gap.

---

## Bước 13 — Đánh giá theo nguồn

Không chỉ báo cáo metric tổng.

### Các tập cần đánh giá riêng

- Roboflow leaf test.
- Zenodo leaf test.
- Laboro test.
- Plantfactory test.
- AgRob test.
- OpenField test.
- IMX179 test.

### Metric

- Precision.
- Recall.
- mAP@0.5.
- mAP@0.5:0.95.
- Per-class AP.
- Confusion matrix.
- Inference time.
- FPS.
- RAM/CPU/nhiệt độ trên Raspberry Pi.

---

# 7. Quy tắc gắn nhãn thống nhất

## 7.1. Bệnh lá

- Khoanh vùng biểu hiện bệnh nhìn thấy được.
- Không khoanh quá rộng chứa nhiều nền.
- Không suy đoán bệnh nếu biểu hiện không rõ.
- Tạo box riêng cho các vùng bệnh tách biệt.
- Ảnh không chắc chắn đưa vào `review_required`.
- Chuẩn bị ảnh mẫu đúng/sai cho từng class.

## 7.2. Quả và độ chín

- Khoanh toàn bộ quả.
- Gắn nhãn theo phần bề mặt nhìn thấy.
- Không suy đoán theo tuổi cây.
- Quả bị che vẫn gắn nhãn nếu nhận diện đủ chắc chắn.
- Bỏ qua quả quá nhỏ theo ngưỡng thống nhất.

### Tiêu chí màu tham khảo

- `fruit_green_unripe`: phần lớn bề mặt màu xanh.
- `fruit_turning`: xuất hiện màu vàng, cam hoặc đỏ nhưng chưa chiếm phần lớn bề mặt.
- `fruit_ripe`: màu chín chiếm phần lớn bề mặt nhìn thấy.

## 7.3. Quả quá chín và hư hỏng

### `fruit_overripe`

- Màu quá đậm.
- Bề mặt nhăn hoặc dấu hiệu già có thể quan sát.
- Chưa chắc có tổn thương hư hỏng rõ.

### `fruit_damaged`

- Nứt quả.
- Thối.
- Mốc.
- Đốm hư hại rõ.
- Biến dạng nghiêm trọng.
- Tổn thương cơ học rõ.

Không gộp mọi khuyết điểm nhỏ vào `fruit_damaged`.

---

# 8. Checklist cho đội AI

## Giai đoạn A — Chuẩn bị

- [ ] Tải các dataset được chọn.
- [ ] Lưu file nén gốc.
- [ ] Tạo SHA-256.
- [ ] Lưu README và license.
- [ ] Tạo `sources.csv`.
- [ ] Ghi phiên bản và ngày tải.

## Giai đoạn B — Audit

- [ ] Thống kê ảnh, label và object.
- [ ] Kiểm tra ảnh lỗi.
- [ ] Kiểm tra box lỗi.
- [ ] Xem ít nhất 100 ảnh mỗi dataset.
- [ ] Ghi ảnh cần review.
- [ ] Ghi ảnh bị loại.
- [ ] Xác minh class `Reddish`.
- [ ] Xác minh metadata bộ Morpheus.

## Giai đoạn C — Chuẩn hóa

- [ ] Chốt ontology bệnh lá.
- [ ] Chốt ontology độ chín.
- [ ] Tạo `class_mapping.yaml`.
- [ ] Chuyển VOC sang YOLO.
- [ ] Remap class.
- [ ] Chuyển Healthy thành negative.
- [ ] Không tự động gộp class không tương đương.

## Giai đoạn D — Chống leakage

- [ ] Tính exact hash.
- [ ] Tính perceptual hash.
- [ ] Nhóm frame theo video.
- [ ] Nhóm ảnh theo cây/ngày/zone.
- [ ] Tạo `duplicate_report.csv`.
- [ ] Chia split theo group.

## Giai đoạn E — Dataset v1

- [ ] Tạo `tomato_leaf_disease_v1`.
- [ ] Tạo `tomato_ripeness_v1`.
- [ ] Tạo `data.yaml`.
- [ ] Tạo manifest.
- [ ] Tạo báo cáo phân bố class.
- [ ] Tạo báo cáo nguồn.
- [ ] Tạo báo cáo license.

## Giai đoạn F — Camera IMX179

- [ ] Thu thập ảnh thực tế.
- [ ] Gắn nhãn.
- [ ] Tạo test set độc lập.
- [ ] Fine-tune.
- [ ] Đánh giá domain gap.

## Giai đoạn G — Baseline

- [ ] Train model bệnh lá.
- [ ] Train model độ chín.
- [ ] Phân tích false positive.
- [ ] Phân tích false negative.
- [ ] Phân tích confusion matrix.
- [ ] Đánh giá theo từng nguồn.
- [ ] Sửa dữ liệu trước khi tăng epoch.
- [ ] Benchmark Raspberry Pi.

---

# 9. Definition of Done cho dataset v1

Dataset được coi là hoàn thiện phiên bản 1 khi:

- Có nguồn và license rõ ràng.
- Có checksum file gốc.
- Có class mapping.
- Có thống kê ảnh và object.
- Không có label vượt ngoài ảnh.
- Không có ảnh lỗi nghiêm trọng.
- Không có ảnh trùng giữa các split.
- Có negative sample.
- Có ảnh camera IMX179.
- Có test set độc lập.
- Có manifest tái tạo split.
- Có dataset version.
- Có báo cáo phân bố class.
- Có báo cáo chất lượng nhãn.
- Có baseline model.
- Có metric theo class.
- Có metric theo nguồn.
- Có phân tích lỗi.

---

# 10. Lộ trình triển khai đề xuất

## Sprint 1 — Tải và audit

- Tải Laboro, Plantfactory, AgRob.
- Tải hai dataset bệnh lá.
- Tải và audit bộ Morpheus.
- Kiểm tra license.
- Thống kê ảnh và object.
- Audit nhãn.
- Xác định ảnh trùng.

## Sprint 2 — Chuẩn hóa

- Chốt class.
- Chuyển VOC sang YOLO.
- Remap label.
- Tạo negative sample.
- Chia split chống leakage.
- Tạo dataset v1.

## Sprint 3 — Baseline

- Train bệnh lá.
- Train độ chín ba class.
- Đánh giá từng nguồn.
- Phân tích lỗi.
- Sửa nhãn và bổ sung dữ liệu.

## Sprint 4 — Camera IMX179

- Thu thập ảnh thật.
- Gắn nhãn.
- Tạo test set.
- Fine-tune.
- Benchmark Raspberry Pi.

## Sprint 5 — Mở rộng

Chỉ thực hiện sau khi baseline ổn định:

- Sáu mức chín.
- `fruit_overripe`.
- `fruit_damaged`.
- Classifier phụ Old/Damaged.
- Thử model đa lớp.
- So sánh một model và hai model riêng.

---

# 11. Kết luận

Hướng làm phù hợp nhất:

1. Bệnh lá dùng bốn class.
2. Độ chín bắt đầu với ba class.
3. Laboro là nguồn nền cho độ chín.
4. Plantfactory và AgRob dùng để bổ sung.
5. Bộ Morpheus phải được audit trước khi gộp.
6. IEEE và NCHU dành cho sáu mức chín ở giai đoạn sau.
7. Kaggle dùng cho classifier Old/Damaged hoặc phải gắn box lại.
8. Không dùng dataset có license chưa rõ trong bản phát hành.
9. Không chia ngẫu nhiên frame cùng video.
10. Bắt buộc bổ sung ảnh camera IMX179.
11. Ưu tiên chất lượng nhãn, chống leakage và khả năng tái tạo trước khi tăng epoch hoặc đổi model.
