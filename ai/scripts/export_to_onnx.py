"""Xuat model YOLOv8 (.pt) sang ONNX (.onnx) de chay trong trinh duyet voi
ai/scripts/leaf_disease_tester.html.

Cach dung:
    pip install ultralytics
    python export_to_onnx.py duong/dan/toi/best.pt

Chay o dau cung duoc, mien la co Python + ultralytics:
  - Local (neu may da co Python).
  - Hoac them 1 cell moi vao cuoi notebook train tren Kaggle (da co san ultralytics),
    roi tai file .onnx sinh ra ve may qua panel Output cua Kaggle.
  - Hoac Google Colab.

Output: file .onnx cung thu muc voi file .pt dau vao, vi du best.onnx.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Dung: python export_to_onnx.py duong_dan_toi_model.pt")
    sys.exit(1)

pt_path = Path(sys.argv[1])
if not pt_path.exists():
    print(f"Khong tim thay file: {pt_path}")
    sys.exit(1)

model = YOLO(str(pt_path))
onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
print(f"\nDa xuat: {onnx_path}")
print("Mo ai/scripts/leaf_disease_tester.html trong trinh duyet, chon file .onnx nay o muc 1.")
