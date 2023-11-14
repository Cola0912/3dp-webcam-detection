import cv2
import numpy as np
import os

# 画像のパスを展開
image_path = os.path.expanduser("~/princam/test2.jpg")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# エラーチェック: 画像が正しく読み込まれたかを確認
if img is None:
    print("Error: Could not read the image.")
    exit()

# CLAHE (Contrast Limited Adaptive Histogram Equalization) の適用
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

# アダプティブスレッショルディングの適用
binary = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# エッジ検出
edges = cv2.Canny(binary, 50, 150)

# 輪郭を検出
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ノズルの先端と思われる領域を認識
for contour in contours:
    if cv2.contourArea(contour) > 50:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 結果の表示
cv2.imshow("Nozzle Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
