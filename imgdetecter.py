import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from sklearn.linear_model import LinearRegression

home = os.path.expanduser("~")
image_path = os.path.join(home, '3dp-webcam-detection/testpicturs/Print_failure-3.jpg')

image = cv2.imread(image_path)
if image is None:
    print("画像の読み込みに失敗しました。ファイルパスを確認してください。")
else:
    print("画像の読み込みに成功しました。")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
#plt.show()

blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 9.5, -1],
                           [-1, -1, -1]])
sharpened_image = cv2.filter2D(blurred_image, -1, sharpen_kernel)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
#plt.show()

gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2HSV)
_, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_wakakusa = np.array([70, 50, 50])
upper_wakakusa = np.array([78, 255, 255])

hsv_image = image.copy()
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)
wakakusa_mask = cv2.inRange(hsv_image, lower_wakakusa, upper_wakakusa)

red_binary_image = cv2.bitwise_and(image, image, mask=red_mask)
wakakusa_binary_image = cv2.bitwise_and(image, image, mask=wakakusa_mask)

plt.imshow(binary_image, cmap='gray')
#plt.show()
plt.imshow(cv2.cvtColor(red_binary_image, cv2.COLOR_BGR2RGB))
#plt.show()
plt.imshow(cv2.cvtColor(wakakusa_binary_image, cv2.COLOR_BGR2RGB))
#plt.show()

contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
contour_image = image.copy()
cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
#plt.show()

contours, _ = cv2.findContours(wakakusa_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    printed_matter = max(contours, key=cv2.contourArea)
    contour_image2 = image.copy()
    cv2.drawContours(contour_image2, [printed_matter], -1, (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(contour_image2, cv2.COLOR_BGR2RGB))
    #plt.show()
else:
    print("輪郭が検出されませんでした。")

perimeter = cv2.arcLength(largest_contour, True)
approximation = cv2.approxPolyDP(largest_contour, 0.1 * perimeter, True)
if len(approximation) == 3:
    cv2.drawContours(image, [approximation], 0, (0, 255, 0), 5)
    for point in approximation:
        cv2.circle(image, tuple(point[0]), 5, (255, 0, 0), -1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.show()

def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

edge_lengths = [calculate_distance(approximation[i][0], approximation[(i+1) % 3][0]) for i in range(3)]
longest_edge = max(edge_lengths)
print("最長辺の長さ:", longest_edge)

centroid = np.mean(approximation, axis=0, dtype=int)[0]

min_edge_length = 116.18089343777659
max_edge_length = 267.4210163767986
max_offset_x    = -190
min_offset_x    = -85
max_offset_y    = 150
min_offset_y    = 55

offset_x = min_offset_x + (max_offset_x - min_offset_x) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)
offset_y = min_offset_y + (max_offset_y - min_offset_y) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)

nozzle_offset_x = offset_x
nozzle_offset_y = offset_y
nozzle_position = (centroid[0]+math.floor(nozzle_offset_x), centroid[1] + math.floor(nozzle_offset_y))
nozzle_image = image.copy()
cv2.circle(nozzle_image, nozzle_position, 10, (255, 0, 0), -1)
plt.imshow(cv2.cvtColor(nozzle_image, cv2.COLOR_BGR2RGB))
#plt.show()

contours = [printed_matter]
topmost = min(contours[0], key=lambda x: x[0][1])
y_range = 10
print_surface = [pt for pt in contours[0] if topmost[0][1] - y_range <= pt[0][1] <= topmost[0][1] + y_range]
print("印刷平面として定義された輪郭ポイント:")
print(print_surface)

model = LinearRegression()
x = np.array([pt[0][0] for pt in print_surface]).reshape(-1, 1)
y = np.array([pt[0][1] for pt in print_surface])
model.fit(x, y)

x_start = min(x)[0]
x_end = max(x)[0]
y_start = model.predict([[x_start]])[0]
y_end = model.predict([[x_end]])[0]
cv2.line(image, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 0, 255), 2)
for pt in print_surface:
    cv2.circle(image, tuple(pt[0]), 2, (255, 0, 0), -1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.show()

m = model.coef_[0]
c = model.intercept_
x0, y0 = nozzle_position[0], nozzle_position[1]
distance = abs(m*x0 - y0 + c) / np.sqrt(m**2 + 1)
threshold = 10
print(distance)
if distance > threshold:
    print("警告: ノズルの位置と印刷平面が離れすぎており、印刷が失敗している可能性があります。")
else:
    print("ノズルと印刷平面の距離は正常範囲内です。")

plt.show()