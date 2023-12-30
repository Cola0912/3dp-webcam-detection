import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
import argparse

def process_image(image):
    # ガウシアンぼかしを適用
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    sharpened_image = cv2.filter2D(blurred_image, -1, np.array([[-1, -1, -1],
                                                               [-1, 9.5, -1],
                                                               [-1, -1, -1]]))
    gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2HSV)
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # 色範囲の定義
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_wakakusa = np.array([70, 50, 50])
    upper_wakakusa = np.array([78, 255, 255])

    # 赤色部分のマスク
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 若草色部分のマスク
    wakakusa_mask = cv2.inRange(hsv_image, lower_wakakusa, upper_wakakusa)

    # マスクから輪郭を検出
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_wakakusa, _ = cv2.findContours(wakakusa_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ノズル（赤色三角形）の位置を特定
    if contours_red:
        largest_red_contour = max(contours_red, key=cv2.contourArea)
        peri = cv2.arcLength(largest_red_contour, True)
        approximation = cv2.approxPolyDP(largest_red_contour, 0.02 * peri, True)
        
        if len(approximation) == 3:  # Detecting triangle for nozzle
            M = cv2.moments(largest_red_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                nozzle_position = (cx, cy)
                
                # Calculate edge lengths and the longest edge
                edge_lengths = [calculate_distance(approximation[i][0], approximation[(i + 1) % 3][0]) for i in range(3)]
                longest_edge = max(edge_lengths)
                
                # Calculate offsets
                offset_x = min_offset_x + (max_offset_x - min_offset_x) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)
                offset_y = min_offset_y + (max_offset_y - min_offset_y) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)
                nozzle_position = (cx + int(offset_x), cy + int(offset_y))
                cv2.circle(image, nozzle_position, 10, (255, 0, 0), -1)  # Mark nozzle position
            else:
                nozzle_position = None
    else:
        nozzle_position = None

    # 印刷平面（若草色）の位置を特定
    if contours_wakakusa:
        largest_wakakusa_contour = max(contours_wakakusa, key=cv2.contourArea)
        # 最も高いY座標を持つポイントを見つける
        topmost = tuple(largest_wakakusa_contour[largest_wakakusa_contour[:,:,1].argmin()][0])

        # Y座標が最も高いポイントから±10ピクセルの範囲にある輪郭ポイントを探す
        y_range = 10
        print_surface = [pt for pt in largest_wakakusa_contour if topmost[1] - y_range <= pt[0][1] <= topmost[1] + y_range]

        # 線形回帰モデルを作成
        model = LinearRegression()
        x = np.array([pt[0][0] for pt in print_surface]).reshape(-1, 1)  # X座標を抽出
        y = np.array([pt[0][1] for pt in print_surface])  # Y座標を抽出
        model.fit(x, y)
        m = model.coef_[0]  # slope
        c = model.intercept_  # y-intercept

        # 回帰直線を描画
        x_start = min(x)[0]
        x_end = max(x)[0]
        y_start = model.predict([[x_start]])[0]
        y_end = model.predict([[x_end]])[0]
        cv2.line(image, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 255), 2)

        for pt in print_surface:
            cv2.circle(image, tuple(pt[0]), 2, (0, 0, 255), -1)  # Mark print surface points
    else:
        print_surface = None

    # Calculate the distance between nozzle and print surface if both are detected
    if nozzle_position and print_surface is not None:
        x0, y0 = nozzle_position
        distance = abs(m * x0 - y0 + c) / np.sqrt(m ** 2 + 1)  # Perpendicular distance

        # Print warning based on the threshold
        if distance > threshold:
            print(f"警告: ノズルの位置と印刷平面が離れすぎており、印刷が失敗している可能性があります。距離: {distance}")
        else:
            print("ノズルと印刷平面の距離は正常範囲内です。")

    return image

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# プリンター固有の値
min_edge_length = 116.18089343777659
max_edge_length = 267.4210163767986
max_offset_x    = -190
min_offset_x    = -85
max_offset_y    = 150
min_offset_y    = 55
threshold       = 10 # ノズルと印刷平面との距離のしきい値　これより距離が遠ければ失敗してる可能性が高い。



def main(input_image_path, output_image_path):
    image = cv2.imread(input_image_path)
    if image is None:
        print("画像が見つかりません。")
        return

    processed_image = process_image(image)
    cv2.imwrite(output_image_path, processed_image)  # 処理済み画像を指定された出力パスに保存

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image.')
    parser.add_argument('input_image', help='Path to the input image file')
    parser.add_argument('output_image', help='Path to the output image file')

    args = parser.parse_args()

    main(args.input_image, args.output_image)