import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import time
import sys

# カメラの初期化
camera_index = 4  # 使用するカメラのインデックスを指定
cap = cv2.VideoCapture(camera_index)

# プリンター固有の値（そのまま使用）
min_edge_length = 116.18089343777659
max_edge_length = 267.4210163767986
min_offset_x    = -85
max_offset_x    = -190
min_offset_y    = 55
max_offset_y    = 150

# 色範囲の初期値
red_h_min_init = 0
red_h_max_init = 180
red_s_min_init = 227
red_s_max_init = 255
red_v_min_init = 82
red_v_max_init = 255

green_h_min_init = 51
green_h_max_init = 96
green_s_min_init = 161
green_s_max_init = 255
green_v_min_init = 0
green_v_max_init = 255

# 距離のしきい値初期値
threshold_x_min_init = 25
threshold_x_max_init = 25
threshold_y_min_init = 35
threshold_y_max_init = 35

# 明るさと先鋭化の初期値
brightness_init = 65
sharpen_init = 0

# 緑色の輪郭面積のしきい値初期値
green_area_threshold_init = 70
plane_movement_threshold_init = 3
nozzle_movement_threshold_init = 20
distance_change_threshold_init = 14

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# 単位時間（3秒）ごとの失敗率計算用変数の初期化
unit_time = 3
start_time = time.time()
success_frames = 0
failure_frames = 0
nozzle_detected_in_interval = False

# 印刷平面の位置とノズル位置を記録するリスト
plane_positions = []
nozzle_positions = []
nozzle_plane_distances = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました")
        break

    lower_red = np.array([red_h_min_init, red_s_min_init, red_v_min_init])
    upper_red = np.array([red_h_max_init, red_s_max_init, red_v_max_init])
    lower_green = np.array([green_h_min_init, green_s_min_init, green_v_min_init])
    upper_green = np.array([green_h_max_init, green_s_max_init, green_v_max_init])

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    message = "輪郭が検出されません"
    nozzle_detected = False
    plane_detected = False
    is_failure = True

    if contours_red:
        largest_contour = max(contours_red, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.03 * perimeter
        approximation = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approximation) == 3:
            centroid = np.mean(approximation, axis=0, dtype=int)[0]
            edge_lengths = [calculate_distance(approximation[i][0], approximation[(i+1)%3][0]) for i in range(3)]
            longest_edge = max(edge_lengths)

            offset_x = min_offset_x + (max_offset_x - min_offset_x) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)
            offset_y = min_offset_y + (max_offset_y - min_offset_y) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)
            nozzle_position = (int(centroid[0] + offset_x), int(centroid[1] + offset_y))

            nozzle_detected = True
            nozzle_positions.append(nozzle_position)

            if len(nozzle_positions) >= 2:
                nozzle_movement = calculate_distance(np.array(nozzle_positions[-1]), np.array(nozzle_positions[-2]))
            else:
                nozzle_movement = 0

            contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours_green = [cnt for cnt in contours_green if cv2.contourArea(cnt) >= green_area_threshold_init]

            if valid_contours_green:
                printed_matter = max(valid_contours_green, key=cv2.contourArea)
                topmost = tuple(printed_matter[printed_matter[:,:,1].argmin()][0])
                print_surface = [pt for pt in printed_matter if topmost[1] - 10 <= pt[0][1] <= topmost[1] + 10]

                if len(print_surface) > 1:
                    x_points = np.array([pt[0][0] for pt in print_surface])
                    y_points = np.array([pt[0][1] for pt in print_surface])
                    x = x_points.reshape(-1,1)
                    y = y_points
                    model = LinearRegression()
                    model.fit(x, y)
                    x_start = min(x_points)
                    x_end = max(x_points)
                    y_start = int(model.predict([[x_start]])[0])
                    y_end = int(model.predict([[x_end]])[0])

                    plane_detected = True
                    plane_position = ((x_start + x_end) / 2, (y_start + y_end) / 2)
                    plane_positions.append(plane_position)

                    if len(plane_positions) >= 2:
                        plane_movement = calculate_distance(np.array(plane_positions[-1]), np.array(plane_positions[-2]))
                    else:
                        plane_movement = 0

                    nozzle_plane_distance = calculate_distance(np.array(nozzle_position), np.array(plane_position))
                    nozzle_plane_distances.append(nozzle_plane_distance)

                    if len(nozzle_plane_distances) >= 2:
                        distance_change = abs(nozzle_plane_distances[-1] - nozzle_plane_distances[-2])
                    else:
                        distance_change = 0

                    x_min_line = x_start
                    x_max_line = x_end
                    y_min_line = min(y_start, y_end)
                    y_max_line = max(y_start, y_end)

                    x0, y0 = nozzle_position

                    condition1 = x0 <= x_max_line + threshold_x_max_init
                    condition2 = x0 >= x_min_line - threshold_x_min_init
                    condition3 = y0 <= y_max_line + threshold_y_max_init
                    condition4 = y0 >= y_min_line - threshold_y_min_init
                    condition5 = plane_movement <= plane_movement_threshold_init
                    condition6 = nozzle_movement >= nozzle_movement_threshold_init
                    condition7 = distance_change <= distance_change_threshold_init

                    if condition1 and condition2 and condition3 and condition4:
                        if condition6 and condition7:
                            message = "樹脂付着"
                            is_failure = True
                        elif condition5:
                            message = "印刷は正常です"
                            is_failure = False
                        else:
                            message = "樹脂付着"
                            is_failure = True
                    else:
                        message = "ノズル位置がしきい値を超えています"
                        is_failure = True
                else:
                    message = "データ不足のため判定不可"
                    is_failure = True
            else:
                message = "印刷物が検出されません"
                is_failure = True
        else:
            message = "ノズルが検出されません"
            is_failure = True
    else:
        message = "輪郭が検出されません"
        is_failure = True

    if nozzle_detected and plane_detected:
        nozzle_detected_in_interval = True
        if is_failure:
            failure_frames += 1
        else:
            success_frames += 1

    current_time = time.time()
    if current_time - start_time >= unit_time:
        sys.stdout.write("\r\033[K")  # 行全体をクリア
        if nozzle_detected_in_interval:
            total_detected_frames = success_frames + failure_frames
            if total_detected_frames > 0:
                failure_percentage = (failure_frames / total_detected_frames) * 100
                sys.stdout.write(f"\r失敗率: {failure_percentage:.1f}% - メッセージ: {message}")
                sys.stdout.flush()
            else:
                sys.stdout.write("\r失敗率: 不明 - メッセージ: 不明")
                sys.stdout.flush()
        else:
            sys.stdout.write("\r失敗率: 不明 - メッセージ: 不明")
            sys.stdout.flush()

        start_time = current_time
        success_frames = 0
        failure_frames = 0
        nozzle_detected_in_interval = False
        plane_positions = []
        nozzle_positions = []
        nozzle_plane_distances = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
