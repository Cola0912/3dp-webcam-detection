import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageDraw, ImageFont
import time  # 追加：時間計測のため

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

# フォントの設定
font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"  # IPAゴシックを使用
font_size = 24
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    print("指定されたフォントが見つかりません。デフォルトフォントを使用します。")
    font = ImageFont.load_default()

# ウィンドウの作成
cv2.namedWindow('Result')
cv2.namedWindow('Red Mask')
cv2.namedWindow('Green Mask')
cv2.namedWindow('Trackbars')

# トラックバーのコールバック関数（何もしない）
def nothing(x):
    pass

red_h_min_init = 0
red_h_max_init = 180
red_s_min_init = 227
red_s_max_init = 255
red_v_min_init = 82
red_v_max_init = 255

# 緑色の範囲初期値
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
brightness_init = 65  # 50〜150 (50=0.5, 150=1.5)
sharpen_init = 0      # 0〜20 (10=1.0)

# 緑色の輪郭面積のしきい値初期値
green_area_threshold_init = 70  # 面積がこれ未満の輪郭を無視

# 印刷平面位置変化のしきい値初期値
plane_movement_threshold_init = 3  # しきい値ピクセル

# ノズル移動量と距離変化量のしきい値初期値
nozzle_movement_threshold_init = 20
distance_change_threshold_init = 14

# 赤色のトラックバーを作成（初期値を設定）
cv2.createTrackbar('Red Hue Min', 'Trackbars', red_h_min_init, 180, nothing)
cv2.createTrackbar('Red Hue Max', 'Trackbars', red_h_max_init, 180, nothing)
cv2.createTrackbar('Red Sat Min', 'Trackbars', red_s_min_init, 255, nothing)
cv2.createTrackbar('Red Sat Max', 'Trackbars', red_s_max_init, 255, nothing)
cv2.createTrackbar('Red Val Min', 'Trackbars', red_v_min_init, 255, nothing)
cv2.createTrackbar('Red Val Max', 'Trackbars', red_v_max_init, 255, nothing)

# 緑色のトラックバーを作成（初期値を設定）
cv2.createTrackbar('Green Hue Min', 'Trackbars', green_h_min_init, 180, nothing)
cv2.createTrackbar('Green Hue Max', 'Trackbars', green_h_max_init, 180, nothing)
cv2.createTrackbar('Green Sat Min', 'Trackbars', green_s_min_init, 255, nothing)
cv2.createTrackbar('Green Sat Max', 'Trackbars', green_s_max_init, 255, nothing)
cv2.createTrackbar('Green Val Min', 'Trackbars', green_v_min_init, 255, nothing)
cv2.createTrackbar('Green Val Max', 'Trackbars', green_v_max_init, 255, nothing)

# 4つのしきい値トラックバーを追加
cv2.createTrackbar('Threshold X Min', 'Trackbars', threshold_x_min_init, 200, nothing)
cv2.createTrackbar('Threshold X Max', 'Trackbars', threshold_x_max_init, 200, nothing)
cv2.createTrackbar('Threshold Y Min', 'Trackbars', threshold_y_min_init, 200, nothing)
cv2.createTrackbar('Threshold Y Max', 'Trackbars', threshold_y_max_init, 200, nothing)

# 明るさのトラックバーを追加
cv2.createTrackbar('Brightness', 'Trackbars', brightness_init, 150, nothing)

# 先鋭化のトラックバーを追加
cv2.createTrackbar('Sharpen', 'Trackbars', sharpen_init, 20, nothing)

# 緑色の輪郭面積のしきい値トラックバーを追加
cv2.createTrackbar('Green Area Threshold', 'Trackbars', green_area_threshold_init, 100000, nothing)

# 印刷平面位置変化のしきい値トラックバーを追加
cv2.createTrackbar('Plane Movement Threshold', 'Trackbars', plane_movement_threshold_init, 100, nothing)

# ノズル移動量と距離変化量のしきい値トラックバーを追加
cv2.createTrackbar('Nozzle Movement Threshold', 'Trackbars', nozzle_movement_threshold_init, 100, nothing)
cv2.createTrackbar('Distance Change Threshold', 'Trackbars', distance_change_threshold_init, 100, nothing)

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# 単位時間（10秒）ごとの失敗率計算用変数の初期化
unit_time = 10  # 秒
start_time = time.time()
success_frames = 0
failure_frames = 0
nozzle_detected_in_interval = False

percentage_message = "不明"
percentage_text_color = (255, 255, 255)

# 印刷平面の位置とノズル位置を記録するリスト
plane_positions = []
nozzle_positions = []
nozzle_plane_distances = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました")
        break

    # トラックバーの値を取得
    red_h_min = cv2.getTrackbarPos('Red Hue Min', 'Trackbars')
    red_h_max = cv2.getTrackbarPos('Red Hue Max', 'Trackbars')
    red_s_min = cv2.getTrackbarPos('Red Sat Min', 'Trackbars')
    red_s_max = cv2.getTrackbarPos('Red Sat Max', 'Trackbars')
    red_v_min = cv2.getTrackbarPos('Red Val Min', 'Trackbars')
    red_v_max = cv2.getTrackbarPos('Red Val Max', 'Trackbars')

    green_h_min = cv2.getTrackbarPos('Green Hue Min', 'Trackbars')
    green_h_max = cv2.getTrackbarPos('Green Hue Max', 'Trackbars')
    green_s_min = cv2.getTrackbarPos('Green Sat Min', 'Trackbars')
    green_s_max = cv2.getTrackbarPos('Green Sat Max', 'Trackbars')
    green_v_min = cv2.getTrackbarPos('Green Val Min', 'Trackbars')
    green_v_max = cv2.getTrackbarPos('Green Val Max', 'Trackbars')

    threshold_x_min = cv2.getTrackbarPos('Threshold X Min', 'Trackbars')
    threshold_x_max = cv2.getTrackbarPos('Threshold X Max', 'Trackbars')
    threshold_y_min = cv2.getTrackbarPos('Threshold Y Min', 'Trackbars')
    threshold_y_max = cv2.getTrackbarPos('Threshold Y Max', 'Trackbars')

    brightness = cv2.getTrackbarPos('Brightness', 'Trackbars') / 100  # 0.5〜1.5
    sharpen_factor = cv2.getTrackbarPos('Sharpen', 'Trackbars') / 10  # 0.0〜2.0

    green_area_threshold = cv2.getTrackbarPos('Green Area Threshold', 'Trackbars')

    plane_movement_threshold = cv2.getTrackbarPos('Plane Movement Threshold', 'Trackbars')

    nozzle_movement_threshold = cv2.getTrackbarPos('Nozzle Movement Threshold', 'Trackbars')
    distance_change_threshold = cv2.getTrackbarPos('Distance Change Threshold', 'Trackbars')

    # 色範囲の定義
    lower_red = np.array([red_h_min, red_s_min, red_v_min])
    upper_red = np.array([red_h_max, red_s_max, red_v_max])

    lower_green = np.array([green_h_min, green_s_min, green_v_min])
    upper_green = np.array([green_h_max, green_s_max, green_v_max])

    # 明るさの調整
    frame_brightness = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)

    # ノイズ軽減
    blurred_image = cv2.GaussianBlur(frame_brightness, (7, 7), 0)

    # 先鋭化の調整
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9 + sharpen_factor, -1],
                               [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, sharpen_kernel)

    # HSV色空間に変換
    hsv_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2HSV)

    # マスクの作成
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # マスクのノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # 赤色マスクから輪郭を検出
    contours_red, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    message = "輪郭が検出されません"
    text_color = (0, 0, 255)  # 赤色

    output_frame = frame.copy()

    nozzle_detected = False  # ノズル位置が検出されたかを記録
    plane_detected = False   # 印刷平面が検出されたかを記録
    is_failure = True        # 失敗フラグ

    if contours_red:
        # 一番大きい輪郭を選択
        largest_contour = max(contours_red, key=cv2.contourArea)

        # 輪郭の近似
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon_multiplier = 0.03  # 閾値を変更可能
        epsilon = epsilon_multiplier * perimeter
        approximation = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approximation) == 3:
            # 三角形を描画
            cv2.drawContours(output_frame, [approximation], 0, (0, 255, 0), 2)

            # 重心の計算
            centroid = np.mean(approximation, axis=0, dtype=int)[0]

            # 辺の長さを計算
            edge_lengths = [calculate_distance(approximation[i][0], approximation[(i+1)%3][0]) for i in range(3)]
            longest_edge = max(edge_lengths)

            # オフセットの計算
            offset_x = min_offset_x + (max_offset_x - min_offset_x) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)
            offset_y = min_offset_y + (max_offset_y - min_offset_y) * (longest_edge - min_edge_length) / (max_edge_length - min_edge_length)

            # ノズルの位置を計算
            nozzle_position = (int(centroid[0] + offset_x), int(centroid[1] + offset_y))

            nozzle_detected = True  # ノズル位置が検出されたことを記録

            # ノズル位置を記録
            nozzle_positions.append(nozzle_position)

            # ノズルの移動量を計算
            if len(nozzle_positions) >= 2:
                nozzle_movement = calculate_distance(np.array(nozzle_positions[-1]), np.array(nozzle_positions[-2]))
            else:
                nozzle_movement = 0

            # ノズルの位置を描画
            cv2.circle(output_frame, nozzle_position, 5, (255, 0, 0), -1)

            # 緑色マスクから輪郭を検出
            contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 面積のしきい値を適用
            valid_contours_green = [cnt for cnt in contours_green if cv2.contourArea(cnt) >= green_area_threshold]

            if valid_contours_green:
                printed_matter = max(valid_contours_green, key=cv2.contourArea)
                cv2.drawContours(output_frame, [printed_matter], -1, (0, 255, 255), 2)

                # 最も上にあるポイントを取得
                y_range = 10
                topmost = tuple(printed_matter[printed_matter[:,:,1].argmin()][0])
                print_surface = [pt for pt in printed_matter if topmost[1] - y_range <= pt[0][1] <= topmost[1] + y_range]

                if len(print_surface) > 1:
                    # 線形回帰による印刷平面の推定
                    x_points = np.array([pt[0][0] for pt in print_surface])
                    y_points = np.array([pt[0][1] for pt in print_surface])
                    x = x_points.reshape(-1,1)
                    y = y_points
                    model = LinearRegression()
                    model.fit(x, y)
                    m = model.coef_[0]
                    c = model.intercept_

                    # 回帰直線の描画
                    x_start = min(x_points)
                    x_end = max(x_points)
                    y_start = int(model.predict([[x_start]])[0])
                    y_end = int(model.predict([[x_end]])[0])
                    cv2.line(output_frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

                    plane_detected = True  # 印刷平面が検出されたことを記録

                    # 印刷平面の位置を記録
                    plane_position = ((x_start + x_end) / 2, (y_start + y_end) / 2)
                    plane_positions.append(plane_position)

                    # 印刷平面の移動量を計算
                    if len(plane_positions) >= 2:
                        plane_movement = calculate_distance(np.array(plane_positions[-1]), np.array(plane_positions[-2]))
                    else:
                        plane_movement = 0

                    # ノズルと印刷平面の距離を計算
                    nozzle_plane_distance = calculate_distance(np.array(nozzle_position), np.array(plane_position))
                    # 距離を記録
                    nozzle_plane_distances.append(nozzle_plane_distance)

                    # 距離の変化量を計算
                    if len(nozzle_plane_distances) >= 2:
                        distance_change = abs(nozzle_plane_distances[-1] - nozzle_plane_distances[-2])
                    else:
                        distance_change = 0

                    # ノズル位置と印刷平面のX,Y最小・最大値
                    x_min_line = x_start
                    x_max_line = x_end
                    y_min_line = min(y_start, y_end)
                    y_max_line = max(y_start, y_end)

                    # 条件の判定
                    x0, y0 = nozzle_position

                    condition1 = x0 <= x_max_line + threshold_x_max
                    condition2 = x0 >= x_min_line - threshold_x_min
                    condition3 = y0 <= y_max_line + threshold_y_max
                    condition4 = y0 >= y_min_line - threshold_y_min
                    condition5 = plane_movement <= plane_movement_threshold

                    # 新しい条件の判定
                    condition6 = nozzle_movement >= nozzle_movement_threshold
                    condition7 = distance_change <= distance_change_threshold

                    if condition1 and condition2 and condition3 and condition4:
                        if condition6 and condition7:
                            message = "樹脂付着"
                            text_color = (0, 0, 255)  # 赤色
                            is_failure = True
                        elif condition5:
                            message = "印刷は正常です"
                            text_color = (0, 255, 0)  # 緑色
                            is_failure = False
                        else:
                            message = "樹脂付着"
                            text_color = (0, 0, 255)  # 赤色
                            is_failure = True
                    else:
                        message = "ノズル位置がしきい値を超えています"
                        text_color = (0, 0, 255)  # 赤色
                        is_failure = True

                    # デバッグ情報を描画
                    # cv2.putText(output_frame, f"x0: {x0}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    # cv2.putText(output_frame, f"x_min_line - ThXMin: {x_min_line - threshold_x_min}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    # cv2.putText(output_frame, f"x_max_line + ThXMax: {x_max_line + threshold_x_max}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    # cv2.putText(output_frame, f"y0: {y0}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    # cv2.putText(output_frame, f"y_min_line - ThYMin: {y_min_line - threshold_y_min}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    # cv2.putText(output_frame, f"y_max_line + ThYMax: {y_max_line + threshold_y_max}", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    # cv2.putText(output_frame, f"Nozzle Movement: {nozzle_movement:.2f}", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    # cv2.putText(output_frame, f"Distance Change: {distance_change:.2f}", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                else:
                    message = "データ不足のため判定不可"
                    text_color = (0, 0, 255)
                    is_failure = True
            else:
                message = "印刷物が検出されません"
                text_color = (0, 0, 255)
                is_failure = True
        else:
            message = "ノズルが検出されません"
            text_color = (0, 0, 255)
            is_failure = True
    else:
        message = "輪郭が検出されません"
        text_color = (0, 0, 255)
        is_failure = True

    # ノズルが検出された場合のみカウンターを更新
    if nozzle_detected and plane_detected:
        nozzle_detected_in_interval = True
        if is_failure:
            failure_frames += 1
        else:
            success_frames += 1

    # 単位時間が経過したかをチェック
    current_time = time.time()
    if current_time - start_time >= unit_time:
        if nozzle_detected_in_interval:
            total_detected_frames = success_frames + failure_frames
            if total_detected_frames > 0:
                failure_percentage = (failure_frames / total_detected_frames) * 100
                if failure_percentage >= 70:
                    percentage_text_color = (255, 0, 0)  # 赤色
                else:
                    percentage_text_color = (0, 255, 0)  # 緑色
                percentage_message = f"失敗検知率: {failure_percentage:.1f}%"
                print(f"単位時間内の失敗率: {failure_percentage:.1f}%")
            else:
                percentage_message = "不明"
                percentage_text_color = (255, 255, 255)
                print("単位時間内の失敗率: 不明")
        else:
            percentage_message = "不明"
            percentage_text_color = (255, 255, 255)
            print("単位時間内の失敗率: 不明")

        # カウンターのリセット
        start_time = current_time
        success_frames = 0
        failure_frames = 0
        nozzle_detected_in_interval = False
        plane_positions = []           # 印刷平面の位置リストをリセット
        nozzle_positions = []          # ノズル位置のリストをリセット
        nozzle_plane_distances = []    # ノズルと印刷平面の距離リストをリセット

    # OpenCVの画像をPillowの画像に変換
    output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(output_frame_rgb)

    # Pillowでテキストを描画
    draw = ImageDraw.Draw(pil_im)
    draw.text((50, 30), message, font=font, fill=text_color)
    draw.text((50, 60), percentage_message, font=font, fill=percentage_text_color)

    # Pillowの画像をOpenCVの画像に戻す
    output_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    # 結果を表示
    cv2.imshow('Result', output_frame)
    cv2.imshow('Red Mask', red_mask)
    cv2.imshow('Green Mask', green_mask)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
