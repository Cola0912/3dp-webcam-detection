import cv2
import numpy as np
import time

# グローバル変数
selected_color = None
hsv_color = None
fps = 0
positions = []  # 過去の位置を保存
radii = []  # 半径の履歴
initial_radius = None  # ボールの最初の半径
tracking_status = "COLOR PICK"  # 初期状態はCOLOR PICK

# 初期値
trajectory_length = 15  # 軌跡の長さ
time_threshold = 0.2  # 直近0.2秒のデータを使用
smooth_threshold = 50  # 急激な位置変化を無視するための閾値
arrow_scale_factor = 2  # 速さに応じた矢印の長さの比率
arrow_min_scale_factor = 1.2  # 矢印の最小長さをボールの直径の1.2倍に設定
ellipse_flatness = 0.7  # 楕円の進行方向への扁平具合（0〜1、1で完全な円）

# 色検出の厳しさを設定する変数
color_tolerance_hue = 5  # 色相の許容範囲
color_tolerance_saturation = 50  # 彩度の許容範囲
color_tolerance_value = 50  # 明度の許容範囲

# マウスクリックイベントの処理
def select_color(event, x, y, flags, param):
    global selected_color, hsv_color, tracking_status
    if event == cv2.EVENT_LBUTTONDOWN:
        # クリックしたピクセルの色を取得（BGR）
        selected_color = frame[y, x]
        # BGRからHSVに変換
        hsv_color = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]
        tracking_status = "TRACKING"  # 色が選択されたらTRACKINGに変更

# カメラキャプチャの設定
cap = cv2.VideoCapture('shorimae.webm')

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", select_color)

# FPS計測のためのタイマー
prev_time = time.time()

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    if not ret:
        break

    # 現在の時間と前のフレームの時間からFPSを計算
    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff > 0.001:  # 時間差が極端に小さくない場合だけFPSを計算
        fps = 1 / time_diff
    prev_time = current_time

    # 色が選択されていない場合はCOLOR PICKを表示
    if hsv_color is None:
        tracking_status = "COLOR PICK"
    else:
        # BGR色空間からHSV色空間に変換
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 色相、彩度、明度の値をintにキャスト
        hue = int(hsv_color[0])
        saturation = int(hsv_color[1])
        value = int(hsv_color[2])

        # 許容範囲に基づいて色検出の範囲を設定
        color_lower = np.array([
            np.clip(hue - color_tolerance_hue, 0, 179), 
            np.clip(saturation - color_tolerance_saturation, 50, 255), 
            np.clip(value - color_tolerance_value, 50, 255)
        ], dtype=np.uint8)
        
        color_upper = np.array([
            np.clip(hue + color_tolerance_hue, 0, 179), 
            np.clip(saturation + color_tolerance_saturation, 50, 255), 
            np.clip(value + color_tolerance_value, 50, 255)
        ], dtype=np.uint8)

        # 選択した色に基づくマスクを作成
        mask = cv2.inRange(hsv_frame, color_lower, color_upper)

        # 輪郭を抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # ボールは1つだけとして最大の輪郭を選択
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            # ボールの最初の半径を保存（最初の検出時）
            if initial_radius is None:
                initial_radius = radius

            # ボールの半径が開始時の25%以下なら無視
            if radius < initial_radius * 0.25:
                if tracking_status != "LOST":
                    tracking_status = "LOST"
                    positions.clear()
                    radii.clear()
                # ボールが小さすぎる場合、残りの処理をスキップ
            else:
                # 直近の半径リストを更新
                radii.append(radius)
                # リストのサイズを制限
                if len(radii) > trajectory_length:
                    radii.pop(0)

                # ボールを追跡できているので状態をTRACKINGに
                tracking_status = "TRACKING"
                
                # 前の位置から急激に離れている場合は無視
                if len(positions) == 0 or np.sqrt((x - positions[-1][0]) ** 2 + (y - positions[-1][1]) ** 2) < smooth_threshold:
                    positions.append((x, y, current_time))
                    # リストのサイズを制限
                    if len(positions) > trajectory_length:
                        positions.pop(0)
                else:
                    # 急激な位置変化があった場合、位置リストをクリア
                    positions.clear()
                    positions.append((x, y, current_time))

                # 時間しきい値を超えたデータを削除
                positions = [(px, py, pt) for (px, py, pt) in positions if current_time - pt <= time_threshold]

                if len(radii) > 0:
                    average_radius = np.mean(radii)
                    if abs(radius - average_radius) / average_radius > 0.5:
                        if tracking_status != "LOST":
                            tracking_status = "LOST"
                            positions.clear()
                            radii.clear()
                        # 半径の変化が大きい場合、残りの処理をスキップ

                if len(positions) > 1:
                    # 直近0.2秒のデータで平均座標を計算
                    avg_x = int(np.mean([p[0] for p in positions]))
                    avg_y = int(np.mean([p[1] for p in positions]))

                    # 移動方向を計算
                    dx = positions[-1][0] - positions[0][0]
                    dy = positions[-1][1] - positions[0][1]
                    speed = np.sqrt(dx**2 + dy**2)

                    # ボールの直径
                    ball_diameter = 2 * radius

                    # 矢印の長さ（速さに応じた長さとボールの直径1.2倍の最小値）
                    arrow_length = max(int(speed * arrow_scale_factor), int(ball_diameter * arrow_min_scale_factor))
                    # 進行方向の正規化
                    if speed != 0:
                        dir_x = dx / speed
                        dir_y = dy / speed
                    else:
                        dir_x, dir_y = 0, 0

                    end_point = (int(avg_x + dir_x * arrow_length), int(avg_y + dir_y * arrow_length))
                    cv2.arrowedLine(frame, (avg_x, avg_y), end_point, (0, 0, 255), 2, tipLength=0.5)

                    # 矢印の長さを基に、進行方向に扁平な楕円を描画
                    ellipse_major_axis = arrow_length  # 矢印の長さを主軸に
                    ellipse_minor_axis = int(arrow_length * ellipse_flatness)  # 楕円の扁平具合を指定
                    angle = np.degrees(np.arctan2(dy, dx)) if speed != 0 else 0  # 楕円の回転角度

                    cv2.ellipse(frame, (avg_x, avg_y), (ellipse_major_axis, ellipse_minor_axis), 
                                angle, 0, 360, (255, 0, 0), 2)
                    
                    # **軌跡の描画を追加**
                    # positionsの過去の位置を線で結ぶ
                    for i in range(1, len(positions)):
                        pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
                        pt2 = (int(positions[i][0]), int(positions[i][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                
                # 座標を黒で表示
                cv2.putText(frame, f"X: {int(x)} Y: {int(y)}", (int(x) - 50, int(y) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                # ボールを円で囲む
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        else:
            # ボールが検出されなかった場合は状態をLOSTにし、無駄な処理を避ける
            if tracking_status != "LOST":
                tracking_status = "LOST"
                positions.clear()
                radii.clear()
            # 'LOST' と状態を表示
            cv2.putText(frame, "LOST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # FPSと状態を表示
    cv2.putText(frame, f"FPS: {int(fps)} | {tracking_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 映像を表示
    cv2.imshow("Video", frame)

    # 'Esc'キーでプログラムを終了
    if cv2.waitKey(1) & 0xFF == 27:  # 27はエスケープキーのASCIIコード
        break

# カメラリソースを解放
cap.release()
cv2.destroyAllWindows()
