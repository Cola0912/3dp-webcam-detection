import cv2
import numpy as np
import time  # 時間を扱うために追加

# ウェブカメラから映像を取得
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラデバイス

# グローバル変数
track_window = None
roi_hist = None
term_crit = None
tracking = False
lost = False  # ボールを見失ったかどうかのフラグ
exit_edge = None  # 画面外に出た際のエッジ
roi_defined = False  # 追跡領域が定義されたかどうかのフラグ
trajectory_points = []  # ボールの軌跡を保存するリスト (位置, タイムスタンプ) のタプル
smoothed_center = None  # スムーズ化された中心座標
smoothing_window = 5  # スムーズ化のためのウィンドウサイズ（フレーム数）
prev_center = None  # 前回の中心座標

# 固定された追跡対象の色（選択された色：[90, 79, 206]）
selected_color = [90, 79, 206]  # 追跡対象のBGR色
color_selected = True  # 色選択は既に済んでいると仮定

# 移動平均でスムーズな座標を計算
def smooth_positions(positions, window_size):
    window_size = min(window_size, len(positions))
    # 位置のみを抽出して平均化
    avg_position = np.mean([pos for pos, _ in positions[-window_size:]], axis=0)
    return avg_position.astype(int)

# 最も近いフレームエッジを計算
def get_nearest_edge(point, frame_shape):
    x, y = point
    h, w = frame_shape[:2]
    distances = {
        'left': x,
        'right': w - x,
        'top': y,
        'bottom': h - y
    }
    exit_edge = min(distances, key=distances.get)
    return exit_edge

# 追跡処理
def process_frame(frame):
    global track_window, roi_hist, term_crit, tracking, lost, exit_edge, roi_defined
    global trajectory_points, smoothed_center, prev_center

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if tracking and roi_hist is not None and roi_defined:
        # CAMShiftで追跡
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # 追跡の領域描画
        pts = cv2.boxPoints(ret)
        pts = np.int32(pts)
        center = np.mean(pts, axis=0).astype(int)
        diameter = np.linalg.norm(pts[0] - pts[2])  # 対角線の長さを直径とする
        radius = int(diameter / 2)  # 円の半径を計算

        # 面積と円形度を計算
        area = cv2.contourArea(pts)
        perimeter = cv2.arcLength(pts, True)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # 一定の面積と円形度を持つ場合のみボールと判断
        if area > 500 and 0.7 < circularity <= 1.2:
            # ボールの中心座標を軌跡として保存（タイムスタンプ付き）
            current_time = time.time()
            trajectory_points.append((tuple(center), current_time))

            # 古いポイントを削除（1秒以上前のもの）
            trajectory_points = [(pos, t) for pos, t in trajectory_points if current_time - t <= 1]

            # 過去のボールの位置の平均を取ってスムーズ化
            smoothed_center = smooth_positions(trajectory_points, smoothing_window)

            # スムーズ化された位置でボールの輪郭を円で表示
            cv2.circle(frame, tuple(smoothed_center), radius, (0, 0, 255), 2)  # 赤い円を描画

            # スムーズ化されたボールの中心座標を表示
            cv2.circle(frame, tuple(smoothed_center), 5, (255, 0, 0), -1)

            # 過去のボールの軌跡を描画（1秒以内のもの）
            points = [pos for pos, t in trajectory_points]
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)

            # 中心座標、直径を表示
            cv2.putText(frame, f"Center: {smoothed_center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Diameter: {int(diameter)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 移動方向の矢印表示
            if prev_center is not None:
                direction_vector = np.array(smoothed_center) - np.array(prev_center)
                norm = np.linalg.norm(direction_vector)
                if norm > 0:
                    arrow_length = int(norm * 3)
                    arrow_tip = np.array(smoothed_center) + (arrow_length * direction_vector / norm)
                    arrow_tip = arrow_tip.astype(int)
                    cv2.arrowedLine(frame, tuple(smoothed_center), tuple(arrow_tip), (0, 255, 255), 3)

            # 現在の中心を保存
            prev_center = smoothed_center

            # ボールがフレーム外に出た場合
            if (smoothed_center[0] < 0 or smoothed_center[0] >= frame.shape[1] or
                smoothed_center[1] < 0 or smoothed_center[1] >= frame.shape[0]):
                lost = True
                tracking = False  # 追跡を停止
                exit_edge = get_nearest_edge(smoothed_center, frame.shape)
            else:
                exit_edge = None  # ボールがフレーム内に戻った場合、exit_edge をリセット
        else:
            # ボールではないと判断し、見失ったとみなす
            lost = True
            tracking = False  # 追跡を停止
            if smoothed_center is not None:
                exit_edge = get_nearest_edge(smoothed_center, frame.shape)
    elif lost:
        # ボールの再検出を試みる（条件を緩める）
        mask = cv2.inRange(hsv, lower_bound_re, upper_bound_re)
        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 面積が最大の輪郭を取得
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            # 面積の閾値を下げる
            if area > 300:
                # ボールを再検出
                x, y, w, h = cv2.boundingRect(max_contour)
                track_window = (x, y, w, h)
                roi = hsv[y:y+h, x:x+w]
                mask_roi = mask[y:y+h, x:x+w]
                roi_hist = cv2.calcHist([roi], [0], mask_roi, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                tracking = True
                lost = False
                exit_edge = None
                trajectory_points = []
                prev_center = None

    # フレーム端を強調表示
    if exit_edge:
        if exit_edge == 'left':
            cv2.line(frame, (0, 0), (0, frame.shape[0]), (0, 0, 255), 5)
        elif exit_edge == 'right':
            cv2.line(frame, (frame.shape[1]-1, 0), (frame.shape[1]-1, frame.shape[0]), (0, 0, 255), 5)
        elif exit_edge == 'top':
            cv2.line(frame, (0, 0), (frame.shape[1], 0), (0, 0, 255), 5)
        elif exit_edge == 'bottom':
            cv2.line(frame, (0, frame.shape[0]-1), (frame.shape[1], frame.shape[0]-1), (0, 0, 255), 5)

    return frame

# 色の指定：選択された色 [90, 79, 206]
hsv_color = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]
# 通常の色範囲
lower_bound = np.array([hsv_color[0] - 10, 50, 50])
upper_bound = np.array([hsv_color[0] + 10, 255, 255])
# 再検出用の色範囲（広げる）
lower_bound_re = np.array([hsv_color[0] - 15, 30, 30])
upper_bound_re = np.array([hsv_color[0] + 15, 255, 255])

# トラッキングウィンドウの設定
ret, frame = cap.read()
if not ret:
    print("カメラからフレームを取得できませんでした。")
    cap.release()
    cv2.destroyAllWindows()
    exit()

h, w, _ = frame.shape

# ROIを選択（フレーム全体を使用）
track_window = (0, 0, w, h)
roi = frame
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 終了条件（追跡の精度）
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 追跡が定義されたことを確認
roi_defined = True
tracking = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 追跡開始
    frame = process_frame(frame)

    # フレームを2倍にリサイズ
    frame_resized = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    # フレーム表示
    cv2.imshow('Basketball Tracking', frame_resized)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
