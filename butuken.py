import cv2
import numpy as np
import time

# 動画ファイルから映像を取得
cap = cv2.VideoCapture('shorimae.webm')

# 総フレーム数を取得
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# グローバル変数
track_window = None
roi_hist = None
term_crit = None
tracking = False
lost = False  # 物体を見失ったかどうかのフラグ
exit_edge = None  # 画面外に出た際のエッジ
roi_defined = False  # 追跡領域が定義されたかどうかのフラグ
trajectory_points = []  # 物体の軌跡を保存するリスト (位置, タイムスタンプ) のタプル
smoothed_center = None  # スムーズ化された中心座標
smoothing_window = 5  # スムーズ化のためのウィンドウサイズ（フレーム数）
prev_center = None  # 前回の中心座標
frame_number = 0  # 現在のフレーム番号

# コールバック関数（何もしない）
def nothing(x):
    pass

# HSVのしきい値トラックバーを作成
cv2.namedWindow('Settings')
cv2.createTrackbar('LH', 'Settings', 0, 179, nothing)
cv2.createTrackbar('LS', 'Settings', 0, 255, nothing)
cv2.createTrackbar('LV', 'Settings', 0, 255, nothing)
cv2.createTrackbar('UH', 'Settings', 0, 179, nothing)
cv2.createTrackbar('US', 'Settings', 0, 255, nothing)
cv2.createTrackbar('UV', 'Settings', 0, 255, nothing)

# 再生位置を制御するトラックバーを作成
cv2.namedWindow('3D Printer Tracking')
cv2.createTrackbar('Position', '3D Printer Tracking', 0, total_frames - 1, nothing)

# 薄い緑色の初期値を設定
initial_LH = 35
initial_LS = 50
initial_LV = 50
initial_UH = 85
initial_US = 255
initial_UV = 255

cv2.setTrackbarPos('LH', 'Settings', initial_LH)
cv2.setTrackbarPos('LS', 'Settings', initial_LS)
cv2.setTrackbarPos('LV', 'Settings', initial_LV)
cv2.setTrackbarPos('UH', 'Settings', initial_UH)
cv2.setTrackbarPos('US', 'Settings', initial_US)
cv2.setTrackbarPos('UV', 'Settings', initial_UV)

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

    # ブラーを適用してノイズを低減
    frame_blurred = cv2.medianBlur(frame, 5)

    # HSV色空間に変換
    hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

    # Vチャンネルに対してCLAHEを適用
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge((h, s, v_eq))

    # トラックバーからHSVのしきい値を取得
    l_h = cv2.getTrackbarPos('LH', 'Settings')
    l_s = cv2.getTrackbarPos('LS', 'Settings')
    l_v = cv2.getTrackbarPos('LV', 'Settings')
    u_h = cv2.getTrackbarPos('UH', 'Settings')
    u_s = cv2.getTrackbarPos('US', 'Settings')
    u_v = cv2.getTrackbarPos('UV', 'Settings')

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # マスクの作成
    mask = cv2.inRange(hsv_eq, lower_bound, upper_bound)

    # 物体の検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 面積が最大の輪郭を取得
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 500:  # 面積の閾値のみを使用
            # バウンディングボックスを取得
            x, y, w, h = cv2.boundingRect(max_contour)
            track_window = (x, y, w, h)

            # 物体の中心を計算
            M = cv2.moments(max_contour)
            if M['m00'] != 0:
                center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            else:
                center = (x + w//2, y + h//2)

            # 物体の中心座標を軌跡として保存（タイムスタンプ付き）
            current_time = time.time()
            trajectory_points.append((tuple(center), current_time))

            # 古いポイントを削除（1秒以上前のもの）
            trajectory_points = [(pos, t) for pos, t in trajectory_points if current_time - t <= 1]

            # 過去の位置の平均を取ってスムーズ化
            smoothed_center = smooth_positions(trajectory_points, smoothing_window)

            # バウンディングボックスを描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # スムーズ化された中心座標を表示
            cv2.circle(frame, tuple(smoothed_center), 5, (255, 0, 0), -1)

            # 過去の軌跡を描画（1秒以内のもの）
            points = [pos for pos, t in trajectory_points]
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)

            # 中心座標を表示
            cv2.putText(frame, f"Center: {smoothed_center}", (10, 30),
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

            # 物体がフレーム外に出た場合
            if (smoothed_center[0] < 0 or smoothed_center[0] >= frame.shape[1] or
                smoothed_center[1] < 0 or smoothed_center[1] >= frame.shape[0]):
                lost = True
                tracking = False  # 追跡を停止
                exit_edge = get_nearest_edge(smoothed_center, frame.shape)
            else:
                exit_edge = None  # フレーム内に戻った場合、exit_edge をリセット
        else:
            # 面積が小さい場合は見失ったとみなす
            lost = True
            tracking = False
            if smoothed_center is not None:
                exit_edge = get_nearest_edge(smoothed_center, frame.shape)
    else:
        # 物体が検出されない場合
        lost = True
        tracking = False
        if smoothed_center is not None:
            exit_edge = get_nearest_edge(smoothed_center, frame.shape)

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

# 終了条件（追跡の精度）
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 最初のフレームを読み込む
ret, frame = cap.read()
if not ret:
    print("動画ファイルからフレームを取得できませんでした。")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while True:
    # 現在の再生位置を取得
    current_pos = cv2.getTrackbarPos('Position', '3D Printer Tracking')

    # フレーム番号が変わった場合、新しい位置にシーク
    if frame_number != current_pos:
        frame_number = current_pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
    else:
        # 次のフレームを取得
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        # トラックバーの位置を更新
        cv2.setTrackbarPos('Position', '3D Printer Tracking', frame_number)

    # フレームをリサイズ（必要に応じて調整）
    frame_resized = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

    # 追跡開始
    frame_processed = process_frame(frame_resized)

    # フレーム表示
    cv2.imshow('3D Printer Tracking', frame_processed)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
