import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ビデオファイルから映像を取得
cap = cv2.VideoCapture('shorimae.webm')  # shorimae.webm を使用

# 動画の総フレーム数を取得
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# グローバル変数
track_window1 = None
track_window2 = None
roi_hist1 = None
roi_hist2 = None
term_crit = None
tracking1 = False
tracking2 = False
roi_defined1 = False
roi_defined2 = False
prev_center1 = None
prev_center2 = None
hsv_at_mouse = None  # マウス位置のHSV色
latest_x_position1 = None  # 赤い三角形の最新のX座標
latest_x_position2 = None  # 追加の物体（緑色）の最新のX座標

# グラフの初期設定
fig, ax = plt.subplots()
ax.set_xlim(0, cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ax.set_ylim(-cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
graph_data1, = ax.plot([], [], 'ro')  # 赤い三角形のプロット
graph_data2, = ax.plot([], [], 'bo')  # 緑色物体のプロット

# 追跡ウィンドウの終了条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 指定された色範囲の物体を検出する関数
def detect_colored_object(hsv, lower_bound, upper_bound):
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) >= 3:  # 三角形または多角形と認識
            candidate = approx
            break

    return candidate

# 緑色物体の上側の輪郭の直線を認識する関数
def detect_upper_line(contour):
    if contour is not None:
        # 輪郭の点群からY座標でソートして上の部分を取得
        top_points = np.array([p[0] for p in contour])

        # 上側の直線をフィッティング
        if len(top_points) >= 2:
            [vx, vy, x, y] = cv2.fitLine(top_points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = vx.item(), vy.item(), x.item(), y.item()  # 配列からスカラーに変換
            lefty = int((-x * vy / vx) + y)
            righty = int(((cap.get(cv2.CAP_PROP_FRAME_WIDTH) - x) * vy / vx) + y)
            return (0, lefty), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), righty)

    return None

# フレームの処理を行う関数
def process_frame(frame):
    global track_window1, track_window2, roi_hist1, roi_hist2, term_crit, tracking1, tracking2
    global roi_defined1, roi_defined2, prev_center1, prev_center2, latest_x_position1, latest_x_position2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # トラックバーから色範囲を取得（赤い三角形用）
    lower_bound1 = np.array([cv2.getTrackbarPos('Hue Min', 'Red Triangle Tracking'),
                             cv2.getTrackbarPos('Sat Min', 'Red Triangle Tracking'),
                             cv2.getTrackbarPos('Val Min', 'Red Triangle Tracking')])
    upper_bound1 = np.array([cv2.getTrackbarPos('Hue Max', 'Red Triangle Tracking'),
                             cv2.getTrackbarPos('Sat Max', 'Red Triangle Tracking'),
                             cv2.getTrackbarPos('Val Max', 'Red Triangle Tracking')])
    
    # トラックバーから色範囲を取得（緑色物体用）
    lower_bound2 = np.array([cv2.getTrackbarPos('Hue Min (Green)', 'Green Object Tracking'),
                             cv2.getTrackbarPos('Sat Min (Green)', 'Green Object Tracking'),
                             cv2.getTrackbarPos('Val Min (Green)', 'Green Object Tracking')])
    upper_bound2 = np.array([cv2.getTrackbarPos('Hue Max (Green)', 'Green Object Tracking'),
                             cv2.getTrackbarPos('Sat Max (Green)', 'Green Object Tracking'),
                             cv2.getTrackbarPos('Val Max (Green)', 'Green Object Tracking')])

    # 赤い三角形の追跡
    if not tracking1:
        triangle_contour = detect_colored_object(hsv, lower_bound1, upper_bound1)
        if triangle_contour is not None:
            x, y, w, h = cv2.boundingRect(triangle_contour)
            track_window1 = (x, y, w, h)
            roi1 = hsv[y:y+h, x:x+w]
            mask_roi1 = cv2.inRange(roi1, lower_bound1, upper_bound1)
            roi_hist1 = cv2.calcHist([roi1], [0], mask_roi1, [180], [0, 180])
            cv2.normalize(roi_hist1, roi_hist1, 0, 255, cv2.NORM_MINMAX)
            tracking1 = True
            roi_defined1 = True

    if tracking1 and roi_hist1 is not None and roi_defined1:
        dst1 = cv2.calcBackProject([hsv], [0], roi_hist1, [0, 180], 1)
        ret1, track_window1 = cv2.CamShift(dst1, track_window1, term_crit)
        pts1 = cv2.boxPoints(ret1)
        pts1 = np.int32(pts1)
        center1 = np.mean(pts1, axis=0).astype(int)
        latest_x_position1 = center1[0]
        cv2.circle(frame, tuple(center1), 5, (0, 0, 255), -1)

    # 緑色物体の追跡
    green_contour = None  # 初期化しておくことで、追跡がない場合でも参照エラーを防止
    if not tracking2:
        green_contour = detect_colored_object(hsv, lower_bound2, upper_bound2)
        if green_contour is not None:
            x, y, w, h = cv2.boundingRect(green_contour)
            track_window2 = (x, y, w, h)
            roi2 = hsv[y:y+h, x:x+w]
            mask_roi2 = cv2.inRange(roi2, lower_bound2, upper_bound2)
            roi_hist2 = cv2.calcHist([roi2], [0], mask_roi2, [180], [0, 180])
            cv2.normalize(roi_hist2, roi_hist2, 0, 255, cv2.NORM_MINMAX)
            tracking2 = True
            roi_defined2 = True

    if tracking2 and roi_hist2 is not None and roi_defined2:
        dst2 = cv2.calcBackProject([hsv], [0], roi_hist2, [0, 180], 1)
        ret2, track_window2 = cv2.CamShift(dst2, track_window2, term_crit)
        pts2 = cv2.boxPoints(ret2)
        pts2 = np.int32(pts2)
        center2 = np.mean(pts2, axis=0).astype(int)
        latest_x_position2 = center2[0]
        cv2.circle(frame, tuple(center2), 5, (255, 0, 0), -1)

        # 緑色物体の上側の輪郭の直線を描画
        if green_contour is not None:
            line_points = detect_upper_line(green_contour)
            if line_points is not None:
                cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)

    return frame

# グラフを更新する関数
def update_graph(frame_num):
    ax.clear()
    ax.set_xlim(0, cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ax.set_ylim(-cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    if latest_x_position1 is not None:
        ax.plot(latest_x_position1, 0, 'ro')  # 赤い三角形の最新の位置
    if latest_x_position2 is not None:
        ax.plot(latest_x_position2, 0, 'bo')  # 緑色物体の最新の位置
    return graph_data1, graph_data2

# マウスイベントコールバック関数
def on_mouse(event, x, y, flags, param):
    global hsv_at_mouse, frame
    if event == cv2.EVENT_MOUSEMOVE and frame is not None:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_at_mouse = hsv_frame[y, x]

# トラックバーのコールバック関数
def on_trackbar(val):
    global current_frame
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# ウィンドウとトラックバーの設定
cv2.namedWindow('Red Triangle Tracking')
cv2.createTrackbar('Position', 'Red Triangle Tracking', 0, total_frames - 1, on_trackbar)

# 赤い三角形の色範囲を設定するためのトラックバーの追加
cv2.createTrackbar('Hue Min', 'Red Triangle Tracking', 170, 179, lambda x: None)
cv2.createTrackbar('Hue Max', 'Red Triangle Tracking', 179, 179, lambda x: None)
cv2.createTrackbar('Sat Min', 'Red Triangle Tracking', 178, 255, lambda x: None)
cv2.createTrackbar('Sat Max', 'Red Triangle Tracking', 255, 255, lambda x: None)
cv2.createTrackbar('Val Min', 'Red Triangle Tracking', 103, 255, lambda x: None)
cv2.createTrackbar('Val Max', 'Red Triangle Tracking', 188, 255, lambda x: None)

# 緑色物体の色範囲を設定するためのトラックバーの追加
cv2.namedWindow('Green Object Tracking')
cv2.createTrackbar('Hue Min (Green)', 'Green Object Tracking', 60, 179, lambda x: None)
cv2.createTrackbar('Hue Max (Green)', 'Green Object Tracking', 90, 179, lambda x: None)
cv2.createTrackbar('Sat Min (Green)', 'Green Object Tracking', 100, 255, lambda x: None)
cv2.createTrackbar('Sat Max (Green)', 'Green Object Tracking', 255, 255, lambda x: None)
cv2.createTrackbar('Val Min (Green)', 'Green Object Tracking', 100, 255, lambda x: None)
cv2.createTrackbar('Val Max (Green)', 'Green Object Tracking', 255, 255, lambda x: None)

# 移動距離のしきい値を設定するトラックバーの追加（初期値50、最大500）
cv2.createTrackbar('Move Threshold', 'Red Triangle Tracking', 50, 500, lambda x: None)

# マウスコールバックの設定
cv2.setMouseCallback('Red Triangle Tracking', on_mouse)

# グラフのアニメーション設定
ani = FuncAnimation(fig, update_graph, blit=False, interval=100)

# ビデオフレームの処理ループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # トラックバーの位置を取得
    current_frame = cv2.getTrackbarPos('Position', 'Red Triangle Tracking')

    # フレームの処理
    frame = process_frame(frame)

    # マウス位置のHSV値を表示
    if hsv_at_mouse is not None:
        hsv_text = f"HSV: {hsv_at_mouse}"
        cv2.putText(frame, hsv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # フレームを1倍にリサイズ（元のサイズを維持）
    frame_resized = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

    # フレームを表示
    cv2.imshow('Red Triangle Tracking', frame_resized)

    # グラフを更新
    plt.pause(0.001)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
