import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# グローバル変数
selected_color_1 = None
selected_color_2 = None
hsv_color_1 = None
hsv_color_2 = None
positions_1 = []  # shorimae1の過去の位置
positions_2 = []  # shorimae2の過去の位置
tracking_status_1 = "COLOR PICK"
tracking_status_2 = "COLOR PICK"
trajectory = []  # 軌跡を保存するためのリスト
start_frame = 0  # 再生開始位置

# カメラキャプチャの設定
cap1 = cv2.VideoCapture('shorimae1.webm')
cap2 = cv2.VideoCapture('shorimae2.webm')

cv2.namedWindow("Video1")
cv2.namedWindow("Video2")

# トラックバーのコールバック関数
def on_trackbar(val):
    global start_frame
    start_frame = val
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# トラックバーの追加
total_frames_1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.createTrackbar('Start Frame', 'Video1', 0, total_frames_1 - 1, on_trackbar)

# マウスクリックイベントの処理
def select_color(event, x, y, flags, param):
    global selected_color_1, hsv_color_1, selected_color_2, hsv_color_2, tracking_status_1, tracking_status_2
    if param == 'video1' and event == cv2.EVENT_LBUTTONDOWN:
        selected_color_1 = frame1[y, x]
        hsv_color_1 = cv2.cvtColor(np.uint8([[selected_color_1]]), cv2.COLOR_BGR2HSV)[0][0]
        tracking_status_1 = "TRACKING"
    elif param == 'video2' and event == cv2.EVENT_LBUTTONDOWN:
        selected_color_2 = frame2[y, x]
        hsv_color_2 = cv2.cvtColor(np.uint8([[selected_color_2]]), cv2.COLOR_BGR2HSV)[0][0]
        tracking_status_2 = "TRACKING"

cv2.setMouseCallback("Video1", select_color, param='video1')
cv2.setMouseCallback("Video2", select_color, param='video2')

# グラフの設定
plt.ion()  # インタラクティブモードをオンにする
fig, ax = plt.subplots()
point, = ax.plot([], [], 'ro')
ax.set_xlim(-500, 500)  # X軸の表示範囲
ax.set_ylim(-500, 500)  # Y軸の表示範囲

def update_graph(x, y):
    global trajectory

    # 現在の時刻と位置を追加
    trajectory.append((x, y, time.time()))

    # 古い軌跡を削除（過去5秒間のもののみ保持）
    current_time = time.time()
    trajectory = [(px, py, pt) for (px, py, pt) in trajectory if current_time - pt <= 5]

    # 軌跡を描画
    ax.clear()
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    for px, py, pt in trajectory:
        alpha = max(0, 1 - (current_time - pt) / 5)  # 時間に応じて色を薄くする
        ax.plot(px, py, 'o', color=(1, 1, 0, alpha))  # 黄色で表示、alphaで透明度を設定

    # 現在の位置を赤で表示
    point.set_data(x, y)
    ax.plot(x, y, 'ro')

    fig.canvas.draw()
    fig.canvas.flush_events()

# 動画をトラックバーで指定されたフレームから再生するために位置を設定
cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while True:
    # フレームをキャプチャ
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # shorimae1.webmのトラッキング
    if hsv_color_1 is not None:
        hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hue_1, saturation_1, value_1 = int(hsv_color_1[0]), int(hsv_color_1[1]), int(hsv_color_1[2])
        lower_bound_1 = np.array([np.clip(hue_1 - 5, 0, 179), np.clip(saturation_1 - 50, 50, 255), np.clip(value_1 - 50, 50, 255)], dtype=np.uint8)
        upper_bound_1 = np.array([np.clip(hue_1 + 5, 0, 179), np.clip(saturation_1 + 50, 50, 255), np.clip(value_1 + 50, 50, 255)], dtype=np.uint8)
        mask1 = cv2.inRange(hsv_frame1, lower_bound_1, upper_bound_1)
        contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours1:
            largest_contour_1 = max(contours1, key=cv2.contourArea)
            ((x1, y1), radius1) = cv2.minEnclosingCircle(largest_contour_1)
            positions_1.append((x1, y1))
            cv2.circle(frame1, (int(x1), int(y1)), int(radius1), (0, 255, 0), 2)

    # shorimae2.webmのトラッキング
    if hsv_color_2 is not None:
        hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        hue_2, saturation_2, value_2 = int(hsv_color_2[0]), int(hsv_color_2[1]), int(hsv_color_2[2])
        lower_bound_2 = np.array([np.clip(hue_2 - 5, 0, 179), np.clip(saturation_2 - 50, 50, 255), np.clip(value_2 - 50, 50, 255)], dtype=np.uint8)
        upper_bound_2 = np.array([np.clip(hue_2 + 5, 0, 179), np.clip(saturation_2 + 50, 50, 255), np.clip(value_2 + 50, 50, 255)], dtype=np.uint8)
        mask2 = cv2.inRange(hsv_frame2, lower_bound_2, upper_bound_2)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours2:
            largest_contour_2 = max(contours2, key=cv2.contourArea)
            ((x2, y2), radius2) = cv2.minEnclosingCircle(largest_contour_2)
            positions_2.append((x2, y2))
            cv2.circle(frame2, (int(x2), int(y2)), int(radius2), (0, 255, 0), 2)

    # グラフの更新
    if positions_1 and positions_2:
        x = positions_2[-1][0]  # shorimae2のx位置
        y = positions_1[-1][0]  # shorimae1のx位置をY軸の値として使用
        update_graph(x, y)

    # 映像を表示
    cv2.imshow("Video1", frame1)
    cv2.imshow("Video2", frame2)

    # 'Esc'キーでプログラムを終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

# カメラリソースを解放
cap1.release()
cap2.release()
cv2.destroyAllWindows()
