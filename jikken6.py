import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

# グローバル変数
# ビデオ1用（横からの映像）
selected_color_1a = None  # ノズルのマーカーの色
selected_color_1b = None  # 印刷物の色
hsv_color_1a = None
hsv_color_1b = None
hsv_range_1a = None
hsv_range_1b = None
click_count_1 = 0

# ビデオ2用（正面からの映像）
selected_color_2a = None
selected_color_2b = None
hsv_color_2a = None
hsv_color_2b = None
hsv_range_2a = None
hsv_range_2b = None
click_count_2 = 0

start_frame = 0  # 再生開始位置
threshold = 50   # ノズルと印刷物のしきい値

# カメラキャプチャの設定
cap1 = cv2.VideoCapture('shorimae1.webm')  # 横からの映像
cap2 = cv2.VideoCapture('shorimae2.webm')  # 正面からの映像

cv2.namedWindow("Video1")
cv2.namedWindow("Video2")

# フレームのサイズを取得
frame_width_1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height_1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width_2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height_2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# トラックバーのコールバック関数
def on_trackbar(val):
    global start_frame
    start_frame = val
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

def on_threshold_trackbar(val):
    global threshold
    threshold = val

# トラックバーの追加
total_frames_1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.createTrackbar('Start Frame', 'Video1', 0, total_frames_1 - 1, on_trackbar)
cv2.createTrackbar('Threshold', 'Video1', 0, 200, on_threshold_trackbar)
cv2.setTrackbarPos('Threshold', 'Video1', threshold)

# マウスクリックイベントの処理
def select_color(event, x, y, flags, param):
    global selected_color_1a, hsv_color_1a, hsv_range_1a, selected_color_1b, hsv_color_1b, hsv_range_1b
    global selected_color_2a, hsv_color_2a, hsv_range_2a, selected_color_2b, hsv_color_2b, hsv_range_2b
    global click_count_1, click_count_2

    if param == 'video1' and event == cv2.EVENT_LBUTTONDOWN:
        if click_count_1 == 0:
            selected_color_1a = frame1[y, x]
            hsv_color_1a = cv2.cvtColor(np.uint8([[selected_color_1a]]), cv2.COLOR_BGR2HSV)[0][0]
            hsv_range_1a = np.array([hsv_color_1a - [15, 80, 80], hsv_color_1a + [15, 80, 80]])
            click_count_1 += 1
            print("Video1: ノズルのマーカーの色を選択しました。")
        elif click_count_1 == 1:
            selected_color_1b = frame1[y, x]
            hsv_color_1b = cv2.cvtColor(np.uint8([[selected_color_1b]]), cv2.COLOR_BGR2HSV)[0][0]
            hsv_range_1b = np.array([hsv_color_1b - [15, 80, 80], hsv_color_1b + [15, 80, 80]])
            click_count_1 += 1
            print("Video1: 印刷物の色を選択しました。")

    elif param == 'video2' and event == cv2.EVENT_LBUTTONDOWN:
        if click_count_2 == 0:
            selected_color_2a = frame2[y, x]
            hsv_color_2a = cv2.cvtColor(np.uint8([[selected_color_2a]]), cv2.COLOR_BGR2HSV)[0][0]
            hsv_range_2a = np.array([hsv_color_2a - [15, 80, 80], hsv_color_2a + [15, 80, 80]])
            click_count_2 += 1
            print("Video2: ノズルのマーカーの色を選択しました。")
        elif click_count_2 == 1:
            selected_color_2b = frame2[y, x]
            hsv_color_2b = cv2.cvtColor(np.uint8([[selected_color_2b]]), cv2.COLOR_BGR2HSV)[0][0]
            hsv_range_2b = np.array([hsv_color_2b - [15, 80, 80], hsv_color_2b + [15, 80, 80]])
            click_count_2 += 1
            print("Video2: 印刷物の色を選択しました。")

cv2.setMouseCallback("Video1", select_color, param='video1')
cv2.setMouseCallback("Video2", select_color, param='video2')

# グラフの設定
plt.ion()  # インタラクティブモードをオンにする
fig, ax = plt.subplots(figsize=(6, 6))  # グラフを600x600ピクセルに設定
ax.set_xlim(0, frame_width_2)  # X軸はビデオ2の幅に合わせる
ax.set_ylim(0, frame_width_1)  # Y軸はビデオ1の幅に合わせる（ビデオ1のX座標を使用するため）
ax.invert_yaxis()    # Y軸を反転（必要に応じて）

def update_graph(nozzle_pos, print_obj_pos, print_obj_size, reg_line_side=None, reg_line_front=None):
    ax.clear()
    ax.set_xlim(0, frame_width_2)
    ax.set_ylim(0, frame_width_1)
    ax.invert_yaxis()

    # ノズルの位置をプロット
    if nozzle_pos is not None:
        ax.plot(nozzle_pos[0], nozzle_pos[1], 'o', color='red', label='Nozzle')

    # 印刷物の形状を矩形で描画
    if print_obj_pos is not None and print_obj_size is not None:
        rect = plt.Rectangle(
            (print_obj_pos[0] - print_obj_size[0]/2, print_obj_pos[1] - print_obj_size[0]/2),
            print_obj_size[0], print_obj_size[0],  # サイズは幅を使用（高さはビデオ1のX座標で表現）
            linewidth=1, edgecolor='green', facecolor='none', label='Printed Object'
        )
        ax.add_patch(rect)

    # 回帰直線を用いて四角形を描画
    if reg_line_side is not None and reg_line_front is not None:
        # ビデオ1（横からの映像）の回帰直線の端点
        (x1_side, y1_side), (x2_side, y2_side) = reg_line_side
        # ビデオ2（正面からの映像）の回帰直線の端点
        (x1_front, y1_front), (x2_front, y2_front) = reg_line_front

        # グラフの座標系に合わせて座標を取得
        # ビデオ1のX座標をグラフのY座標として使用
        y_coords = [x1_side, x2_side]
        y_min = min(y_coords)
        y_max = max(y_coords)

        # ビデオ2のX座標をグラフのX座標として使用
        x_coords = [x1_front, x2_front]
        x_min = min(x_coords)
        x_max = max(x_coords)

        # 四角形を描画
        rect_regression = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1, edgecolor='blue', facecolor='none', label='Regression Rect'
        )
        ax.add_patch(rect_regression)

    if nozzle_pos is not None or (print_obj_pos is not None and print_obj_size is not None) or (reg_line_side is not None and reg_line_front is not None):
        ax.legend()

    fig.canvas.draw()
    fig.canvas.flush_events()

# 平滑化のための関数（移動平均）
def smooth_hsv(prev_hsv, new_hsv, alpha=0.5):
    return alpha * prev_hsv + (1 - alpha) * new_hsv

# 動画をトラックバーで指定されたフレームから再生するために位置を設定
cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while True:
    # フレームをキャプチャ
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # ビデオ1の処理（横からの映像）
    # ノズルのマーカー
    nozzle_pos_side = None
    regression_line_endpoints_side = None  # 回帰直線の端点を初期化
    if hsv_range_1a is not None:
        hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        lower_bound_1a = np.clip(hsv_range_1a[0], [0, 30, 30], [179, 255, 255]).astype(np.uint8)
        upper_bound_1a = np.clip(hsv_range_1a[1], [0, 30, 30], [179, 255, 255]).astype(np.uint8)
        mask1a = cv2.inRange(hsv_frame1, lower_bound_1a, upper_bound_1a)

        # モルフォロジー演算
        kernel = np.ones((5, 5), np.uint8)
        mask1a = cv2.morphologyEx(mask1a, cv2.MORPH_OPEN, kernel)
        mask1a = cv2.morphologyEx(mask1a, cv2.MORPH_CLOSE, kernel)

        contours1a, _ = cv2.findContours(mask1a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours1a:
            largest_contour_1a = max(contours1a, key=cv2.contourArea)
            ((x1a, y1a), radius1a) = cv2.minEnclosingCircle(largest_contour_1a)
            nozzle_pos_side = (x1a, y1a)
            cv2.circle(frame1, (int(x1a), int(y1a)), int(radius1a), (0, 255, 0), 2)

            # HSV範囲の更新
            mask = np.zeros_like(mask1a)
            cv2.drawContours(mask, [largest_contour_1a], -1, 255, -1)
            mean_hsv = cv2.mean(hsv_frame1, mask=mask)[:3]
            hsv_color_1a = smooth_hsv(hsv_color_1a, np.array(mean_hsv), alpha=0.8)
            hsv_range_1a = np.array([hsv_color_1a - [15, 80, 80], hsv_color_1a + [15, 80, 80]])

    # 印刷物
    print_obj_pos_side = None
    print_obj_size_side = None
    if hsv_range_1b is not None:
        frame1_blurred = cv2.GaussianBlur(frame1, (5, 5), 0)
        hsv_frame1 = cv2.cvtColor(frame1_blurred, cv2.COLOR_BGR2HSV)

        lower_bound_1b = np.clip(hsv_range_1b[0], [0, 30, 30], [179, 255, 255]).astype(np.uint8)
        upper_bound_1b = np.clip(hsv_range_1b[1], [0, 30, 30], [179, 255, 255]).astype(np.uint8)

        mask1b = cv2.inRange(hsv_frame1, lower_bound_1b, upper_bound_1b)

        # モルフォロジー演算
        kernel = np.ones((5, 5), np.uint8)
        mask1b = cv2.morphologyEx(mask1b, cv2.MORPH_OPEN, kernel)
        mask1b = cv2.morphologyEx(mask1b, cv2.MORPH_CLOSE, kernel)

        contours1b, _ = cv2.findContours(mask1b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours1b:
            # 面積が大きい上位3つの輪郭を取得
            contours1b = sorted(contours1b, key=cv2.contourArea, reverse=True)[:3]
            for contour in contours1b:
                area = cv2.contourArea(contour)
                if area > 1000:  # 面積の閾値を上げる
                    x, y, w, h = cv2.boundingRect(contour)
                    print_obj_pos_side = (x + w / 2, y + h / 2)
                    print_obj_size_side = (w, h)
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # HSV範囲の更新
                    mask = np.zeros_like(mask1b)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    mean_hsv = cv2.mean(hsv_frame1, mask=mask)[:3]
                    hsv_color_1b = smooth_hsv(hsv_color_1b, np.array(mean_hsv), alpha=0.8)
                    hsv_range_1b = np.array([hsv_color_1b - [15, 80, 80], hsv_color_1b + [15, 80, 80]])

                    # **ここから緑色の認識アルゴリズムを統合**

                    # 印刷物のバウンディングボックス内のROIを取得
                    roi = frame1_blurred[y:y+h, x:x+w]
                    hsv_roi = hsv_frame1[y:y+h, x:x+w]

                    # ROI内でのマスクを作成
                    mask_roi = cv2.inRange(hsv_roi, lower_bound_1b, upper_bound_1b)

                    # モルフォロジー演算
                    kernel_roi = np.ones((3, 3), np.uint8)
                    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel_roi)
                    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel_roi)

                    # ROI内の輪郭を検出
                    contours_roi, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours_roi:
                        # 面積でフィルタリング
                        valid_contours_roi = [cnt for cnt in contours_roi if cv2.contourArea(cnt) >= 500]

                        if valid_contours_roi:
                            # 最大の輪郭を取得
                            largest_contour_roi = max(valid_contours_roi, key=cv2.contourArea)
                            # グローバル座標に変換
                            contour_global = largest_contour_roi + np.array([[[x, y]]])

                            # 輪郭を描画
                            cv2.drawContours(frame1, [contour_global], -1, (0, 255, 255), 2)

                            # トップポイントを取得
                            topmost = tuple(contour_global[contour_global[:,:,1].argmin()][0])
                            y_range = 5
                            print_surface = [pt for pt in contour_global if topmost[1] - y_range <= pt[0][1] <= topmost[1] + y_range]

                            if len(print_surface) > 1:
                                # 線形回帰による印刷平面の推定
                                x_points = np.array([pt[0][0] for pt in print_surface])
                                y_points = np.array([pt[0][1] for pt in print_surface])
                                x_points = x_points.reshape(-1, 1)
                                model = LinearRegression()
                                model.fit(x_points, y_points)

                                # 回帰直線の描画
                                x_start = x_points.min()
                                x_end = x_points.max()
                                y_start = int(model.predict([[x_start]])[0])  # 修正
                                y_end = int(model.predict([[x_end]])[0])    # 修正
                                cv2.line(frame1, (int(x_start), y_start), (int(x_end), y_end), (0, 0, 255), 2)

                                # 回帰直線の端点を保存
                                regression_line_endpoints_side = ((x_start, y_start), (x_end, y_end))
                    # **ここまで緑色の認識アルゴリズムを統合**

                    break  # 最初に見つかった適切な輪郭を使用
        else:
            print("ビデオ1で印刷物が見つかりませんでした。")

    # ビデオ2の処理（正面からの映像）
    # ノズルのマーカー
    nozzle_pos_front = None
    regression_line_endpoints_front = None  # 回帰直線の端点を初期化
    if hsv_range_2a is not None:
        hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        lower_bound_2a = np.clip(hsv_range_2a[0], [0, 30, 30], [179, 255, 255]).astype(np.uint8)
        upper_bound_2a = np.clip(hsv_range_2a[1], [0, 30, 30], [179, 255, 255]).astype(np.uint8)
        mask2a = cv2.inRange(hsv_frame2, lower_bound_2a, upper_bound_2a)

        # モルフォロジー演算
        kernel = np.ones((5, 5), np.uint8)
        mask2a = cv2.morphologyEx(mask2a, cv2.MORPH_OPEN, kernel)
        mask2a = cv2.morphologyEx(mask2a, cv2.MORPH_CLOSE, kernel)

        contours2a, _ = cv2.findContours(mask2a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours2a:
            largest_contour_2a = max(contours2a, key=cv2.contourArea)
            ((x2a, y2a), radius2a) = cv2.minEnclosingCircle(largest_contour_2a)
            nozzle_pos_front = (x2a, y2a)
            cv2.circle(frame2, (int(x2a), int(y2a)), int(radius2a), (0, 255, 0), 2)

            # HSV範囲の更新
            mask = np.zeros_like(mask2a)
            cv2.drawContours(mask, [largest_contour_2a], -1, 255, -1)
            mean_hsv = cv2.mean(hsv_frame2, mask=mask)[:3]
            hsv_color_2a = smooth_hsv(hsv_color_2a, np.array(mean_hsv), alpha=0.8)
            hsv_range_2a = np.array([hsv_color_2a - [15, 80, 80], hsv_color_2a + [15, 80, 80]])

    # 印刷物
    print_obj_pos_front = None
    print_obj_size_front = None
    if hsv_range_2b is not None:
        frame2_blurred = cv2.GaussianBlur(frame2, (5, 5), 0)
        hsv_frame2 = cv2.cvtColor(frame2_blurred, cv2.COLOR_BGR2HSV)

        lower_bound_2b = np.clip(hsv_range_2b[0], [0, 30, 30], [179, 255, 255]).astype(np.uint8)
        upper_bound_2b = np.clip(hsv_range_2b[1], [0, 30, 30], [179, 255, 255]).astype(np.uint8)

        mask2b = cv2.inRange(hsv_frame2, lower_bound_2b, upper_bound_2b)

        # モルフォロジー演算
        kernel = np.ones((5, 5), np.uint8)
        mask2b = cv2.morphologyEx(mask2b, cv2.MORPH_OPEN, kernel)
        mask2b = cv2.morphologyEx(mask2b, cv2.MORPH_CLOSE, kernel)

        contours2b, _ = cv2.findContours(mask2b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours2b:
            # 面積が大きい上位3つの輪郭を取得
            contours2b = sorted(contours2b, key=cv2.contourArea, reverse=True)[:3]
            for contour in contours2b:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    print_obj_pos_front = (x + w / 2, y + h / 2)
                    print_obj_size_front = (w, h)
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # HSV範囲の更新
                    mask = np.zeros_like(mask2b)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    mean_hsv = cv2.mean(hsv_frame2, mask=mask)[:3]
                    hsv_color_2b = smooth_hsv(hsv_color_2b, np.array(mean_hsv), alpha=0.8)
                    hsv_range_2b = np.array([hsv_color_2b - [15, 80, 80], hsv_color_2b + [15, 80, 80]])

                    # **ここから緑色の認識アルゴリズムを統合**

                    # 印刷物のバウンディングボックス内のROIを取得
                    roi = frame2_blurred[y:y+h, x:x+w]
                    hsv_roi = hsv_frame2[y:y+h, x:x+w]

                    # ROI内でのマスクを作成
                    mask_roi = cv2.inRange(hsv_roi, lower_bound_2b, upper_bound_2b)

                    # モルフォロジー演算
                    kernel_roi = np.ones((3, 3), np.uint8)
                    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel_roi)
                    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel_roi)

                    # ROI内の輪郭を検出
                    contours_roi, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours_roi:
                        # 面積でフィルタリング
                        valid_contours_roi = [cnt for cnt in contours_roi if cv2.contourArea(cnt) >= 500]

                        if valid_contours_roi:
                            # 最大の輪郭を取得
                            largest_contour_roi = max(valid_contours_roi, key=cv2.contourArea)
                            # グローバル座標に変換
                            contour_global = largest_contour_roi + np.array([[[x, y]]])

                            # 輪郭を描画
                            cv2.drawContours(frame2, [contour_global], -1, (0, 255, 255), 2)

                            # トップポイントを取得
                            topmost = tuple(contour_global[contour_global[:,:,1].argmin()][0])
                            y_range = 5
                            print_surface = [pt for pt in contour_global if topmost[1] - y_range <= pt[0][1] <= topmost[1] + y_range]

                            if len(print_surface) > 1:
                                # 線形回帰による印刷平面の推定
                                x_points = np.array([pt[0][0] for pt in print_surface])
                                y_points = np.array([pt[0][1] for pt in print_surface])
                                x_points = x_points.reshape(-1, 1)
                                model = LinearRegression()
                                model.fit(x_points, y_points)

                                # 回帰直線の描画
                                x_start = x_points.min()
                                x_end = x_points.max()
                                y_start = int(model.predict([[x_start]])[0])  # 修正
                                y_end = int(model.predict([[x_end]])[0])    # 修正
                                cv2.line(frame2, (int(x_start), y_start), (int(x_end), y_end), (0, 0, 255), 2)

                                # 回帰直線の端点を保存
                                regression_line_endpoints_front = ((x_start, y_start), (x_end, y_end))
                    # **ここまで緑色の認識アルゴリズムを統合**

                    break
        else:
            print("ビデオ2で印刷物が見つかりませんでした。")

    # グラフの更新
    # ノズルの位置を3次元的に推定
    if nozzle_pos_side is not None and nozzle_pos_front is not None:
        nozzle_pos = (nozzle_pos_front[0], nozzle_pos_side[0])  # X: ビデオ2のX、Y: ビデオ1のX
    else:
        nozzle_pos = None

    # 印刷物の位置とサイズを3次元的に推定
    if print_obj_pos_side is not None and print_obj_pos_front is not None and \
       print_obj_size_side is not None and print_obj_size_front is not None:
        print_obj_pos = (print_obj_pos_front[0], print_obj_pos_side[0])  # X: ビデオ2のX、Y: ビデオ1のX
        print_obj_size = (print_obj_size_front[0], print_obj_size_side[0])  # 幅: ビデオ2の幅、高さ: ビデオ1の幅
    else:
        print_obj_pos = None
        print_obj_size = None

    update_graph(nozzle_pos, print_obj_pos, print_obj_size, regression_line_endpoints_side, regression_line_endpoints_front)

    # ノズルと印刷物の距離を計算
    if nozzle_pos is not None and print_obj_pos is not None:
        distance = np.linalg.norm(np.array(nozzle_pos) - np.array(print_obj_pos))
        if distance > threshold:
            print("印刷失敗: ノズルと印刷物の位置がしきい値を超えました。距離:", distance)
    else:
        if print_obj_pos is None:
            print("印刷物が見つかりませんでした。")
        if nozzle_pos is None:
            print("ノズルが見つかりませんでした。")

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
