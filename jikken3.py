import cv2
import numpy as np
import matplotlib.pyplot as plt

# ウェブカメラから映像を取得
cap = cv2.VideoCapture(0)

# グラフの設定
plt.ion()  # インタラクティブモードを有効化
fig, ax = plt.subplots()
point, = ax.plot([], [], 'bo')  # 青い点をプロット
ax.set_xlim(0, 640)  # カメラのフレーム幅に合わせてX軸を設定
ax.set_ylim(-1, 1)  # Y軸は固定（使わないので0固定）

# グラフのタイトルと軸ラベルを設定
ax.set_title("X Movement of Magenta Circle")
ax.set_xlabel("ugoiterunora naaaa")  # 横軸の名前を変更
ax.set_ylabel("Fixed")  # 縦軸の名前を変更

# 初期プロットを描画してキャッシュを作成
fig.canvas.draw()

# トラックバーのコールバック関数（何もしない）
def nothing(x):
    pass

# ウィンドウを作成し、トラックバーを追加
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Lower H', 'Trackbars', 125, 180, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', 155, 180, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', 120, 255, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', 70, 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Cross Threshold', 'Trackbars', 50, 255, nothing)  # 十字のしきい値用トラックバー

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # トラックバーの位置からHSVのしきい値を取得
    lower_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
    upper_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
    lower_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
    upper_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
    lower_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
    upper_v = cv2.getTrackbarPos('Upper V', 'Trackbars')

    # 十字のしきい値を取得
    cross_threshold = cv2.getTrackbarPos('Cross Threshold', 'Trackbars')

    # フレームをHSV色空間に変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # マゼンタ色を検出するマスクを作成
    lower_magenta = np.array([lower_h, lower_s, lower_v])
    upper_magenta = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower_magenta, upper_magenta)

    # 輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    best_match_score = float('inf')

    # フレームのコピーを作成してマゼンタの円の認識結果を描画
    result_frame = frame.copy()

    for contour in contours:
        # 輪郭の面積を計算して小さなノイズを除去
        area = cv2.contourArea(contour)
        if area > 1000:
            # 最小外接円を取得
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 20:  # 最小半径を設定して小さなノイズを除去
                # 円の形状に近いかどうかを確認
                circularity = 4 * np.pi * (area / (cv2.arcLength(contour, True) ** 2))
                if 0.7 < circularity <= 1.2:
                    # 領域の平均色を計算
                    mean_color = cv2.mean(frame[int(y-radius):int(y+radius), int(x-radius):int(x+radius)])[:3]

                    # 色の一致度を評価（マゼンタにどれだけ近いか）
                    color_diff = abs(mean_color[2] - 255) + abs(mean_color[1] - 0) + abs(mean_color[0] - 255)

                    # 大きさと色の一致度を使って最適な輪郭を選択
                    match_score = abs(radius - 50) + color_diff  # 大きさの目標半径50に基づいて評価
                    if match_score < best_match_score:
                        best_match_score = match_score
                        best_match = (int(x), int(y), int(radius))

                    # マゼンタの円をリザルトフレームに描画
                    cv2.circle(result_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # 最も適した円を認識
    if best_match is not None:
        center_x, center_y, radius = best_match

        # 円の領域を切り取る
        x1, y1, x2, y2 = max(0, center_x - radius), max(0, center_y - radius), min(frame.shape[1], center_x + radius), min(frame.shape[0], center_y + radius)
        roi = frame[y1:y2, x1:x2]

        # 円の中心に青い点を描画
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # 円内部の黒い十字を認識
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, cross_threshold, 255, cv2.THRESH_BINARY_INV)  # 十字のしきい値をトラックバーで設定

        # 十字の輪郭を検出
        inner_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_cross = False

        for inner_contour in inner_contours:
            # 十字の形状の検出
            rect = cv2.boundingRect(inner_contour)
            aspect_ratio = rect[2] / rect[3]  # 横幅 / 高さ

            if 0.8 < aspect_ratio < 1.2:  # アスペクト比が1に近い（正方形に近い）場合
                if 100 < cv2.contourArea(inner_contour) < 2000:  # 面積の条件で絞り込み
                    # 十字と認識された部分を描画
                    cv2.drawContours(roi, [inner_contour], -1, (255, 0, 0), 2)
                    found_cross = True

        # 十字が見つかった場合のみトラッキング対象とする
        if found_cross:
            # グラフ上にX位置をプロット
            point.set_data(center_x, 0)  # Y位置は0に固定
            ax.draw_artist(ax.patch)
            ax.draw_artist(point)
            fig.canvas.flush_events()

    # フレームとリザルトフレームを表示
    cv2.imshow('Magenta Circle Detection with Cross Check and X Movement Plot', frame)
    cv2.imshow('Recognition Result', result_frame)  # 認識結果のリザルトウィンドウ

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # インタラクティブモードを無効化
plt.show()
