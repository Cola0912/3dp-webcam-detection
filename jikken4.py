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
ax.set_ylim(0, 1000)  # Y軸は奥行き（距離）

# グラフのタイトルと軸ラベルを設定
ax.set_title("Distance between Circles (Depth)")
ax.set_xlabel("ugoiterunora naaaa")  # 横軸の名前を変更
ax.set_ylabel("Depth (distance)")  # 縦軸の名前を変更

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

    recognized_circles = []

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

                    # 一致度がある程度良ければ認識リストに追加
                    if color_diff < 200:
                        recognized_circles.append((int(x), int(y), int(radius)))

                        # マゼンタの円をリザルトフレームに描画
                        cv2.circle(result_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # 認識された円が2つある場合、その距離を計算
    if len(recognized_circles) == 2:
        (x1, y1, r1), (x2, y2, r2) = recognized_circles
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # グラフ上に距離をプロット（Y軸として扱う）
        point.set_data((x1 + x2) / 2, distance)  # X位置は2つの中心の平均、Y軸に距離をプロット
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
