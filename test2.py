import cv2
import numpy as np

def nothing(x):
    pass

# カメラの初期化
camera_index = 4  # 使用するカメラのインデックスを指定
cap = cv2.VideoCapture(camera_index)

# ウィンドウの作成
cv2.namedWindow('Frame')
cv2.namedWindow('Mask')

# トラックバーを配置するウィンドウを作成
cv2.namedWindow('Trackbars')

# トラックバーの作成
cv2.createTrackbar('Hue Min', 'Trackbars', 0, 180, nothing)
cv2.createTrackbar('Hue Max', 'Trackbars', 180, 180, nothing)
cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました")
        break

    # フレームをリサイズ（必要に応じて）
    # frame = cv2.resize(frame, (640, 480))

    # ガウシアンブラーでノイズを軽減
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # HSV色空間に変換
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # トラックバーの値を取得
    h_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
    s_min = cv2.getTrackbarPos('Sat Min', 'Trackbars')
    s_max = cv2.getTrackbarPos('Sat Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('Val Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('Val Max', 'Trackbars')

    # 色範囲を設定
    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max])

    # マスクの作成
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # マスクのノイズ除去（オプション）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # マスクを適用した結果を取得
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 輪郭の検出
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 検出された輪郭を描画
    if contours:
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # 結果を表示
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
