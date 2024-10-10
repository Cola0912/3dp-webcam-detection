import cv2
import numpy as np

# カメラをキャプチャ
cap = cv2.VideoCapture(0)

# カメラが開けているか確認
if not cap.isOpened():
    print("カメラを開けませんでした。デバイスを確認してください。")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# トラックする対象を指定する（最初のフレームでROIを選択する）
ret, frame = cap.read()
if not ret or frame is None:
    print("カメラの映像を取得できませんでした。")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# ウィンドウを表示してROIを選択
cv2.imshow("ROI選択", frame)
roi = cv2.selectROI("ROI選択", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("ROI選択")

# ROIが選択されなかった場合の処理
if roi == (0, 0, 0, 0):
    print("ROIが選択されませんでした。")
    cap.release()
    cv2.destroyAllWindows()
    exit()

x, y, w, h = roi
template = frame[y:y+h, x:x+w]

# テンプレートのサイズを取得
template_height, template_width = template.shape[:2]

while True:
    # カメラからフレームを取得
    ret, frame = cap.read()
    if not ret:
        break

    # 指定した範囲に対してテンプレートマッチングを実行
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 最も一致する場所の座標
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

    # マッチした場所を矩形で描画
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # フレームを表示
    cv2.imshow("Tracking", frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラ解放とウィンドウ破棄
cap.release()
cv2.destroyAllWindows()
