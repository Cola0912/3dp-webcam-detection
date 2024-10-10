import cv2

# 2つのカメラをキャプチャ開始 (カメラIDは0と1)
cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(6)

if not cap1.isOpened() or not cap2.isOpened():
    print("カメラのいずれかを開くことができませんでした。")
    cap1.release()
    cap2.release()
    exit()

while True:
    # 各カメラからフレームをキャプチャ
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("いずれかのフレームを読み取ることができませんでした。")
        break

    # フレームをウィンドウに表示
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # 'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap1.release()
cap2.release()
cv2.destroyAllWindows()
