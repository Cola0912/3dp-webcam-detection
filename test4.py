import cv2

# カメラのインデックスを指定
camera_indices = [0, 4]  # 使用するカメラのインデックスをリストで指定

# VideoCaptureオブジェクトのリストを作成
caps = []
for idx in camera_indices:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"カメラ {idx} を開くことができません")
    else:
        caps.append((idx, cap))

if not caps:
    print("有効なカメラが見つかりませんでした")
else:
    while True:
        for idx, cap in caps:
            ret, frame = cap.read()
            if not ret:
                print(f"カメラ {idx} からフレームを取得できませんでした")
                continue

            # フレームを表示
            window_name = f"Camera {idx}"
            cv2.imshow(window_name, frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    for idx, cap in caps:
        cap.release()
    cv2.destroyAllWindows()
