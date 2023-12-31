import cv2

def capture_frame(camera_path, output_path):
    cap = cv2.VideoCapture(camera_path)  # カメラデバイスのパス
    if not cap.isOpened():
        print("ob:カメラが開けません")
        return

    ret, frame = cap.read()
    if ret:
        print("ob:撮影しました")
        cv2.imwrite(output_path, frame)
    else:
        print("ob:フレームの取得に失敗しました")

    cap.release()

if __name__ == "__main__":
    camera_path = '/dev/video4'  # カメラデバイスのパス
    output_path = 'detector_bait.jpg'  # 保存する画像のパス
    capture_frame(camera_path, output_path)
