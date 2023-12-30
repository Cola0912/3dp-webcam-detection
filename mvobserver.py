import cv2

def main():
    video_path = 'input_video.mp4'  # 読み込む動画のパス
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画ファイルを開けませんでした。")
        return

    # FPSの調整
    fps = 0.000000001
    cap.set(cv2.CAP_PROP_FPS, fps)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 動画が終了したら終了

        # フレームをファイルに保存（detector.pyがこれを読む）
        frame_path = f'temp_frame_{frame_id}.jpg'
        cv2.imwrite(frame_path, frame)
        
        # detector.pyを呼び出す（例: subprocessを使うなどして）
        # subprocess.call(['python', 'detector.py', frame_path])
        
        frame_id += 1

    cap.release()

if __name__ == "__main__":
    main()
