import cv2
import os

def main():
    # 処理されたフレームのリストを取得（名前順にソート）
    processed_frames = sorted([f for f in os.listdir('.') if f.startswith('processed_frame_')])

    # 例として最初のフレームから動画の高さと幅を取得
    first_frame = cv2.imread(processed_frames[0])
    height, width, layers = first_frame.shape

    # 動画作成のための設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # フォーマット指定
    out = cv2.VideoWriter('output_video.mp4', fourcc, 10.0, (width, height))

    for frame_file in processed_frames:
        frame = cv2.imread(frame_file)
        out.write(frame)  # 動画にフレームを追加

    out.release()

if __name__ == "__main__":
    main()
