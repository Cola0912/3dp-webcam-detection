import cv2
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("ob:画像の読み込みに失敗しました。ファイルパスを確認してください。")
        exit()
    else:
        print("ob:画像の読み込みに成功しました。")
    return image

def main():
    home = os.path.expanduser("~")
    image_path = os.path.join(home, '3dp-webcam-detection/testpicturs/Print_failure-3.jpg')
    image = load_image(image_path)
    cv2.imwrite('detecter_bait.jpg', image)  # 画像を一時ファイルとして保存

if __name__ == "__main__":
    main()