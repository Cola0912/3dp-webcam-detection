import cv2

def display_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        cv2.imshow('Processed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("画像を読み込めませんでした。")

def main(processed_image_path):
    display_image(processed_image_path)

if __name__ == "__main__":
    import sys
    processed_image_path = sys.argv[1]  # 処理済み画像のパスをコマンドライン引数から取得
    main(processed_image_path)
