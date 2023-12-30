import cv2
import matplotlib.pyplot as plt

def display_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

def main():
    display_image('detecter_poop.jpg')  # detector.pyからの処理済み画像を表示

if __name__ == "__main__":
    main()