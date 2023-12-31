import cv2

def find_camera_index():
    index = 1
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

camera_indexes = find_camera_index()
print(f"Detected camera at indexes: {camera_indexes}")

# カメラが見つかった場合、最初のカメラを試しに表示する
if camera_indexes:
    cap = cv2.VideoCapture(camera_indexes[0])
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Camera", frame)
        cv2.waitKey(3000)  # 3秒間表示
    cap.release()
cv2.destroyAllWindows()
