import os
from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.colors import rgb2hex
import random

def is_similar_normal(n1, n2, threshold=0.1):
    """ 2つの法線ベクトルが類似しているか判定する """
    return np.linalg.norm(n1 - n2) < threshold

# ファイルパスを展開する
stl_path = os.path.expanduser('~/3dp-webcam-detection/PrintTestModels/stl/3DBenchy.stl')

# STLファイルを読み込む
your_mesh = mesh.Mesh.from_file(stl_path)

# プロットの準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# メッシュを単純化してプロットし、ランダムな色を割り当てる
prev_normal = None
for i in range(len(your_mesh.vectors)):
    current_normal = your_mesh.normals[i]

    # 前の法線ベクトルと現在の法線ベクトルが類似していない場合のみプロット
    if prev_normal is None or not is_similar_normal(prev_normal, current_normal):
        vectors = your_mesh.vectors[i]
        # ランダムな色を生成
        color = rgb2hex([random.random() for _ in range(3)])
        ax.add_collection3d(art3d.Poly3DCollection([vectors], color=color))

    prev_normal = current_normal

# モデルのサイズを基に軸のリミットを設定
min_point = your_mesh.points.min(axis=0)
max_point = your_mesh.points.max(axis=0)

max_range = np.array([max_point[i] - min_point[i] for i in range(3)]).max() / 2.0
mid_x = (max_point[0] + min_point[0]) * 0.5
mid_y = (max_point[1] + min_point[1]) * 0.5
mid_z = (max_point[2] + min_point[2]) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# プレビュー
plt.show()
