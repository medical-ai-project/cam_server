from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from scipy.ndimage import zoom

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from decimal import Decimal
import os


# you need A pre-trained efficientNet model.
# Don NOT delete the below
import efficientnet.tfkeras as efn


def get_prediction(input_img: np.ndarray, model):
    # input_img = input_img.reshape(1, 600, 600, 3)
    predictions = model.predict(input_img)
    loss = predictions[0]
    print(predictions)
    print(f'loss: {loss}')
    return loss


def generate_grad_cam_images(input_img: np.ndarray, model):
    # grad-CAMの出力
    # NumPy配列をテンソルに変換
    input_img_tensor = tf.convert_to_tensor(input_img.reshape(1, 600, 600, 3))

    # target_layer = model.get_layer("top_conv")
    target_layer = model.get_layer("block7c_project_conv")
    intermediate_model = Model(inputs=[model.inputs], outputs=[target_layer.output, model.output])
    with tf.GradientTape() as tape:
        tape.watch(input_img_tensor)  # ここでテンソルをwatch
        conv_output, predictions = intermediate_model(input_img_tensor)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    print(loss)


    # 勾配を取得
    grads = tape.gradient(loss, conv_output)[0]

    # グローバル平均プーリング
    weights = np.mean(grads, axis=(0, 1))

    # Grad-CAMの計算
    # conv_output[0]はバッチのindex0番目に対応している。今回は１枚のみなので[0]のみ存在している。
    cam = np.dot(conv_output[0], weights)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / cam.max()  # 正規化


    '''
    アップサンプリング前
    '''
    # 可視化
    # plt.subplot(121)
    # plt.imshow(cam, cmap="jet")

    # plt.subplot(122)
    # plt.imshow(input_img[0], cmap='gray')

    # plt.savefig('output/your_filename.png')


    '''
    アップサンプリング適応
    '''
    # 50x50の特徴マップを224x224にアップサンプリング
    zoom_factor = 600 / 19  # 600: ターゲットのサイズ, 19: 元のサイズ
    cam_resized = zoom(cam, zoom_factor)

    # grad_CAMの位置を調整する。
    # shift_x, shift_y = 4, 0  # ずらすピクセル数
    # cam_resized = np.roll(cam_resized, shift_x, axis=0)
    # cam_resized = np.roll(cam_resized, shift_y, axis=1)

    # # プロット
    # plt.subplot(131)
    # plt.title("Original Image")
    # plt.imshow(input_img[0], cmap="gray")

    # plt.subplot(132)
    # plt.title("grad_CAM")
    # plt.imshow(cam_resized, cmap="jet")


    # plt.subplot(133)
    # plt.title("Combined")
    # plt.imshow(input_img[0], cmap="gray")
    # plt.imshow(cam_resized, cmap='jet', alpha=0.5)



    cmap = plt.get_cmap('jet')
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # 範囲を変更
    shift_amount = 10
    new_cmaplist = [(1, 1, 1, 0.0) for _ in range(shift_amount)] + cmaplist[:-shift_amount]


    for i in range(45):
        new_cmaplist[i] = (1, 1, 1, 0.0)  # (R, G, B, Alpha)

    cmap_custom = mcolors.LinearSegmentedColormap.from_list('custom_cmap', new_cmaplist, cmap.N)

    base_output_path = os.path.dirname(os.path.abspath(__file__))
    for i in range(10, -1, -1):
        value = float(Decimal(i) / Decimal(10))  # 厳密に小数を計算する。
        print(value) 
        plt.imshow(input_img[0], cmap='gray')
        plt.imshow(cam_resized, cmap=cmap_custom, alpha=1, vmin=value)
        file_path = os.path.join(base_output_path, f"output/grad_cam{i}.png")
        plt.savefig(file_path)
