import cv2
import numpy as np

#メインはこっから借りてきた　https://qiita.com/mix_dvd/items/98feedc8c98bc7790b30

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    GAUSSIAN_WINDOW_NAME = "AnytimeMask"

    DEVICE_ID = 0
    src = cv2.imread('pic\mask.png', -1)

    # 分類器の指定
    cascade_file = "Python\haarcascade_frontalface_alt2.xml"

    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    #黒背景生成
    black = np.zeros((height, width, 3), np.uint8)
    
    # ウィンドウの準備
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray,scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

        # 検出した顔に印を付ける
        if len(face_list) > 0:
            for (x, y, w, h) in face_list:
                #顔の大きさにマスクをリサイズ
                resizesrc = cv2.resize(src,(w,h))
                mask_material = resizesrc[:,:,3]  # アルファチャンネルだけ抜き出す
                mask_material = cv2.cvtColor(mask_material, cv2.COLOR_GRAY2BGR)  # 3色分に増やす
            
                #黒背景にマスクの貼り付け
                mask = black.copy()
                mask[y:h+y,x:w+x] = mask_material
            
                #黒背景にresizesrcを貼り付け
                front_img = black.copy()
                front_img_material = resizesrc[:,:,:3] #アルファチャンネルをカット
                front_img[y:h+y,x:w+x] = front_img_material

                #マスク反転
                mask = cv2.bitwise_not(mask)

                #カメラ画像からマスク部分を切り抜く
                back_img = cv2.bitwise_and(c_frame,mask)

                img_out =  cv2.bitwise_or(front_img, back_img)

            # フレーム表示
            cv2.imshow(GAUSSIAN_WINDOW_NAME, img_out)
        else:
            cv2.imshow(GAUSSIAN_WINDOW_NAME, c_frame)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()