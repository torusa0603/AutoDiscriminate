import cv2
import numpy as np
from datetime import datetime
import glob

def main():
    score = 127
    # 入力画像の取得
    for fn in glob.glob("./img/*"):
        im = cv2.imread(fn)

        im = cv2.blur(im, (5, 5))
        cv2.imshow(im)

        # グレースケール変換
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # 2値化
        gray = cv2.threshold(gray, score, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # ラベリング処理
        label = cv2.connectedComponentsWithStats(gray)

        saveImgByTime("./result/",gray)

        # ブロブ情報を項目別に抽出
        n = label[0] - 1
        data = np.delete(label[2], 0, 0)
        center = np.delete(label[3], 0, 0)

        # ラベルの個数nだけ色を用意
        print("ブロブの個数:", n)
        print("各ブロブの外接矩形の左上x座標", data[:,0])
        print("各ブロブの外接矩形の左上y座標", data[:,1])
        print("各ブロブの外接矩形の幅", data[:,2])
        print("各ブロブの外接矩形の高さ", data[:,3])
        print("各ブロブの面積", data[:,4])
        print("各ブロブの中心座標:\n",center)

        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(im,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('detected circles',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def saveImgByTime(dirPath,img):
    # 時刻を取得
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = dirPath + date + ".png"
    cv2.imwrite(path, img) # ファイル保存
    print("saved: ", path)

if __name__ == '__main__':
    main()