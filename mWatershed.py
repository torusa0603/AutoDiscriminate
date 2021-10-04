
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import cHandleJsonfile
import os
import math

def doWatershed(nbShowFlag, nstrResultFolderPate):
  if nstrResultFolderPate == "":
    nstrResultFolderPate = "."
  for fn in glob.glob(os.path.join(nstrResultFolderPate, "img/*")):
    #画像の読み込み
    img = cv2.imread(fn)
    if nbShowFlag:
      showPicture(img, 'Sorce')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = adjust(gray, 3.0, 0.0)
    if nbShowFlag:
      showPicture(gray, 'gray')
    #BGR→HSV変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, hsv_s, hsv_v = cv2.split(hsv)
    hsv_s = adjust(hsv_s, 3.0, 0.0)
    hsv_v = adjust(hsv_v, 3.0, 0.0)
    if nbShowFlag:
      showPicture(hsv_s, 'hsv_s')
    if nbShowFlag:
      showPicture(hsv_v, 'hsv_v')
    #彩度の高低で画像を二値化する
#    lower = (0, 150, 0)
    lower = (30, 20, 0)
    upper = (90, 255, 255)
    bin_img_green = cv2.inRange(hsv, lower, upper, 255)
    if nbShowFlag:
      showPicture(bin_img_green, 'bin_img_green')
      #彩度の高低で画像を二値化する
#    lower = (0, 150, 0)
    lower = (0, 150, 0)
    upper = (179, 255, 255)
    bin_img_satuation = cv2.inRange(hsv, lower, upper, 255)
    if nbShowFlag:
      showPicture(bin_img_satuation, 'bin_img_satuation')
    #明度の高低で画像を二値化する
    lower = (0, 0,  50)
    upper = (179, 255, 255)
    bin_img_white = cv2.inRange(hsv, lower, upper, 255)
    if nbShowFlag:
      showPicture(bin_img_white, 'bin_img_white')
    #彩度が高いor明度が高い部分を白くした二値画像を作成
    bin_img = cv2.add(bin_img_satuation, bin_img_white)
    bin_img = cv2.add(bin_img, bin_img_green)
    if nbShowFlag:
      showPicture(bin_img, 'bin_img')
    #オープニング処理
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations = 2)
    #クロージング処理
    kernel = np.ones((3,3),np.uint8)
    opening_closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel,iterations = 5)
    if nbShowFlag:
      showPicture(opening_closing, 'opening_closing')
    #明確な背景を抽出
    sure_bg = cv2.dilate(opening_closing,kernel,iterations=2)
    if nbShowFlag:
      showPicture(sure_bg, 'sure_bg')
    #距離変換処理
    dist_transform = cv2.distanceTransform(opening_closing,cv2.DIST_L2,5)
    if nbShowFlag:
      plt.imshow(dist_transform)
      plt.show()
      plt.clf()
    #明確な前景を抽出
    _, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    if nbShowFlag:
      plt.imshow(sure_fg,cmap='gray')
      plt.show()
      plt.clf()
    #前景・背景以外の部分を抽出
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    if nbShowFlag:
      plt.imshow(unknown,cmap='gray')
      plt.show()
      plt.clf()
    #オブジェクトごとにラベル（番号）を振っていく
    nLabels, markers = cv2.connectedComponents(sure_fg)
#    i_number_of_color_radius = DistincteLabels(img, nLabels, markers)
    if nbShowFlag:
      plt.imshow(markers)
      plt.show()
      plt.clf()
    markers = markers+1
    markers[unknown==255] = 0
    if nbShowFlag:
      plt.imshow(markers)
      plt.show()
      plt.clf()
    opening_img = cv2.cvtColor(opening_closing, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(opening_img,markers)
    #ここまでは来ている
#    nLabels, nMarkers = cv2.connectedComponents(markers)
    i_number_of_color_radius = DistincteLabels(img, nLabels, markers, nstrResultFolderPate)
    np.savetxt(os.path.join(nstrResultFolderPate, "result/color_radius.csv"), i_number_of_color_radius, delimiter=",", fmt="%d")
    if nbShowFlag:
      plt.imshow(markers)
      plt.show()
      plt.close()
    img[markers == -1] = [0,255,0]
    if nbShowFlag:
      showPicture(img, 'img_marked')

def adjust(nimgSorce, nfAlpha=1.0, nfBeta=0.0):
    # 積和演算を行う。
    img_dst = nfAlpha * nimgSorce + nfBeta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(img_dst, 0, 255).astype(np.uint8)

def showPicture(nimgPicture, nstrTitle):
  #画像高さが1080pix以下になるようにリサイズする
  i_height, i_width = nimgPicture.shape[:2]
  i_ratio = 1
  if (i_height / i_ratio) > 1080:
      i_ratio += 1
  img_resize_picture = cv2.resize(nimgPicture, dsize=(int(i_width / i_ratio), int(i_height / i_ratio)))
  #画像を出力する
  cv2.imshow(nstrTitle,img_resize_picture)
  #qボタンが押されたら終了する
  key = ord('a')
  while key != ord('q'):
    key = cv2.waitKey(1)
  cv2.destroyAllWindows()

def DistincteLabels(nimgSrc, niLabelNumber, nlsLabelTable, nstrResultFolderPate):
  m_dicSettingElements={"NumberPerOneMiliMeter" : "", "MaxRadius" : ""} #設定ファイルから読み込む変数群を格納した辞書型変数 
  cHandleJsonfile.ReadElementsFromJsonfile(os.path.join(nstrResultFolderPate,"Setting.json"), 'mWaterShed', m_dicSettingElements)
  print("1")
  i_number_of_color = np.zeros([6, m_dicSettingElements["MaxRadius"]], dtype=np.int) #(赤,黄,緑,青,紫,白)ごとの個数
  for label in range(2, niLabelNumber +1):
    i_label_group_index = np.where(nlsLabelTable == label)
    i_array_label_bgr = nimgSrc[i_label_group_index]
    i_width = i_array_label_bgr.shape[0]
    i_ret_radius = CalculateRadius(i_width, m_dicSettingElements["MaxRadius"], m_dicSettingElements["NumberPerOneMiliMeter"])
    img_label_bgr = np.zeros((1, i_width, 3), dtype='uint8')
    img_label_bgr[0, :, :] = i_array_label_bgr
    img_hls = cv2.cvtColor(img_label_bgr, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(img_hls)
    i_h_mean = h.mean()
    i_l_mean = l.mean()
    s_l_mean = s.mean()
    if i_l_mean > 100 and s_l_mean < 100:
      i_number_of_color[5][i_ret_radius - 1] += 1
    else:
      i_ret_color = DistincteColor(i_h_mean)
      i_number_of_color[i_ret_color][i_ret_radius - 1] += 1
  return i_number_of_color

def DistincteColor(niHue):
  #hueの値を(赤,黄,緑,青,紫)に区分けする為の区切り数値
  i_color_band = (10,40,80,140,160)
  #色判定：赤
  if i_color_band[0] > niHue <= i_color_band[4]:
    i_ret = 0
  #色判定：黄
  elif i_color_band[0] <= niHue < i_color_band[1]:
    i_ret = 1
  #色判定：緑
  elif i_color_band[1] <= niHue < i_color_band[2]:
    i_ret = 2
  #色判定：青
  elif i_color_band[2] <= niHue < i_color_band[3]:
    i_ret = 3
  #色判定：紫
  elif i_color_band[3] <= niHue < i_color_band[4]:
    i_ret = 4
  #hueを判定した色の要素のみに1を入れて返す
  return i_ret

def CalculateRadius(niNamberOfPixels, niMaxRadius, niNumberPerOneMiliMeter):
  for i_radius in range(niMaxRadius):
    i_area = math.pi * (i_radius + 1) ** 2 * niNumberPerOneMiliMeter**2
    if niNamberOfPixels < i_area:
      break
  return i_radius
    

if __name__ == '__main__':
  doWatershed(True, "")