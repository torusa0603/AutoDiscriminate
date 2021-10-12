import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import cHandleJsonfile
import os
import math

def main(nbShowFlag, nstrResultFolderPath, nbDetail):
  if nstrResultFolderPath == "":
    nstrResultFolderPath = "."
  for fn in glob.glob(os.path.join(nstrResultFolderPath, "img/*")):
    #画像の読み込み
    img_raw = cv2.imread(fn)
    if nbShowFlag:
      showPicture(img_raw, 'Sorce')
###  使用する可能性があるためコメントとして残す  ###################
#    #生画像をグレイスケールに変換
#    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
#    img_gray = adjust(img_gray, 3.0, 0.0)
#    if nbShowFlag:
#      showPicture(img_gray, 'gray')
#    #hsvのそれぞれの値に補正を加える                             
#    _, hsv_s, hsv_v = cv2.split(hsv)
#    hsv_s = adjust(hsv_s, 3.0, 0.0)
#    hsv_v = adjust(hsv_v, 3.0, 0.0)
#    if nbShowFlag:
#      showPicture(hsv_s, 'hsv_s')
#    if nbShowFlag:
#      showPicture(hsv_v, 'hsv_v')
#    #彩度の高低で画像を二値化する
#    lower = (0, 150, 0)
#    upper = (179, 255, 255)
#    bin_img_satuation = cv2.inRange(hsv, lower, upper, 255)
#    if nbShowFlag:
#      showPicture(bin_img_satuation, 'bin_img_satuation')
################################################################
    #BGR→HSV変換
    img_hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
    #色相の値が緑であるかで画像を二値化する
    arr_lower_1 = (40, 30, 30)
    arr_upper_1 = (80, 255, 255)
    img_bind_green = cv2.inRange(img_hsv, arr_lower_1, arr_upper_1, 255)
    if nbShowFlag:
      showPicture(img_bind_green, 'bin_img_green')
    #明度の高低で画像を二値化する
    arr_lower_2 = (0, 0,  40)
    arr_upper_2 = (179, 255, 255)
    img_bind_white = cv2.inRange(img_hsv, arr_lower_2, arr_upper_2, 255)
    if nbShowFlag:
      showPicture(img_bind_white, 'bin_img_white')
    #彩度が高いor明度が高い部分を白、その他の部分を黒しとした二値画像を作成
    img_bind_mask = cv2.add(img_bind_green, img_bind_white)
    #img_bind_mask = cv2.add(bin_img, bin_img_green)
    if nbShowFlag:
      showPicture(img_bind_mask, 'bin_img')
    #デッドロック対策のカウンターをセット
    i_count = 1
    #WaterShedを行う
    b_ret, i_number_labels, arr_center,  i_number_of_color_and_radius, npdata_number_of_color_and_radius = doWatershedMethod(img_raw, img_bind_mask, nstrResultFolderPath, nbShowFlag, nbDetail, i_count)
    if b_ret:
      np.savetxt(os.path.join(nstrResultFolderPath, "result/color_radius.csv"), i_number_of_color_and_radius, delimiter=",", fmt="%d")
      if nbDetail:
        for i in range(i_number_labels):
          cv2.putText(img_raw, "ID: " +str(i + 1),                                      ((int)(arr_center[i][0] - 40),(int)(arr_center[i][1] + 30)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
          cv2.putText(img_raw, "Color: " +str(npdata_number_of_color_and_radius[i][0]), ((int)(arr_center[i][0] - 40),(int)(arr_center[i][1] + 60)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
          cv2.putText(img_raw, "Size: " +str(npdata_number_of_color_and_radius[i][1]),  ((int)(arr_center[i][0] - 40),(int)(arr_center[i][1] + 90)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
          cv2.imwrite(os.path.join(nstrResultFolderPath, "img/result.png"), img_raw)
      if nbShowFlag:
        showPicture(img_raw, 'img_marked')





def doWatershedMethod(nimgRaw, nimgBindMask, nstrResultFolderPath, nbShowFlag, nbDetail, niCount):
  #デッドロック対策
  niCount += 1
  #回帰関数を10回繰り返した場合に終了する
  if niCount == 10:
    return False, None, None, None, None
  #オープニング処理
  i_kernel_5_5 = np.ones((5,5),np.uint8)
  img_opening = cv2.morphologyEx(nimgBindMask,cv2.MORPH_OPEN,i_kernel_5_5,iterations = 2)
  #クロージング処理
  i_kernel_3_3 = np.ones((3,3),np.uint8)
  img_opening_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, i_kernel_3_3,iterations = 5)
  if nbShowFlag:
    showPicture(img_opening_closing, 'opening_closing')
  #白色ピクセルの個数が少なければ、終了
  img_closing_5_5 = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, i_kernel_5_5,iterations = 5)
  if nbShowFlag:
    showPicture(img_closing_5_5, 'img_closing_5_5')
  count_white_bit = cv2.countNonZero(img_closing_5_5)
  if (count_white_bit < 5000):
    return False, None, None, None, None
  #明確な背景を抽出
  img_sure_bg = cv2.dilate(img_opening_closing, i_kernel_3_3, iterations=2)
  if nbShowFlag:
    showPicture(img_sure_bg, 'sure_bg')
  #距離変換処理
  img_dist_transform = cv2.distanceTransform(img_opening_closing, cv2.DIST_L2, 5)
  if nbShowFlag:
    plt.imshow(img_dist_transform)
    plt.show()
    plt.clf()
  #明確な前景を抽出
  _, img_sure_fg = cv2.threshold(img_dist_transform, 0.5*img_dist_transform.max(), 255, 0)
  if nbShowFlag:
    plt.imshow(img_sure_fg,cmap='gray')
    plt.show()
    plt.clf()
  #前景・背景以外の部分を抽出
  img_sure_fg = np.uint8(img_sure_fg)
  img_unknown = cv2.subtract(img_sure_bg, img_sure_fg)
  if nbShowFlag:
    plt.imshow(img_unknown,cmap='gray')
    plt.show()
    plt.clf()
  #オブジェクトごとにラベル（番号）を振っていく
  if nbDetail:
    i_number_labels_high, arr_markers, data_high, center_high = cv2.connectedComponentsWithStats(img_sure_fg)
    center_high = center_high[1 : (i_number_labels_high + 1), : ]
  else:
    i_number_labels_high, arr_markers = cv2.connectedComponents(img_sure_fg)
    center_high = "None"
#  i_number_labels_high -= 2
#    i_number_of_color_radius = DistincteLabels(img, nLabels, markers)
  if nbShowFlag:
    plt.imshow(arr_markers)
    plt.show()
    plt.clf()
  arr_markers = arr_markers + 1
  arr_markers[img_unknown == 255] = 0
  if nbShowFlag:
    plt.imshow(arr_markers)
    plt.show()
    plt.clf()
  img_opening_closing_bgr = cv2.cvtColor(img_opening_closing, cv2.COLOR_GRAY2BGR)
  markers_watershed = cv2.watershed(img_opening_closing_bgr, arr_markers)


#  img_opening_closing[markers_watershed == -1] = 0
#  img_opening_closing[markers_watershed == 1] = 0
#  nLabels_1, nMarkers_1 = cv2.connectedComponents(img_opening_closing)
#  if nbShowFlag:
#    showPicture(img_opening_closing, 'img_opening_closing')

  
  i_number_of_color_and_radius_high, npdata_number_of_color_and_radius_high = DistincteLabels(nimgRaw, i_number_labels_high, markers_watershed, nstrResultFolderPath)
  nimgRaw[markers_watershed == -1] = [0,255,0]
  img_opening_closing[markers_watershed != 1] = 0
  if nbShowFlag:
    showPicture(img_opening_closing, 'opening_closing_remove')
  b_ret, i_number_labels_low, center_low,  i_number_of_color_and_radius_low, npdata_number_of_color_and_radius_low = doWatershedMethod(nimgRaw, img_opening_closing, nstrResultFolderPath, nbShowFlag, nbDetail, niCount)
  if b_ret:
    i_number_of_color_and_radius_high += i_number_of_color_and_radius_low
    i_number_labels_high += i_number_labels_low
    npdata_number_of_color_and_radius_high = np.append(npdata_number_of_color_and_radius_high, npdata_number_of_color_and_radius_low, axis=0)
    if nbDetail:
      center_high = np.append(center_high, center_low, axis=0)
  if nbShowFlag:
    plt.imshow(markers_watershed)
    plt.show()
    plt.close()
  i_number_labels_high -= 1
  return True, i_number_labels_high, center_high, i_number_of_color_and_radius_high, npdata_number_of_color_and_radius_high

###  使用していない為コメント化  ###########################
#def adjust(nimgSorce, nfAlpha=1.0, nfBeta=0.0):
#    # 積和演算を行う。
#    img_dst = nfAlpha * nimgSorce + nfBeta
#    # [0, 255] でクリップし、uint8 型にする。
#    return np.clip(img_dst, 0, 255).astype(np.uint8)
##########################################################

def calcCircularity(contour, area):
  perimeter = cv2.arcLength(contour, True)
  i_circle_level = (int)((4.0 * np.pi * area / (perimeter * perimeter)) * 100)
  return i_circle_level


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
  #ウィンドウ情報を消去する
  cv2.destroyAllWindows()

def DistincteLabels(nimgSrc, niLabelNumber, nlsLabelTable, nstrResultFolderPath):
  m_dicSettingElements={"NumberPerOneMiliMeter" : "", "MaxRadius" : ""} #設定ファイルから読み込む変数群を格納した辞書型変数 
  cHandleJsonfile.ReadElementsFromJsonfile(os.path.join(nstrResultFolderPath,"Setting.json"), 'mWaterShed', m_dicSettingElements)
  i_number_of_color_and_radius = np.zeros([6, m_dicSettingElements["MaxRadius"]], dtype=np.int) #(赤,黄,緑,青,紫,白)ごとの個数
  npdata_number_of_color_and_radius = np.array([[0, 0]])
  for label in range(2, niLabelNumber + 1):
    i_label_group_index = np.where(nlsLabelTable == label) #現ラベル数のブロブ情報
    i_array_label_bgr = nimgSrc[i_label_group_index] #rgb情報
    i_width = i_array_label_bgr.shape[0] #所属ブロブ個数
    #半径を決定
    i_ret_radius = CalculateRadius(i_width, m_dicSettingElements["MaxRadius"], m_dicSettingElements["NumberPerOneMiliMeter"])
    img_label_bgr = np.zeros((1, i_width, 3), dtype='uint8') #rgb情報を格納するnumpy型変数
    #opencvのメソッドを使用するために一列の長い画像として情報を格納する
    img_label_bgr[0, :, :] = i_array_label_bgr 
    #rgb→hsv変換
    img_hsv = cv2.cvtColor(img_label_bgr, cv2.COLOR_BGR2HSV)
    #hsvそれぞれの平均値を算出
    h, s, v = cv2.split(img_hsv)
    i_h_mean = h.mean()
    i_s_mean = s.mean()
    i_v_mean = v.mean()
#    #デバックに使用する    
#    b, g, r = cv2.split(img_label_bgr)
#    i_b_mean = b.mean()
#    i_g_mean = g.mean()
#    i_r_mean = r.mean()
    #明度が高く彩度が低いブロブを白色とし、その条件外のブロブはhueを使用して色を決定させる
    if i_v_mean > 100 and i_s_mean < 100:
      i_ret_color = 5
    else:
      i_ret_color = DistincteColor(i_h_mean)
    i_number_of_color_and_radius[i_ret_color][i_ret_radius - 1] += 1
    npdata_color_and_radius = np.array([i_ret_color, i_ret_radius - 1])
    npdata_number_of_color_and_radius = np.append(npdata_number_of_color_and_radius, np.array([npdata_color_and_radius]), axis=0)
  npdata_number_of_color_and_radius = npdata_number_of_color_and_radius[1 : len(npdata_number_of_color_and_radius), : ]
  #(色, 半径)それぞれの個数を配列として返す
  return i_number_of_color_and_radius, npdata_number_of_color_and_radius

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

def CalculateRadius(niNamberOfPixels, niMaxRadius, ndNumberPerOneMiliMeter):
  #半径毎の面積ピクセル数を計算し、比較する
  for i_radius in range(niMaxRadius):
    i_area = math.pi * (i_radius + 1) ** 2 * ndNumberPerOneMiliMeter**2
    #計算した面積ピクセル数がブロブのピクセル数よりも大きくなった時にブレイク
    if niNamberOfPixels < i_area:
      break
  #半径を返す
  return i_radius
    

if __name__ == '__main__':
  main(True, "", True)