import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import cHandleJsonfile
import os
import math

m_dicSettingElements={"NumberPerOneMiliMeter" : "", "MaxRadius" : "","circle_level_threshold":"","white_satuation":""} #設定ファイルから読み込む変数群を格納した辞書型変数 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
### メイン関数
def main(nbShowFlag, nstrExeFolderPath, nbDetail, nbLimitFileName):
  if nstrExeFolderPath == "":
    nstrExeFolderPath = "."
  #設定ファイルを読み込む
  cHandleJsonfile.ReadElementsFromJsonfile(os.path.join(nstrExeFolderPath,"Setting.json"), 'mWaterShed', m_dicSettingElements)
  str_file_name = ""
  if nbLimitFileName:
    #img.pngというファイル名のみ解析する
    #アプリに組み込む際は限定しておいた方が余計なファイルに惑わされない
    str_file_name = "img/img.png"
  else:
    #PNGファイルなら何でも解析する
    str_file_name = "img/*.png"
  for fn in glob.glob(os.path.join(nstrExeFolderPath, str_file_name)):
    #画像の読み込み
    img_raw = cv2.imread(fn)
    if nbShowFlag:
      showPicture(img_raw, 'Sorce')
###########  使用する可能性があるためコメントとして残す  ############
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
    b_ret, i_number_labels, arr_center,  i_number_of_color_and_radius, npdata_number_of_color_and_radius = doWatershedMethod(img_raw, img_bind_mask, nstrExeFolderPath, nbShowFlag, nbDetail, i_count)
    if b_ret:
      if nbDetail:
        color_band = ["R","Y","G","B","V","W"]
        for i in range(i_number_labels):
          cv2.putText(img_raw, "ID: " +str(i + 1),                                      ((int)(arr_center[i][0] - 40),(int)(arr_center[i][1] + 30)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
          cv2.putText(img_raw, "Color: " + color_band[npdata_number_of_color_and_radius[i][0]], ((int)(arr_center[i][0] - 40),(int)(arr_center[i][1] + 60)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
          cv2.putText(img_raw, "Size: " +str(npdata_number_of_color_and_radius[i][1]),  ((int)(arr_center[i][0] - 40),(int)(arr_center[i][1] + 90)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
          cv2.imwrite(os.path.join(nstrExeFolderPath, "img/result.png"), img_raw)
      if nbShowFlag:
        showPicture(img_raw, 'img_marked')
    else:
      #0埋めのリストを作成する
      i_number_of_color_and_radius = np.zeros([6, m_dicSettingElements["MaxRadius"]], dtype=np.int)
    np.savetxt(os.path.join(nstrExeFolderPath, "result/color_radius.csv"), i_number_of_color_and_radius, delimiter=",", fmt="%d")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
### WaterShedを行う
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
  markers_watershed[markers_watershed == -1] = 1
  if nbShowFlag:
    plt.imshow(markers_watershed)
    plt.show()
    plt.close()
  #calcCircularity(markers_watershed)
  i_number_of_color_and_radius_high, npdata_number_of_color_and_radius_high, i_number_labels_high, center_high = DistincteLabels(nimgRaw, i_number_labels_high, markers_watershed, nstrResultFolderPath, nbShowFlag, nbDetail, center_high)
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
  i_number_labels_high -= 1
  return True, i_number_labels_high, center_high, i_number_of_color_and_radius_high, npdata_number_of_color_and_radius_high
###  使用していない為コメント化  ###########################
#def adjust(nimgSorce, nfAlpha=1.0, nfBeta=0.0):
#    # 積和演算を行う。
#    img_dst = nfAlpha * nimgSorce + nfBeta
#    # [0, 255] でクリップし、uint8 型にする。
#    return np.clip(img_dst, 0, 255).astype(np.uint8)
##########################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
### 円形度を計算する
def calcCircularity(nlsLabelTable, niLabel, nbShowFlag):
  nlsLabelTableCopy = nlsLabelTable.copy()
  nlsLabelTableCopy[nlsLabelTable != niLabel] = 0
  nlsLabelTableCopy[nlsLabelTable == niLabel] = 1
  if nbShowFlag:
    plt.imshow(nlsLabelTableCopy)
    plt.show()
    plt.clf()
  contour , hierarchy = cv2.findContours(nlsLabelTableCopy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  if len(contour) == 0:
    #円周がとれないのなら0とする
    return 0
  perimeter = cv2.arcLength(contour[0], True)
  area = cv2.contourArea(contour[0])
  i_circle_level = (int)((4.0 * np.pi *area / (perimeter * perimeter) )* 100)
  return i_circle_level
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
### イメージを表示させる
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
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
### 各ラベルを判別させる
def DistincteLabels(nimgSrc, niLabelNumber, nlsLabelTable, nstrResultFolderPath, nbShowFlag, nbDetail, nCenterPoint):
  i_number_of_color_and_radius = np.zeros([6, m_dicSettingElements["MaxRadius"]], dtype=np.int) #(赤,黄,緑,青,紫,白)ごとの個数
  i_label_number = niLabelNumber
  npdata_number_of_color_and_radius = np.array([[0, 0]])
  for label in range(2, niLabelNumber + 1):
    i_circle_level = calcCircularity(nlsLabelTable, label, nbShowFlag)
    if (i_circle_level >= m_dicSettingElements["circle_level_threshold"]):
      i_label_group_index = np.where(nlsLabelTable == label) #現ラベル数のブロブ情報
      i_array_label_bgr = nimgSrc[i_label_group_index] #rgb情報
      i_width = i_array_label_bgr.shape[0] #所属ブロブ個数
      #半径を決定
      i_ret_radius = CalculateRadius(i_width, m_dicSettingElements["MaxRadius"], m_dicSettingElements["NumberPerOneMiliMeter"])
      #メディアンフィルター
      img_label_bgr_median = cv2.medianBlur(i_array_label_bgr,5)
      img_label_bgr = np.zeros((1, i_width, 3), dtype='uint8') #rgb情報を格納するnumpy型変数
      #opencvのメソッドを使用するために一列の長い画像として情報を格納する
      img_label_bgr[0, :, :] = img_label_bgr_median 
      #rgb→hsv変換
      img_hsv = cv2.cvtColor(img_label_bgr, cv2.COLOR_BGR2HSV)
      #hsvそれぞれの平均値を算出
      hsv_list =  list(cv2.split(img_hsv))
      #h, s, v = cv2.split(img_hsv)
      #i_s_mean = hsv_list[1].mean()
      #i_v_mean = hsv_list[2].mean()
      hsv_without_white = [hsv_list[0][0][i] for i in range(hsv_list[0].shape[1]) if (hsv_list[1][0][i] > m_dicSettingElements["white_satuation"])]
#      #デバックに使用する    
#      b, g, r = cv2.split(img_label_bgr)
#      i_b_mean = b.mean()
#      i_g_mean = g.mean()
#      i_r_mean = r.mean()
      #明度が高く・彩度が低いブロブを白色とし、その条件外のブロブはhueを使用して色を決定させる
      white_ratio = (hsv_list[0].shape[1]-len(hsv_without_white)) / hsv_list[0].shape[1]
      if white_ratio > 0.9:
        i_ret_color = 5
      else:
        i_h_mean = sum(hsv_without_white) / len(hsv_without_white)
        i_ret_color = DistincteColor(i_h_mean)
      i_number_of_color_and_radius[i_ret_color][i_ret_radius - 1] += 1
      npdata_color_and_radius = np.array([i_ret_color, i_ret_radius - 1])
      npdata_number_of_color_and_radius = np.append(npdata_number_of_color_and_radius, np.array([npdata_color_and_radius]), axis=0)
    else:
      if nbDetail:
        nCenterPoint = np.delete(nCenterPoint, label - 2 - (niLabelNumber - i_label_number), axis = 0)
      i_label_number -= 1
  npdata_number_of_color_and_radius = npdata_number_of_color_and_radius[1 : len(npdata_number_of_color_and_radius), : ]
  #(色, 半径)それぞれの個数を配列として返す
  return i_number_of_color_and_radius, npdata_number_of_color_and_radius, i_label_number, nCenterPoint
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
### 色を判別させる
def DistincteColor(niHue):
  #hueの値を(赤,黄,緑,青,紫)に区分けする為の区切り数値
  i_color_band = (10,50,100,140,160)
  #色判定：赤
  if i_color_band[0] > niHue or niHue >= i_color_band[4]:
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
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
### 半径を測定する
def CalculateRadius(niNamberOfPixels, niMaxRadius, ndNumberPerOneMiliMeter):
  #半径毎の面積ピクセル数を計算し、比較する
  for i_radius in range(niMaxRadius):
    i_area = math.pi * (i_radius + 1) ** 2 * ndNumberPerOneMiliMeter**2
    #計算した面積ピクセル数がブロブのピクセル数よりも大きくなった時にブレイク
    if niNamberOfPixels < i_area:
      break
  #半径を返す
  return i_radius
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



if __name__ == '__main__':
  main(True, "", True, False)