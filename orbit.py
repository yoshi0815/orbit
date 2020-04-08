# -*- coding: utf-8 -*-
import cv2
import numpy as np

#読み込む動画を指定
cap = cv2.VideoCapture('602004501.077949.mp4')

#読み込む動画の画面サイズとVideoWriter()内に記述する画面サイズを揃える
Width = int(cap.get(3))
Height = int(cap.get(4))

#動画出力設定（拡張子は.avi）
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi',fourcc, 23.0,(Width,Height))

orbit = []

#マスク画像取得
while(1):
    _, frame = cap.read()

    def getMask(l, u):
#HSVに変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array(l)
        upper = np.array(u)
        if lower[0] >= 0:
            mask = cv2.inRange(hsv, lower, upper)
        else:
            #赤用(彩度、明確判定は簡略化)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            mask = np.zeros(h.shape, dtype=np.uint8)
            mask[((h < lower[0]*-1) | (h > upper[0])) & (s > lower[1])] = 255
        
        return cv2.bitwise_and(frame,frame, mask= mask)
#輪郭取得
    def getContours(img,t,r,drawOrbit):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        imgEdge, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#輪郭描画
        cv2.drawContours(frame, contours, -1, (0,255,0),1)
        cv2.imshow('rinkaku', frame)
#一番大きい輪郭を抽出
        contours.sort(key=cv2.contourArea, reverse=True)
        cnt = contours[0]
#円描画
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        if radius > r:
            if drawOrbit == 1:
                orbit.append(center)
                for j in orbit:
                    cv2.circle(frame,j,10,(10,0,55),-1)# 軌道描画(真ん中３つの数値いじれば色変わるよ)
            return cv2.circle(frame,center,radius,(0,255,0),2)# 最小外接円描画
            
        else:
            return frame
        
#白、赤マスク
    res_white = getMask([0,0,100], [180,45,255])
    res_red = getMask([-10,45,30], [170,255,255])
#輪郭取得
    getContours(res_white, 45, 75, 0)# (画像, 明度閾値, 最小半径, 軌道描画 0=なし 1=あり)
    contours_frame = getContours(res_red, 45, 75, 1)
#動画書き込み
    out.write(contours_frame)
#再生
    cv2.imshow('video',contours_frame)
    k = cv2.waitKey(46) & 0xFF
#Qで終了
    if k == ord('q'):
        break

out.release()
cv2.destroyALLWindows()
