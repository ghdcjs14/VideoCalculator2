#!/usr/bin/etc python

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import subprocess
from deep_convnet import DeepConvNet

# font
FONT = cv2.FONT_HERSHEY_PLAIN

calculation = '+'

class Button(object):

    def __init__(self, text, x, y, width, height, command=None):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.left = x
        self.top = y
        self.right = x + width - 1
        self.bottom = y + height - 1

        self.hover = False
        self.clicked = False
        self.cntClick = 0
        self.command = command

    def handle_event(self, event, x, y, flags, param):
        self.hover = (self.left <= x <= self.right and \
                      self.top <= y <= self.bottom)

        if self.hover and event == cv2.EVENT_LBUTTONUP:
            self.clicked = False
            #print(event, x, y, flags, param)
            self.cntClick += 1
            #print(self.cntClick)

            if self.command:
                self.command()

    def draw(self, frame):
        status = self.cntClick % 4
        if status == 0 :
            calculation = '+'
            cv2.putText(frame, calculation, (40, 40), FONT, 3, (0, 0, 255), 2)
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)
        elif status == 1:
            calculation = '-'
            cv2.putText(frame, calculation, (40, 40), FONT, 3, (0, 255, 0), 2)
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)
        elif status == 2:
            calculation = '*'
            cv2.putText(frame, calculation, (40, 40), FONT, 3, (0, 255, 0), 2)
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)
        elif status == 3:
            calculation = '%'
            cv2.putText(frame, calculation, (40, 40), FONT, 3, (0, 255, 0), 2)
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)

        return calculation

class Recognition:

     boxes = []
     numStr = []


     def doPreprocessing(self, imgSrc):
         # 전처리 과정1: Grayscale -> Erosion -> Resize -> Binary
         # 전처리 과정2
         # Grayscale ->  Morph Gradient -> Adaptive Threshold -> Morph Close -> HoughLinesP

         # 1. Grayscale
         grayImg = cv2.imread(imgSrc, cv2.IMREAD_GRAYSCALE)
         cv2.imwrite('image_result/01GrayImage.jpg', grayImg)

         # 2. Erosion
         kernel = np.ones((5,5),np.uint8)
         erosion = cv2.erode(grayImg,kernel,iterations=1)
         cv2.imwrite('image_result/02Erosion.jpg', erosion)

         # 2. Morph Gradient - Naver
         gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)
         cv2.imwrite('image_result/02Gradient.jpg', gradient)

         # 3. Resize
         #cv2.resize(erosion, (28, 28))
         #cv2.imwrite('image_result/03Resize.jpg', erosion)

         # 4. Binary
         #ret, thresh = cv2.threshold(erosion, 90, 255, 0)
         #cv2.imwrite('image_result/03Thresh.jpg', thresh)

         # 4. Adaptive Threshold : 잡영 제거
         thresh = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
         cv2.imwrite('image_result/03Thresh.jpg', thresh)

         # 4. Morph Close : 작은 구멍을 메우고 경계를 강화
         kernel = np.ones((10, 10), np.uint8)
         closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
         cv2.imwrite('image_result/04Closing.jpg', closing)

         # 5. Long Line Remove(HoughLinesP() 사용) : 글씨 추출에 방해 되는 요소 제거
         threshold = 100  # 선 추출 정확도
         minLength = 80  # 추출할 선의 길이
         lineGap = 5  # 5픽셀 이내로 겹치는 선은 제외
         rho = 1

         #lines = cv2.HoughLinesP(closing, rho, np.pi / 180, threshold, minLength, lineGap)

         #limit = 10
         #if lines != None:
         #    for line in lines:
         #        gapY = np.abs(line[0][3] - line[0][1])
         #        gapX = np.abs(line[0][2] - line[0][0])
         #        if gapY > limit & limit > 0:
         #            # remove line
         #            cv2.line(closing, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 10)
         #    cv2.imwrite('image_result/05HoughLinesP.jpg', closing)

         # 6. edge 검출 알고리즘 적용
         padding = 2
         image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         cv2.drawContours(image, contours, -1, 255, 3)
         if len(contours) > 0:
             for contour in contours:
                 x, y, w, h = cv2.boundingRect(contour)

                 if w>18 and h>18 and not (w>100 and h>100) :
                     self.boxes.append([cv2.boundingRect(contour),cv2.resize(thresh[y:y + h, x:x + w], (28, 28))])
                     #self.boxes.append(cv2.resize(thresh[y:y + h, x:x + w], (28, 28)))
                     cv2.rectangle(grayImg, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0),
                                   1)

                 #cnt = contours[contour]
                 #contousImg = cv2.drawContours(grayImg, contour, -1, (50, 20, 0), 2)


         cv2.imwrite('image_result/04Contours.jpg', grayImg)
         ##Buble Sort on python
         for i in range(len(self.boxes)):
             for j in range(len(self.boxes) - (i + 1)):
                 if self.boxes[j][0][0] > self.boxes[j + 1][0][0]:
                     temp = self.boxes[j]
                     self.boxes[j] = self.boxes[j + 1]
                     self.boxes[j + 1] = temp

         return (Image.open('image_result/04Contours.jpg'))


     def predictBoxes(self):

         network = DeepConvNet()
         network.load_params("deep_convnet_params.pkl")

         # show boxes...
         for box in self.boxes:
             # print(box)
             # print(box.shape)
             npbox = np.array([[box[1]]])
             y = network.predict(npbox)
             #print(np.argmax(y, axis=1))
             self.numStr += str(np.argmax(y, axis=1))
             #cv2.imshow('box', box[1])
             #cv2.waitKey(0)

         #print(self.numStr)

     def recognizeVideo(self):

         network = DeepConvNet()
         network.load_params("deep_convnet_params.pkl")

         # create button instance
         button = Button('QUIT', 0, 0, 100, 30)

         # 전처리 과정1: Grayscale -> Erosion -> Resize -> Binary
         # 전처리 과정2
         # Grayscale ->  Morph Gradient -> Adaptive Threshold -> Morph Close -> HoughLinesP

         capture = cv2.VideoCapture(0)
         capture.set(3, 640)
         capture.set(4, 480)
         print('image width %d' % capture.get(3))
         print('image height %d' % capture.get(4))

         while (1):
             ret, frame = capture.read()

             # 1. GRAY Image로 변경
             grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

             # 2. Erosion
             kernel = np.ones((5, 5), np.uint8)
             erosion = cv2.erode(grayFrame, kernel, iterations=1)

             # 3. Morph Gradient : 경계 이미지 추출
             kernel = np.ones((5, 5), np.uint8)
             gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)

             # 4. Adaptive Threshold : 잡영 제거
             thresh = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)

             # 5. Morph Close : 작은 구멍을 메우고 경계를 강화
             kernel = np.ones((10, 10), np.uint8)
             closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

             # 6. Contour 추출
             # contours는 point의 list형태. 예제에서는 사각형이 하나의 contours line을 구성하기 때문에 len(contours) = 1. 값은 사각형의 꼭지점 좌표.
             # hierachy는 contours line의 계층 구조
             contourFrame, contours, hierachy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

             if len(contours) > 0:
                 for contour in contours:
                     x, y, w, h = cv2.boundingRect(contour)
                     #print(w, h)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                     self.boxes.append([cv2.boundingRect(contour), cv2.resize(thresh[y:y + h, x:x + w], (28, 28))])
                     #if h > 10 & w > 10:
                         #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

             ##Buble Sort on python
             for i in range(len(self.boxes)):
                 for j in range(len(self.boxes) - (i + 1)):
                     if self.boxes[j][0][0] > self.boxes[j + 1][0][0]:
                         temp = self.boxes[j]
                         self.boxes[j] = self.boxes[j + 1]
                         self.boxes[j + 1] = temp

             # show boxes...
             for box in self.boxes:
                 # print(box)
                 #  print(box.shape)
                 npbox = np.array([[box[1]]])
                 y = network.predict(npbox)
                 # print(np.argmax(y, axis=1))
                 self.numStr.append(np.argmax(y, axis=1))
                 cv2.putText(frame, str(np.argmax(y, axis=1)),
                             (box[0][0] + box[0][2]%2, box[0][1] + box[0][3] *2), # text 출력 위치
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

             # add button to frame
             calculation = button.draw(frame)

             if len(self.numStr) >= 2:
                 cv2.putText(frame, str(eval(str(self.numStr[0][0])
                                              + calculation + str(self.numStr[1][0])))
                             , (100,50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

             #print(self.numStr)

             cv2.imshow('VideoCalculator', frame)

             # assign mouse click to method in button instance
             cv2.setMouseCallback("VideoCalculator", button.handle_event)

             self.boxes = []
             self.numStr = []

             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break;

         capture.release()
         cv2.destroyAllWindows()



     def OrganizeImage(self,img_src):

          img=cv2.imread(img_src,cv2.IMREAD_COLOR)

          # convert to gray image
          convert_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          #cv2.imwrite('image_result/gray.jpg', convert_img)

          # 노이즈 제거를 위한 스무딩 작업
          blur = cv2.GaussianBlur(convert_img,(1,1),0)
          #cv2.imwrite('image_result/blur.jpg',blur)

          canny_img=cv2.Canny(blur,0,255)
          #cv2.imwrite('image_result/canny.jpg',canny_img)

          #edge 검출 알고리즘 적용
          cnts, contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          box1=[]
          box2 = []

          # 이미지를 다듬는 전처리 과정 나누기
          for i in range(len(contours)):
               cnt=contours[i]
               area = cv2.contourArea(cnt)
               x,y,w,h = cv2.boundingRect(cnt)

               rect_area=w*h  #area size
               aspect_ratio = float(w)/h # ratio = width/height
               if(aspect_ratio>=2.0) and (rect_area>=1000):
                    continue;
               elif(x==0) or (y >= 900):
                    continue;
               else:
                   #print(img[y:y+h, x:x+w])
                   box2.append(cv2.resize(convert_img[y:y+h, x:x+w],(28,28)))
                    #print(box2)
                    #print(img[y:y+h, x:x+w])
                    #cv2.imshow('adf', img[y:y+h, x:x+w])
                    #cv2.waitKey(0)
                   box1.append(cv2.boundingRect(cnt))

          for i in range(len(box1)): ##Buble Sort on python
               for j in range(len(box1)-(i+1)):
                    if box1[j][0]>box1[j+1][0]:
                         temp=box1[j]
                         temp2=box2[j]

                         box1[j]=box1[j+1]
                         box2[j]=box2[j+1]

                         box1[j+1]=temp
                         box2[j+1]=temp2

          #to find number measuring length between rectangles
          for m in range(len(box1)):
               count=0
               for n in range(m+1,(len(box1)-1)):
                    delta_x=abs(box1[n+1][0]-box1[m][0])
                    if delta_x > 150:
                         break
                    delta_y =abs(box1[n+1][1]-box1[m][1])
                    if delta_x ==0:
                         delta_x=1
                    if delta_y ==0:
                         delta_y=1
                    gradient =float(delta_y) / float(delta_x)
                    if gradient < 0.25:
                         count+=1

          cv2.imwrite('image_result/snake.jpg',img) #가장 자리 처리 완료

          network = DeepConvNet()
          network.load_params("deep_convnet_params.pkl")
          #print(box1)
          nine = cv2.imread('images/9.jpg', cv2.IMREAD_GRAYSCALE)
          nine = cv2.resize(nine,(28,28))
          #print(nine.shape)
          #cv2.imwrite('image_result/nine.jpg', nine)
          nine = np.array([[nine]])
          #lalal = np.array(box2)
          #print(ndarr.shape)
          #print(nine.shape)


          for box in box2 :
              #print(box)
              #print(box.shape)
              npbox = np.array([[box]])
              y = network.predict(npbox)
              print(np.argmax(y, axis=1))
              cv2.imshow('adf',box)
              cv2.waitKey(0)

          #우리 프로젝트의 경우 적정 길이는 아직 미정
          ret, number_th = cv2.threshold(convert_img,100,255,cv2.THRESH_BINARY)
          cv2.imwrite('image_result/number_th.jpg',number_th)
          kernel = np.ones((3,3))
          er_number = cv2.erode(number_th,kernel,iterations=2)
          cv2.imwrite('image_result/er_number.jpg', er_number)

          return (Image.open('image_result/er_number.jpg'))

     def ExtractNumber(self,img_src):
          #결과 텍스트 파일을 열어 읽습니다.
          txt_result = pytesseract.image_to_string((Image.open(img_src)))
          print('txt_result: '+txt_result)
          f_number = open("image_result2.txt",'w', encoding='UTF-8', newline='')
          f_number.write(txt_result.replace(" ", ""))
          f_number.close()
          return (txt_result)


recogtest=Recognition()
recogtest.recognizeVideo()
#recogtest.doPreprocessing('images/sample03.png')
#recogtest.predictBoxes()
#recogtest.OrganizeImage('images/photo_1.jpg')
#result=recogtest.ExtractNumber('images/digits3.jpg')
#print(result)