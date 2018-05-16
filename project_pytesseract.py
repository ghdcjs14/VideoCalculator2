#!/usr/bin/etc python

import cv2
import numpy as np
import tesserocr as tr
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
     tssStr = []
     cnnStr = []
     tssBoxes = []

     def doTesserectOCR(self, frame):
         api = tr.PyTessBaseAPI()

         # since tesserocr accepts PIL images, converting opencv image to pil
         pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
         # set pil image for ocr
         api.SetImage(pil_img)
         api.Recognize()
         iterator = api.GetIterator()
         # for w in tr.iterate_level(iterator, tr.RIL.TEXTLINE):
         #     print("D = "+ w.GetUTF8Text(tr.RIL.WORD))

         # dictOCR = {}
         level = tr.RIL.TEXTLINE
         for r in tr.iterate_level(iterator, level):
             try:
                 text = r.GetUTF8Text(level)
                 text = text.replace(" ", "");
                 text = text.replace("\n", "");
                 self.tssStr = text
                 left, top, right, bottom = r.BoundingBox(level)
                 # dictOCR[index] = left, top, right, bottom, text
                 if text != '':
                     print(text)
                     cv2.rectangle(frame, (left, top), (right, bottom), color=(0, 0, 255))
                     cv2.putText(frame, str(eval(text)), (right, top), cv2.FONT_HERSHEY_PLAIN, 3.0, color=(0, 0, 255))
                     # cv2.waitKey(2000)
             except BaseException as e:
                 print(e)
                 continue
         return frame


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

             # val1 rectangle
             cv2.rectangle(frame, (40, 140), (40 + 250, 140 + 80), (0, 0, 255), 3)
             va1 = frame[140:140 + 80, 140:40 + 250]


             # val2 rectangle
             cv2.rectangle(frame, (390, 140), (390 + 250, 140 + 80), (0, 0, 255), 3)
             #tesserocrFrame = self.doTesserectOCR(frame)

             # 1. GRAY Image로 변경
             grayFrame = cv2.cvtColor(va1, cv2.COLOR_BGR2GRAY)

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

             # 5. Long Line Remove(HoughLinesP() 사용) : 글씨 추출에 방해 되는 요소 제거
             threshold = 100  # 선 추출 정확도
             minLength = 80  # 추출할 선의 길이
             lineGap = 5  # 5픽셀 이내로 겹치는 선은 제외
             rho = 1

             lines = cv2.HoughLinesP(closing, rho, np.pi / 180, threshold, minLength, lineGap)

             limit = 10
             if lines is not None:
                 for line in lines:
                     gapY = np.abs(line[0][3] - line[0][1])
                     gapX = np.abs(line[0][2] - line[0][0])
                     if gapY > limit and limit > 0:
                         # remove line
                         cv2.line(closing, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 10)

             # 6. Contour 추출
             # contours는 point의 list형태. 예제에서는 사각형이 하나의 contours line을 구성하기 때문에 len(contours) = 1. 값은 사각형의 꼭지점 좌표.
             # hierachy는 contours line의 계층 구조
             contourFrame, contours, hierachy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

             padding = 5
             if len(contours) > 0:
                 for contour in contours:
                     x, y, w, h = cv2.boundingRect(contour)
                     #print(w, h)
                     #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

                     if h > 30 and w > 10:
                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                         self.boxes.append([cv2.boundingRect(contour), cv2.resize(thresh[y:y + h + padding, x:x + w + padding], (28, 28))])

             ##Buble Sort on python
             for i in range(len(self.boxes)):
                 for j in range(len(self.boxes) - (i + 1)):

                     # x 값으로 비교
                     if self.boxes[j][0][0] > self.boxes[j + 1][0][0]:
                         temp = self.boxes[j]
                         self.boxes[j] = self.boxes[j + 1]
                         self.boxes[j + 1] = temp

             # show boxes...
             for box in self.boxes:
                 npbox = np.array([[box[1]]])
                 y = network.predict(npbox)
                 #print(np.argmax(y, axis=1))
                 self.cnnStr.append(np.argmax(y, axis=1))
                 cv2.putText(frame, str(np.argmax(y, axis=1)),
                             (box[0][0] + box[0][2]%2, box[0][1] + box[0][3] *2), # text 출력 위치
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

             print(self.cnnStr)

             # add button to frame
             calculation = button.draw(frame)

             if len(self.cnnStr) >= 2:

                 cv2.putText(frame, str(eval(str(self.cnnStr[0][0])
                                              + calculation + str(self.cnnStr[1][0])))
                             , (100,50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)



             cv2.imshow('VideoCalculator', frame)

             # assign mouse click to method in button instance
             cv2.setMouseCallback("VideoCalculator", button.handle_event)

             self.boxes = []
             self.cnnStr = []

             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break;

         capture.release()
         cv2.destroyAllWindows()


recogtest=Recognition()
recogtest.recognizeVideo()
