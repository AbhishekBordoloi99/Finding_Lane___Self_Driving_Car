import cv2
import numpy as np
#import matplotlib.pyplot as plt
def canny(image):
    #image=cv2.imread(image)
    gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('Grayscaled.jpg', gray)
    blur=cv2.GaussianBlur(gray, (5,5), 0)
    outline_img= cv2.Canny(blur, 50,100)
    #cv2.imwrite('Canny.jpg', outline_img)
    return outline_img

def focus_on(image):
    triangle=np.array([[(200,image.shape[0]),(1100,image.shape[0]),(550,250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image=cv2.bitwise_and(mask, image)
    cv2.imwrite('Cropped_image.jpg',masked_image)
    return masked_image
    
def line_generate(image, lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (0,255,0),10)
    return line_image 



vid=cv2.VideoCapture('test2.mp4')
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('Video.mp4', fourcc, 20, (240,240))
while vid.isOpened():
    ret,capture= vid.read()
    #image = cv2.imread(capture)
    load_image=np.copy(capture)
    canny_image= canny(capture)
    cropped_image=focus_on(canny_image)
    lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image=line_generate(capture, lines)
    final_image=cv2.addWeighted(capture, 0.8, line_image, 1,1)
    
    cv2.imwrite('Road_line.jpg', final_image)
    cv2.imshow('Region', final_image)
    if cv2.waitKey(1)== ord('C'):
        break

cv2.release()
cv2.destroyAllWindows()
