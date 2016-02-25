import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import RPi.GPIO as GPIO
import time,sys,os
GPIO.setmode(GPIO.BCM)
GPIO.setup(14,GPIO.OUT, initial = 0)
GPIO.setup(15,GPIO.OUT, initial = 0)
GPIO.setup(2,GPIO.OUT, initial = 0)
GPIO.setup(3,GPIO.OUT, initial = 0)
#import psychopy
#import libardrone as ar
##from matplotlib import pyplot as plt

##ar.set_camera_view(self, True)
imgsize = (150,150)
camera = PiCamera()
rawCapture = PiRGBArray(camera)
camera.resolution = imgsize
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=imgsize)

time.sleep(0.1)



##cap = cv2.VideoCapture(0)
##img = cv2.imread('iki.jpeg',0)
##imgc = cv2.imread('iki.jpeg',1)
"""
imgc = cv2.medianBlur(img,5)

imgc = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

img = cv2.medianBlur(img,5)

img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
"""
#titles = ['Adaptive Gaussian Thresholding']
#images = [th3]

def slope():
	img = cv2.imread("frame.jpeg",0)
	imgc = cv2.imread("frame.jpeg",1)

        turntresh = 10
        
	slope=0
	h = np.size(img, 1)
	w = np.size(img, 0)
	movetresh = w/10
	x = 20
	y=20
##	r = 0
	threshold = 50
	xinc = 6
	yinc = 3
	points = []

	while(x<w):
		if(img[x,y]>threshold and y<h-15):
			y+=xinc
			#r+=1
			#cv2.circle(imgc,(y,x),1, (r,0,0), -1)
		elif(img[x,y]<=threshold):
			#r=0
##			cv2.circle(imgc,(y,x), 1, (0,0,255), -1)
			points.append(( y,x))
			x+=yinc
			y=2
		else:

			#cv2.circle(img,(x,y), 1, (0,0,0), -1)
			#print "("+str(x)+","+str(y)+")"
			y=2
			x+=yinc
	
	if points != []:
                if (points[0][0]-points[-1][0]+0.0)!=0:
                        slope = -(points[0][1]-points[-1][1]+0.0)/(points[0][0]-points[-1][0]+0.0)
                        #print slope
                degree = np.rad2deg(np.arctan(slope))
                total = 0
                for i in points:
                        total+=i[0]
                avarage = total/len(points)
                #print avarage
                cv2.circle(imgc,(avarage,100), 10, (0,255,0), -1)
                #print degree
                print "degree: "+str(degree)+" pos: "+ str(avarage)
                if  degree < 0 and degree > -90+turntresh:
                    GPIO.output(14,1)
                    GPIO.output(15,0)
                    print 'sag'
                elif degree > 0 and degree < 90-turntresh:
                    GPIO.output(15,1)
                    GPIO.output(14,0)
                    print 'sol'
                else:
                    GPIO.output(14,0)
                    GPIO.output(15,0)
                    print 'duz'
                    '''
                if  avarage < w/2-movetresh:
                    GPIO.output(3,1)
##sa[a sola kayma koduna devam et                    
                    print "sola kay"
                else:
                    GPIO.output(3,0)
                    print "saga kay"
                ''' 
	#cv2.imshow("image",img)
	cv2.imshow("imagec",imgc)

while(True):
    # Capture frame-by-frame
##    ret, frame = cap.read()
    GPIO.output(2,1)
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
 
        cv2.imwrite("frame.jpeg",image)
        slope()
    # Our operations on the frame come here

    # Display the resulting frame
    #cv2.imshow('frame',frame)

        rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	


        if cv2.waitKey(1) & 0xFF == ord('q'):
##                cap.release()
                GPIO.output(2,0)
                cv2.destroyAllWindows()
                sys.exit('yeeee')
                break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


