import imutils
import cv2
import numpy as np
import os 
from utils.helping_func import order_points
from utils.helping_func import four_point_transform
from imutils import contours



#Step #1: Detect the exam in an image.
#Step #2: Apply a perspective transform to extract the top-down, birds-eye-view of the exam.
#Step #3: Extract the set of bubbles (i.e., the possible answer choices) from the perspective transformed exam.
#Step #4: Sort the questions/bubbles into rows.
#Step #5: Determine the marked (i.e., “bubbled in”) answer for each row.
#Step #6: Lookup the correct answer in our answer key to determine if the user was correct in their choice.
#Step #7: Repeat for all questions in the exam.

def run(image):
    #read image 
    image = cv2.imread(image)
    orig = image.copy()
    
    #Answer key dict
    ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
    
    #--------------------------------------------Step #1----------------------------------
    print('[INFO] Image Preprocessing started')
    
    #covert too grayscale
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    
    #Apply gaussian blur 
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    #Detecting the edges in the image
    edged = cv2.Canny(blur, 5, 100)
    
    print('[INFO] Image Preprocessing completed')
    
    print('[INFO] Bubble Sheet Detection Started ')
    #find contour and other elements
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #grab the contours
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) > 0:
        #sort out the top 5 biggest contour
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        print('[INFO] Number of Sheet Detected:', len(cnts))
        
        #Confirm that detected contour is rectangle in shape through the having four edges 
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            #if their are 4 edges then required rectangle 
            if len(approx) == 4:
                page_contor = approx 
                break
    
    
    #--------------------------------------------Step #2----------------------------------
    
    #get the coordinateds of rectangle for perceptive change
    pts = order_points(page_contor.reshape(4,2))
    
    #apply transformation
    warped = four_point_transform(orig, pts)
    gray_warped = four_point_transform(gray, pts)
    
    print('[INFO] Perpective Changed Applied')
    
    #--------------------------------------------Step #3----------------------------------
    
    
    #apply thresholding to convert the bubble options as white while the rest as black 
    thresh = cv2.threshold(gray_warped, 0, 250, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    
    #find contour and other elements
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #grab the contours
    cnts = imutils.grab_contours(cnts)
    
    bubble_doc = []
    
    for cms in cnts:
    	# compute the bounding box of the contour, then use the
        #bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(cms)
        asp= w / float(h)
        
        #for an object to detected as bubble it should have sufficent height and as circular this radius 
        #constant in all direction. thus equal height and width. Similar reason behind the asp be near to 1
        
        if (w >= 20 and h >= 20) and (asp >= 0.9 and  asp <= 1.1):
            bubble_doc.append(cms)
            
    print('[INFO] All the Bubbles are Detected')
    
    #--------------------------------------------Step #4----------------------------------
    
    # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    questionCnts = contours.sort_contours(bubble_doc,
    	method="top-to-bottom")[0]
    correct = 0
    
    
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    	# sort the contours for the current question from
    	# left to right, then initialize the index of the
    	# bubbled answer
    	cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    	bubbled = None 
    #--------------------------------------------Step #5----------------------------------
    	for (j, c) in enumerate(cnts):
    		# construct a mask that reveals only the current
    		# "bubble" for the question
    		mask = np.zeros(thresh.shape, dtype="uint8")
    		cv2.drawContours(mask, [c], -1, 255, -1)
    		# apply the mask to the thresholded image, then
    		# count the number of non-zero pixels in the
    		# bubble area
    		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    		total = cv2.countNonZero(mask)
    		# if the current total has a larger number of total
    		# non-zero pixels, then we are examining the currently
    		# bubbled-in answer
    		if bubbled is None or total > bubbled[0]:
    			bubbled = (total, j)
                
    #--------------------------------------------Step #6----------------------------------
    	# initialize the contour color and the index of the
    	# *correct* answer
    	color = (0, 0, 255)
    	k = ANSWER_KEY[q]
    	# check to see if the bubbled answer is correct
    	if k == bubbled[1]:
    		color = (0, 255, 0)
    		correct += 1
    	# draw the outline of the correct answer on the test
    	cv2.drawContours(warped, [cnts[k]], -1, color, 3)
        
    # Print the results 
    score = (correct / 5.0) * 100
    print("[INFO] score: {:.2f}%".format(score))
    cv2.putText(warped, "{:.2f}%".format(score), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    print('[INFO] Bubble Sheets is Marked')
    
    return image, warped



if __name__ == "__main__":
    dir_path = 'input_imgs'
    for i in os.listdir(dir_path):
        image = os.path.join(dir_path, i)
        print('Scoring Started for: ',image)
        orignal, warped = run(image)
        cv2.imshow("Original", orignal)
        cv2.imshow("Exam", warped)
        cv2.waitKey(0)

