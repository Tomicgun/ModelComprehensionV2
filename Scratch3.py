import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Load image, grayscale, Gaussian blur, adaptive threshold
    image = cv2.imread('Test5.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cv2.circle(image, (x + w // 2, y + h // 2), 10, (255, 0, 0), 2)

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('dilate', dilate)
    plt.imshow(image)
    plt.show()
    #cv2.imshow('image', image)
    #cv2.waitKey()