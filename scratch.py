import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Read image
    img = cv2.imread("Test4.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Modify the parameters of MSER
    mser = cv2.MSER.create(delta=5, min_area=10, max_area=200, max_variation=0.5, min_diversity=0.1)
    # Detect regions
    regions, _ = mser.detectRegions(gray)
    # Filter regions by area
    filtered_regions = [p for p in regions if len(p) > 10]
    # Draw the regions on the image
    for p in filtered_regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img, (x,y), 2, (0, 0, 255), 2)
    # Display the result

    cv2.imshow('MSER', img)
    cv2.waitKey(0)
    #plt.imshow(img)
    #plt.show()
