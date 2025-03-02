import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Create MSER object
    mser = cv2.MSER.create()

    # Your image path i-e receipt path
    img = cv2.imread('Test4.png')

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vis = img.copy()

    # detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.polylines(vis, hulls, 1, (0, 255, 0))


    cv2.imshow('MSER', vis)
    cv2.waitKey(0)
    #plt.imshow(vis)
    #plt.show()