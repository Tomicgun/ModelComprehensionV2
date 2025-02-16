import cv2
import easyocr
import matplotlib.pyplot as plt
if __name__ == '__main__':
    img = cv2.imread('Test2.png')

    reader = easyocr.Reader(['en'],gpu=False)

    text = reader.readtext(img)

    for t in text:
        bbox, string, score = t

        try:
            cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        except:
            print(t)

    plt.imshow(img)
    plt.show()
