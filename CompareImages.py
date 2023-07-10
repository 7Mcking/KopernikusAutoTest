# Created by mcking at 10.07.23
from deleteDuplicates import compare2TestImages, compareDifferentCountourAreaValues, stackImages
import cv2

if __name__ == '__main__':

    imagePath1 = './dataset/c10-1623871098865.png'
    imagePath2 = './dataset/c10-1623872887821.png'

    minContourAreaList = 500
    gaussianBlurRadius = [5,5]
    completeStack = compare2TestImages(imagePath1, imagePath2, minContourAreaList, gaussianBlurRadius)
    cv2.imwrite("StackedImages.png", completeStack)

    cv2.imshow("Stacked images", completeStack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


