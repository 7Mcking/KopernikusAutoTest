# Created by mcking at 10.07.23
from deleteDuplicates import compare2TestImages, compareDifferentCountourAreaValues, stackImages
import cv2

if __name__ == '__main__':
    imagePath1 = '/Users/mcking/PycharmProjects/Kopernikus_Auto/dataset/c10-1623873575337.png'
    imagePath2 = '/Users/mcking/PycharmProjects/Kopernikus_Auto/dataset/c10-1623899361010.png'

    minContourAreaList = 500
    gaussianBlurRadius = [5,5]
    completeStack = compare2TestImages(imagePath1, imagePath2, minContourAreaList, gaussianBlurRadius)

    cv2.imshow("Stacked images", completeStack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


