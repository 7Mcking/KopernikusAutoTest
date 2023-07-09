# Created by mcking at 09.07.23
from imaging_interview import *
import cv2
from imutils import paths
import os
import numpy as np
import pickle


def compare2TestImages(imgPath1: str, imgPath2: str, minContourArea: int) \
        -> np.ndarray:
    """
    Compare two images and return their comparison scores along with their
    times

    :param imgPath1: path to the first image
    :param imgPath2: path to the second image
    :return: np.ndarray
    """
    imagePath1 = imgPath1
    imagePath2 = imgPath2

    img1 = cv2.imread(imagePath1)
    img2 = cv2.imread(imagePath2)

    gray1 = preprocess_image_change_detection(img1, [3, 3])
    gray2 = preprocess_image_change_detection(img2, [3, 3])

    resizeDim = (240, 240)

    if gray1.shape != resizeDim:
        gray1 = cv2.resize(gray1, resizeDim, interpolation=cv2.INTER_AREA)
    if gray2.shape != resizeDim:
        gray2 = cv2.resize(gray2, resizeDim, interpolation=cv2.INTER_AREA)

    score, _, _ = compare_frames_change_detection(gray1, gray2, minContourArea)
    print(score)

    allStacked = stackImages(1, ([gray1, gray2], [img1, img2]))
    allStacked = cv2.putText(allStacked, f"Score: {score}", (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=1, color=(0, 0, 255), thickness=3)

    return allStacked


def compareDifferentCountourAreaValues(imagePath1: str, imagePath2: str, listAreaValues: list) \
        -> np.ndarray:
    """
    Function to compare different countsour area values
    :param imagePath1: Path to image
    :param imagePath2: Path to image
    :param listAreaValues: list of area values
    :return: np.ndarray
    """

    """# Iterate over different values of the min Contour Area Values
        imagePath1 = ""
        imagePath2 = ""

        minContourAreaList = [1, 100, 500, 1000, 5000, 10000]
        completeStack = compareDifferentCountourAreaValues(imagePath1, imagePath2, minContourAreaList)

        cv2.imshow("Stacked images", completeStack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

    if len(listAreaValues) % 2 != 0:
        return print("Put the number of list values in the list to be even")
    else:
        minContourAreaList = listAreaValues
        allStacked = []
        for value in minContourAreaList:
            temp = compare2TestImages(imagePath1, imagePath2, value)
            temp = cv2.putText(temp, f"Value: {value}", (220, 220), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)

            allStacked.append(temp)

        half = len(allStacked) // 2
        firstHalf = allStacked[:half]
        secondHalf = allStacked[half:]

        completeStack = stackImages(1, (firstHalf, secondHalf))
        return completeStack


def stackImages(scale: int, imgArray: tuple[list, list]) \
        -> np.ndarray:
    # Code taken from https://www.computervision.zone/topic/chapter-8-contour-shape-detection/

    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def get_dir_info(folderPath: str) -> dict:
    """
    Iterate over the directory and returns relevant information about the directory
    :param folderPath: Path to the directory
    :return: Dictionary containing the directory information
    """
    # Get the paths of all image files in the folder
    dirInfo = {'UniqueShapes': [],
               'TotalFiles': [],
               'ImagesProcessed': [],
               'ImagesUnprocessed': [],
               'ImagesUnprocessedPaths': [],
               'UniqueCameraNumbers': [],
               }

    # Get the images in the directory
    image_paths = list(paths.list_images(folderPath))
    # Counter to determine if all the images are processed
    images_processed = 0

    # Redundant Counter just for the sake of conformity
    images_unprocessed = 0

    # Total Number of Image Files
    total_files = len(image_paths)

    # Set to store unique image shapes
    unique_shapes = set()

    # Iterate over each image
    for image_path in image_paths:

        # Perform String Operations to get the camera number and :
        try:
            camera_number = int(image_path.split('/')[-1].split('-')[0].split('c')[1])
            if camera_number not in dirInfo['UniqueCameraNumbers']:
                dirInfo['UniqueCameraNumbers'].append(camera_number)
                dirInfo[f'ImagePathsForCamera{camera_number}'] = []
                dirInfo[f'ImagePathsForCamera{camera_number}'].append(image_path)
            else:
                dirInfo[f'ImagePathsForCamera{camera_number}'].append(image_path)

        except ValueError:
            try:
                camera_number = int(image_path.split('/')[-1].split('_')[0].split('c')[1])
                if camera_number not in dirInfo['UniqueCameraNumbers']:
                    dirInfo['UniqueCameraNumbers'].append(camera_number)
                    dirInfo[f'ImagePathsForCamera{camera_number}'] = []
                    dirInfo[f'ImagePathsForCamera{camera_number}'].append(image_path)
                else:
                    dirInfo[f'ImagePathsForCamera{camera_number}'].append(image_path)
            except ValueError:
                raise ValueError

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Get the unique shape (dimensions) of the image
        try:
            shape = image.shape
            images_processed += 1
        except AttributeError:
            images_unprocessed += 1
            dirInfo['ImagesUnprocessedPaths'].append(image_path)
            pass

        # Add the shape to the set of unique shapes
        unique_shapes.add(shape)

    dirInfo['UniqueShapes'] = unique_shapes
    dirInfo['TotalFiles'] = total_files
    dirInfo['ImagesProcessed'] = images_processed
    dirInfo['ImagesUnprocessed'] = images_unprocessed

    return dirInfo


def find_duplicate_images(folderPathLists: list[str], min_contour_area: int, gaussian_blur_radius_list: list = None) \
        -> dict:
    """
    Find duplicate images from a folderPathLists

    :param folderPathLists: Optional list of PathLists specifying from which Camera
    :param folderPath: path to folder
    :param min_contour_area: Minimum Contour Area
    :param gaussian_blur_radius_list: gaussian blur radius in x and y directions
    :return:
    """

    image_paths = folderPathLists
    duplicates = {'DuplicateList': []}

    # Compare each pair of image
    for i in range(len(image_paths)):
        for j in range(i, len(image_paths)):

            if image_paths[i] != image_paths[j]:
                if image_paths[i] not in duplicates['DuplicateList']:
                    imagePath1 = image_paths[i]
                    imagePath2 = image_paths[j]

                    image1 = cv2.imread(imagePath1)
                    image2 = cv2.imread(imagePath2)

                    # Load the image
                    try:
                        preprocessed_img1 = preprocess_image_change_detection(image1, gaussian_blur_radius_list)
                        preprocessed_img2 = preprocess_image_change_detection(image2, gaussian_blur_radius_list)
                    except AttributeError:
                        pass

                    if preprocessed_img1.shape != (240, 320):
                        preprocessed_img1 = cv2.resize(preprocessed_img1, (240, 320))
                    if preprocessed_img2.shape != (240, 320):
                        preprocessed_img2 = cv2.resize(preprocessed_img2, (240, 320))

                    if preprocessed_img1.shape == preprocessed_img2.shape:
                        score, _, _ = compare_frames_change_detection(preprocessed_img1, preprocessed_img2,
                                                                      min_contour_area)
                    else:
                        break

                    if score < 2500:
                        # Add duplicates found to the duplicate dict
                        if imagePath2 not in duplicates:
                            duplicates['DuplicateList'].append(imagePath2)
                else:
                    pass

    return duplicates


def main(folder: str, min_contour_area: int) -> None:
    dirInfo = get_dir_info(folder)
    print("Directory Information has been processed")
    print("----------------------------------------------------------------")
    print(f"Unique Image Shapes: {dirInfo['UniqueShapes']}")
    print("----------------------------------------------------------------")
    print(f"Total Files in Directory: {dirInfo['TotalFiles']}")
    print("----------------------------------------------------------------")
    print(f"Total Images Processed: {dirInfo['ImagesProcessed']}")
    print("----------------------------------------------------------------")
    print(f"Images Unprocessed: {dirInfo['ImagesUnprocessed']}")
    print("----------------------------------------------------------------")
    print(f"Total Unique Cameras found:{len({dirInfo['UniqueCameraNumbers']})} and they are"
          f" {dirInfo['UniqueCameraNumbers']}")
    print("----------------------------------------------------------------")
    for i, value in enumerate(dirInfo['UniqueCameraNumbers']):
        print(f"Total Number of files for Camera {value} : {len(dirInfo[f'ImagePathsForCamera{value}'])}")
        print("----------------------------------------------------------------")
    print()
    print("----------------------------------------------------------------")
    print("-----Now Starting finding duplicate Images for each camera------")
    for i, value in enumerate(dirInfo['UniqueCameraNumbers']):
        pathList = dirInfo[f'ImagePathsForCamera{value}']
        duplicate_images = find_duplicate_images(folder, min_contour_area)
    dirInfo['DuplicatePathsList'] = []
    dirInfo['DuplicatePathsList'].append(duplicate_images['DuplicateList'])
    totalDuplicates = len(duplicate_images)

    with open('./dirInfo.pkl', 'wb') as f:
        pickle.dump(dirInfo, f)

    for duplicate in duplicate_images['DuplicateList']:
        os.remove(duplicate)

    print(f"Removed duplicates, Total Number of duplicates found {totalDuplicates}")


if __name__ == '__main__':
    from time import time

    start = time()
    folderPath = "/Users/mcking/PycharmProjects/Kopernikus_Auto/dataset/"
    # duplicates = find_duplicate_images(folderPath, min_contour_area=500)
    try:
        with open('./dirInfo.pkl', 'rb') as f:
            dirInfo = pickle.load(f)
    except Exception as e:
        dirInfo = get_dir_info(folderPath)

    print("Done processing Directory Information")
    print("----------------------------------------------------------------")
    print(f"Unique Image Shapes: {dirInfo['UniqueShapes']}")
    print("----------------------------------------------------------------")
    print(f"Total Files in Directory: {dirInfo['TotalFiles']}")
    print("----------------------------------------------------------------")
    print(f"Total Images Processed: {dirInfo['ImagesProcessed']}")
    print("----------------------------------------------------------------")
    print(f"Images Unprocessed: {dirInfo['ImagesUnprocessed']}")
    print("----------------------------------------------------------------")
    print(f"Total Unique Cameras found:{len({dirInfo['UniqueCameraNumbers']})} and they are"
          f" {dirInfo['UniqueCameraNumbers']}")
    camera10Duplicates = find_duplicate_images(folderPathLists=dirInfo['ImagePathsForCamera20'], min_contour_area=500,
                                               gaussian_blur_radius_list=[3, 3])
    end = time()
    totalTimeInMinutes = round((end - start) / 60)
    print(f"Total Execution time in minutes: {totalTimeInMinutes} minutes")
    print(camera10Duplicates)
