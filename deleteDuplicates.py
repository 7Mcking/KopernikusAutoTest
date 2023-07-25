# Created by mcking at 09.07.23
from imaging_interview import *
import cv2
from imutils import paths
import os
import numpy as np
import pickle
from tqdm import tqdm


def areAllElementsUnique(lst: list):
    return len(lst) == len(set(lst))


def compare2TestImages(imgPath1: str, imgPath2: str, minContourArea: int, gaussianBlurRadius: [int, int]) \
        -> np.ndarray:
    """
    Compare two images and return their comparison scores along with their
    times

    :param minContourArea: Minimum contour area
    :param gaussianBlurRadius: Radius for gaussian blur
    :param imgPath1: path to the first image
    :param imgPath2: path to the second image
    :return: np.ndarray
    """
    imagePath1 = imgPath1
    imagePath2 = imgPath2

    img1 = cv2.imread(imagePath1)
    img2 = cv2.imread(imagePath2)

    gray1 = preprocess_image_change_detection(img1, gaussianBlurRadius)
    gray2 = preprocess_image_change_detection(img2, gaussianBlurRadius)

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


def compareDifferentCountourAreaValues(imagePath1: str, imagePath2: str, listAreaValues: list,
                                       gaussianBlurRadius=[5, 5]) -> np.ndarray:
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
        completeStack = compareDifferentCountourAreaValues(imagePath1, imagePath2, 
        minContourAreaList, gaussianBlurRadius)

        cv2.imshow("Stacked images", completeStack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

    if len(listAreaValues) % 2 != 0:
        return print("Put the number of list values in the list to be even")
    else:
        minContourAreaList = listAreaValues
        allStacked = []
        for value in minContourAreaList:
            temp = compare2TestImages(imagePath1, imagePath2, value, gaussianBlurRadius)
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
               'PathsDeleted': [],
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
            if shape == (6, 10):
                os.remove(image_path)
                dirInfo['PathsDeleted'].append(image_path)
                continue
        except AttributeError:
            images_unprocessed += 1
            dirInfo['PathsDeleted'].append(image_path)
            os.remove(image_path)
            pass

        # Add the shape to the set of unique shapes
        unique_shapes.add(shape)

    dirInfo['UniqueShapes'] = unique_shapes
    dirInfo['TotalFiles'] = total_files
    dirInfo['ImagesProcessed'] = images_processed
    dirInfo['ImagesUnprocessed'] = images_unprocessed

    return dirInfo


def find_duplicate_images(folderPathLists: list[str], min_contour_area: int, gaussianBlurRadius: list[int, int] = None,
                          resizeImage: tuple[int, int] = None) -> dict:
    """
    Find duplicate images from a folderPathLists

    :param resizeImage: tuple of (width, height) to resize the image
    :param folderPathLists: Optional list of PathLists specifying from which Camera
    :param min_contour_area: Minimum Contour Area
    :param gaussianBlurRadius: gaussian blur radius in x and y directions
    :return:
    """

    image_paths = folderPathLists
    # Dictionary to store the duplicate images
    duplicates = {'DuplicateList': []}
    # 1. Create a dcit
    # 2. Iterate over the list of image paths
    #
    # Compare each pair of image
    image_path_dicts = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)

        preprocessed_img1 = cv2.resize(image, resizeImage)
        preprocessed_img1 = preprocess_image_change_detection(preprocessed_img1, gaussianBlurRadius)

        image_path_dicts[image_path] = preprocessed_img1

    for i in tqdm(range(len(image_path_dicts.keys())), desc="Processing..."):

        for j in range(i + 1, len(image_path_dicts.keys()) - 1):

            # Check if the image paths are not the same
            myList = list(image_path_dicts.keys())
            if myList[i] != myList[j]:
                # Check if the image paths are not already in the duplicate list
                if (myList[i] not in duplicates['DuplicateList']) and \
                        (myList[j] not in duplicates['DuplicateList']):

                    imagePath1 = myList[i]
                    imagePath2 = myList[j]

                    # Load the images
                    t0 = time()
                    image1 = image_path_dicts[imagePath1]
                    image2 = image_path_dicts[imagePath2]

                    # Compare the images
                    score, _, _ = compare_frames_change_detection(image1, image2,
                                                                  min_contour_area)

                    # If the score is less than 2500, then the images are duplicates
                    if score < score_threshold:
                        # Add duplicates found to the duplicate dictionary
                        duplicates['DuplicateList'].append(imagePath2)
                else:
                    pass
            else:
                pass

    if areAllElementsUnique(duplicates['DuplicateList']):
        print("All Unique duplicates found")
        print("----------------------------------------------")
        print(f"Total Unique Duplicates Found: {len(duplicates['DuplicateList'])}")
        return duplicates
    else:
        print("The duplicates are not unique")
        raise AssertionError


def getInfoFromExistingDirInfoFile(dirInfoFile: dict) -> None:
    """

    :param dirInfoFile: path to directory information file
    """
    print("Directory Information has been processed")
    print("----------------------------------------------------------------")
    print(f"Unique Image Shapes: {dirInfoFile['UniqueShapes']}")
    print("----------------------------------------------------------------")
    print(f"Total Files in Directory: {dirInfoFile['TotalFiles']}")
    print("----------------------------------------------------------------")
    print(f"Total Images Processed: {dirInfoFile['ImagesProcessed']}")
    print("----------------------------------------------------------------")
    print(f"Images Unprocessed: {dirInfoFile['ImagesUnprocessed']}")
    print("----------------------------------------------------------------")
    print(f"Images deleted: {dirInfoFile['PathsDeleted']}")
    print("----------------------------------------------------------------")
    print(f"Total Unique Cameras found:{len(dirInfoFile['UniqueCameraNumbers'])} and they are"
          f" {dirInfoFile['UniqueCameraNumbers']}")
    print("----------------------------------------------------------------")
    getCameraInfo(dirInfoFile)


def getCameraInfo(dirInfoFile: dict) -> None:
    """
    :param dirInfoFile: path to directory information file
    """

    for i, uniqueCamera in enumerate(dirInfoFile['UniqueCameraNumbers']):
        print(
            f"Total Number of files for Camera {uniqueCamera} : "
            f"{len(dirInfoFile[f'ImagePathsForCamera{uniqueCamera}'])}")
        print("----------------------------------------------------------------")


def processUserInput(dirInfoFile: dict, totalDuplicates: int) -> None:
    """

    :param dirInfoFile: path to directory information file
    :param totalDuplicates: length of the duplicate list from the directory information file
    """
    print(f"Total Number of duplicates found {totalDuplicates}")
    print("----------------------------------------------------------------")
    print("Removing Duplicates from the directory")
    print("----------------------------------------------------------------")

    while True:
        print("Do you want to remove the duplicates? (y/n). Note: This cannot be undone. "
              "The input is Case Sensitive")
        userResponse = input("> Enter your response: ")

        if userResponse == 'y':
            for duplicateList in dirInfoFile['DuplicatePathsList']:
                for eachFile in duplicateList:
                    os.remove(eachFile)
            return print("Duplicates removed")

        elif userResponse == 'n':
            print("----------------------------------------------------------------")
            return print("Duplicates not removed")

        else:
            print("Invalid Response")
            print("----------------------------------------------------------------")


def makeNewDirInfoFile(folderPath, min_contour_area: int, gaussianBlur: [int, int],
                       resizeImage: tuple[int, int] = None) -> dict:
    # Get the directory information
    dirInfo = get_dir_info(folderPath)
    # Get the unique shapes and Camera Numbers
    getCameraInfo(dirInfo)

    # Create an empty list of duplicate paths
    dirInfo['DuplicatePathsList'] = []

    print("----------------------------------------------------------------")
    print("Now Starting finding duplicate Images for each camera")

    # Find the duplicate images for each camera and add them to the duplicate list
    for i, value in enumerate(sorted(dirInfo['UniqueCameraNumbers'])):
        print("----------------------------------------------------------------")
        print(f"Finding Duplicate Images for Camera {value}")
        pathList = dirInfo[f'ImagePathsForCamera{value}']
        duplicate_images = find_duplicate_images(pathList, min_contour_area,
                                                 gaussianBlurRadius=gaussianBlur, resizeImage=resizeImage)
        dirInfo.setdefault(f'DuplicatePathListForCamera{value}', []).append(duplicate_images['DuplicateList'])
        dirInfo['DuplicatePathsList'].append(duplicate_images['DuplicateList'])

    print("----------------------------------------------------------------")

    # Get the total number of duplicates
    totalDuplicates = 0
    for i in range(len(dirInfo['DuplicatePathsList'])):
        totalDuplicates += len((dirInfo['DuplicatePathsList'][i]))

    # Save the directory information to a pickle file
    with open('./dirInfo.pkl', 'wb') as f:
        pickle.dump(dirInfo, f)

    return dirInfo, totalDuplicates


def main(folder: str, min_contour_area: int, gaussianBlur: [int, int], resizeImage=None, score_threshold=2500) -> None:
    # Check if the directory information file exists
    dirInfoFile = './dirInfo.pkl'
    if os.path.exists(dirInfoFile):
        print("Directory Information File exists. Would you like to use it? (y/n)")
        choice = input("Enter your choice: ")
        # If the user wants to use the existing directory information file
        if choice == 'y':
            with open(dirInfoFile, 'rb') as f:
                dirInfo = pickle.load(f)
                getInfoFromExistingDirInfoFile(dirInfo)
                getCameraInfo(dirInfo)
                totalDuplicates = len(dirInfo['DuplicatePathsList'])
                processUserInput(dirInfo, totalDuplicates)

        # If the user wants to create a new directory information file
        elif choice == 'n':
            print("----------------------------------------------------------------")
            print("Creating a new one and deleting the old one...")
            print("----------------------------------------------------------------")
            os.remove(dirInfoFile)
            dirInfo, totalDuplicates = makeNewDirInfoFile(folder, min_contour_area, gaussianBlur,
                                                          resizeImage=resizeImage)
            processUserInput(dirInfo, totalDuplicates)

    # If the directory information file does not exist
    else:
        print("Creating a new one...")
        dirInfo, totalDuplicates = makeNewDirInfoFile(folder, min_contour_area, gaussianBlur, resizeImage=resizeImage)
        processUserInput(dirInfo, totalDuplicates)


if __name__ == '__main__':
    from time import time
    import argparse

    start = time()

    parser = argparse.ArgumentParser(description='Find Duplicate Images in a folder')
    parser.add_argument('--folderPath', type=str, help='Path to the folder containing images',
                        default='./dataset', required=False)
    parser.add_argument('--minContourArea', type=int, help='Minimum Contour Area', default=500)
    parser.add_argument('--gaussianBlur', type=int, nargs=2, help='Gaussian Blur Radius',
                        default=[5, 5])
    parser.add_argument('--resizeImage', type=int, nargs=2, help='Resize Image',
                        default=(240, 240))
    parser.add_argument('--scoreThreshold', type=int, help='Enter Score Threshold', default=2500)
    args = parser.parse_args()
    folderPath = args.folderPath
    min_contour_area = args.minContourArea
    gaussianBlur = args.gaussianBlur
    resizeImage = args.resizeImage
    score_threshold = args.scoreThreshold

    main(folder=folderPath, min_contour_area=min_contour_area, gaussianBlur=gaussianBlur, resizeImage=resizeImage,
         score_threshold=args.scoreThreshold)
    end = time()
    totalTimeInMinutes = round((end - start) / 60)
    print(f"Total Execution time in minutes: {totalTimeInMinutes} minutes")
