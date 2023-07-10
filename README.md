### KopernikusAutoTest

##### Question 1:What did you learn after looking on our dataset?

* First thing that it is quite large. The photos are of a parking lot but from different angles. Total number of cameras placed are 4. Just like the assignment suggested way too many duplicates. For each camera the timestamps are taken in a 5 minute interval approximately.
* Hence I have compared same camera images with each other as it didn't made sense to me to compare with other cameras using compare_frames_change_detection given by you.
* **Unique Image Shapes**: {(619, 1100), (6, 10), (1520, 2688), (480, 640), (675, 1200), (1080, 1920)}. ***But I have added a part if the image size is (6,10) or if the file can't be processed it should remove it.***  
* Total Files in Directory: 1080, Total Images Processed: 1079, Images Unprocessed: 1. Unprocessed path can be found from dirInfo. More about it later.
  *  **Total Number of files for Camera 23** : 484
  *  **Total Number of files for Camera 20** : 324, 
  *  **Total Number of files for Camera 21** : 146
  *  **Total Number of files for Camera 10** : 126, 
*  The images are with all different lighting conditions from day to night.
*  Timestamps provided by the image I was unable to process via python datetime (SCHADE! Quite frustrating why it wont process but it worked online using this website https://www.epochconverter.com/). I didn't try further to use it in my logic to remove duplicates nut can be implemented. 
  
##### Question 2:How does you program work?

* In terms of executing the code itself its pretty simple in the **deleteDuplicates.py** file if you pass the folder path to **main("FolderPath")** it will procede to do the rest. Or one can run from terminal by providing folder path like: **python deleteDuplicates.py --folderPath**. More on it at the end! 
* The logic of the code is:
  * The function **get_dir_info** creates a summary of the folder and images from each camera given in the folder. It is saved in dirInfo
    * It is used to keep track of the state and at the end of the main loop it is **saved as .pkl in the cwd**
  * Function **find_duplicate_images** finds the list of duplicate images from a given camera and returns duplicates. The list can be accessed from **dirInfo**
  * Once all the files are compared and processed  the list of duplicates is accessed and deleted in the main function.

##### Question 3:What values did you decide to use for input parameters and how did you find these values?

* Oh! Its a really good question! This was that made me think about my strategy. So there were **4** parameters I figure out:
* These are **Image dimensions** to be used to compare, **Gaussian Blur**, **score**, **min_area_contour**,
* Except one anamoly of **(6,10)** image size the **smallest is dimension is 480**. I took **half** of it as the image size to compare. Can be even smaller to make it faster! 
* For Gaussian blur I took the **radius to be 3**. 
* For **score threshold** used as a comparator to add a given path to a duplicate list, I have made a function which compares two images given the process mentioned in the pdf doc. Since I was not supposed to come up with the algo I didn't delved into it oo much. But there are resouces online recommenddde to use hash functions. However, I haven't checked if they are faster.
  * The score was ***zero for same images or almost similar images***. It increased as the images had varied illumination or an object present which wasn't there in the scene before.
* Coming to the last part selecting min_contour_area. I have created a function called **compareDifferentContourAreaValues**.
  * It requires 2 images and a list of  of values one would like to try. This has to be even in size because it uses a function called **stackImages** which I found online (https://www.computervision.zone/topic/chapter-8-contour-shape-detection/) which requires it to be **even**. It gives the score from the **compare_frame_change_detection** provided and shows in a single image.
  * The values I tried are **1, 100, 500, 1000, 5000, 10000**. Beyond 1000 scores became unreliable and the score values for 1,100, 1000 were almost similar with a deviation of around 200 for min_contour_area = 1 with the rest.
   
##### Question 4:What you would suggest to implement to improve data collection of unique cases in future?

* For each camera keep the images corresponding to it in a seperate folder. This would be the first one 😊 as it makes life easier.
* Secondly, if they are from the same camera I beleive they should be of the same dimension which is not the case in the given dataset and should be stored with the same dimension.
* The naming convention of the image stored should be constant. For camera 20 Two different ways images were saved, one with timestamp value and the second with the datetime value itself. **This increases the processing time!**
* Also in my opinion to make life easier timestamp should be in UTC and not in local timezone. However, this might be just my frustration speaking out 😊. 

 
##### Question 5:Any other comments about your solution?
* Nothing more. I guess I have said enough 😊. I hope my sollution impresses you and looking forward to hear from you guys soon! Fingers crossed!

## Running the Program:
At First, install libraries
* pip install -r requirements.txt 

2 Ways:
1. From terminal
  * python deleteDuplicates.py --folderPath FolderPath --minContourArea minContourArea --gaussianBlur GaussianBlur --resizeImage resizeImage
2. From IDE by setting the required parameters using the main() function

  


