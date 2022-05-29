# CarDetection
SVM Pipelining Steps:
• Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labelled training set of images and train a classifier Linear SVM classifier.
• A colour transform is applied to the image and append binned colour features, as well as histograms of colour, to HOG feature vector. 
• Normalize your features and randomize a selection for training and testing.
• Implement a sliding-window technique and use SVM classifier to search for vehicles in images.
• Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
• Estimate a bounding box for detected vehicles.

1. Collecting Data:
There are two approaches: 
• we can use an existing and trained classifier directly.
• Build our own classifier by using a dataset which contains of car pictures and not-car pictures.

 To specify which approach is used there is a variable called DataMode, if it assigned to 1 so it will use an existing classifier, otherwise it will build its own one.

2. Extracting HOG features:
This can be reached by using the function extract_feature()
The parameters of HOG are:
* `YCrCb` color space
* orient = 9 # HOG orientations
* pix_per_cell = 8 # HOG pixels per cell
* cell_per_block = 2 # HOG cells per block, which can handel e.g. shadows
* hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
* spatial_size = (32, 32) # Spatial binning dimensions
* hist_bins = 32 # Number of histogram bins
* spatial_feat = True # Spatial features on or off
* hist_feat = True # Histogram features on or off
* hog_feat = True # HOG features on or off

The output image for HOG features of cars and not-cars

![Figure_1](https://user-images.githubusercontent.com/106494037/170894235-fdff379d-9e8b-46b6-a1da-5562e90542a7.png)

3. Sliding window search:
For this SVM-based approach, it uses two scales of the search window 64x64 and 128x128 and search only between [400, 656] in y axis.
For every window, the SVM classifier is used to predict whether it contains a car nor not. If yes, save this window. In the end, a list of windows contains 
detected cars are obtained.

![search_windows](https://user-images.githubusercontent.com/106494037/170894280-d8c31803-396f-4b23-9e5a-3d33fd7653b9.png)

4. Create a heat map of detected vehicles
After obtained a list of windows which may contain cars, a function named generate_heatmap is used to generate a heatmap. Then a threshold is used to filter out the false positives.

![heat_map1](https://user-images.githubusercontent.com/106494037/170894307-c89fe73f-de70-4221-b411-b58a11867454.png)

5. The Output Image
6. ![output_bboxes](https://user-images.githubusercontent.com/106494037/170894334-9903d421-4ab5-4ae5-bfc3-3b76da24f447.png)
7. 


