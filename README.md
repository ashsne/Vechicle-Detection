# Vehicle-Detection

[//]: # (Image References)

[image1]: ./output_images/Pipeline.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/YUV.png
[image4]: ./output_images/boxes.png
[image5]: ./output_images/Heatmap.png
[image6]: ./output_images/Car-NonCar.png


### Goal

The projects aims towards classification of vehicles and non-vehicles through a SVM classifier. An image undergoes following processing steps to produce the image with identified (if any) as shown.
- Feature Extraction
  - Colorspace histograms (YUV)
  - Spatial features
  - Histogram Oriented Gradients (HOG) features
- Classifier
- Sliding Window Search
- Heatmaps
 
### 1. Feature Extraction.

In the field of computer vision, a *features* is a compact representation that encodes information that is relevant for a given task. In our case, features must be informative enough to distinguish between *car* and *non-car* image patches as accurately as possible.

Here is an example of how the `vehicle` and `non-vehicle` classes look like in this dataset:
![alt text][image6]


#### - Colorspace histograms

By analysig various iterations of different colorspaces such as RGB, YUV, LUV, HLS etc it was found that YUV colorspace, particularly Y channel shown in folowing image, histograms are beneficial in identifying the cars.

![alt text][image3]


#### - Spatial features

For the task of car detection I used *colorspace histograms* and *spatial features* to encode the object visual appearence and HOG features to encode the object's *shape*. While color the first two features are easy to understand and implement, HOG features can be a little bit trickier to master.


#### - HOG features

HOG stands for *Histogram of Oriented Gradients* and refer to a powerful descriptor that has met with a wide success in the computer vision community, since its [introduction](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf) in 2005 with the main purpose of people detection. 

![alt text][image2]


### 2. Training the classifier and classifiaction

After getting all the decided features for all the images from the dataset, a SVM classifier is trained for the car identification

Then, the actual training set is composed as the set of all car and all non-car features (labels are given accordingly). Furthermore, feature vectors are standardize in order to have all the features in a similar range and ease training.
```
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

# Use a linear SVC 
svc = LinearSVC(max_iter=10000)
svc.fit(X_train, y_train)

```
In order to have an idea of the classifier performance, we can make a prediction on the test set with `svc.score(X_test, y_test)`. Training the SVM with the features explained above took around 10 minutes on my laptop. 

### 3. Sliding Window Search
Firstly a naive sliding window detection was used, but unfortunately it was vey time consuming. So an alternate method was implemented with Region of Interest (ROI). Small windows are used towards horizon and keeping bigger ones toeards clode tho the car. FOr each such window features are extracteed and then passed through prediction of the vehicles. Such an example is shown below:

![alt text][image4]


### 4. Heatmaps
As you can see there are multiple windows where cas is found. Also as accuracy is not 100%, we will have some mispredictions of the cars. To handle such events we pass these windows through the heatmap function. It creates heat signatures at the centroid of the window. With a threshold value of number of windows we can keep the windows we are interested.

![alt text][image5]


## Pipeline
The process pipeline can be visualized as follows:

![alt text][image1]



