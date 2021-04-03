# Volcano Detection on Venus Summary - Image Classification

Spending approximately 4 years traveling around Venus, the spacecraft named Magellan captured a large amount of data of the planet’s surface. The mission was to acquire
the topological relief of Venus by mapping the surface using synthetic aperture radar (Dheeru and Karra Taniskidou, 2017). Each image in the dataset has a view of less than
100m^2 on the ground. Researchers labelled each image with the following four tags: volcano detection (binary, 1 for presence and 0 for no volcanoes), uncertainty level of detection (4 levels, Figure 1), the number of volcanoes and the radius of volcanoes. Due to the ambiguity of the data, the four labels are not guaranteed to be absolute ground truths. The task is to train models to make predictions on these four labels.

Here is an example of the volcano image:

![alt text](https://github.com/xpada001/volcano_CNN/blob/main/example.png?raw=true)

We conducted classification on the first three labels and regression for radius estimation. In classification tasks, we realized that the dataset is imbalance
– that is, some classes have a large number of images but some classes have only a few. To tackle the imbalance issue, we could either oversample the images in the minor classes
or put weights on classes to balance the dataset. In our experiments, the CNN models we use are generally not stable due to imbalance data. Adding batch normalization layers significantly increases the model stability.


## Data Preprocessing

The dataset contains 7000 images in the training set and 2734 images in the test set. Each image has size 110x110 pixels with values in [0,255]. An image may have volcanoes or do
not have volcanoes. Researchers also labelled their uncertainty of volcano detection as 4 types:
 - Type 1: definitely has volcanoes,
 - Type 2: probably has volcanoes,
 - Type 3: possibly has volcanoes,
 - Type 4: only a pit is visible.

There are much more images without volcanoes than images with volcanoes. Among the images with positive volcano detection, the majority of discovered volcanoes are type 3 or 4, which means that the uncertainty on detected volcanoes is generally very low.

This image set has the imbalance classification problem, which means that the proportion of the positive samples is much less than the proportion of negative samples. To tackle this problem, we can perform oversampling on the underrepresented class or put class weights in our model. The ratio between the majority class and the minority class is about 6:1. We performed oversampling using SMOTE, which generates synthetic images to achieve balance among classes. Then, we also tried to assign different class weights, which are computed using the ratio of samples in each class. The comparison is also performed between the results using these two methods.

## CNN Models and Results

We experimented the CNN models with various parameter values and optimizers(more details can be found in the report), and we noticed that adding batch normalization after each convolution layer and the fully connected layer, our model and the VGG16 model both successfully learned from two classes and made meaningful predictions with good accuracy and F1 score consistently. Below is the architecture of our CNN model:

![alt text](https://github.com/xpada001/volcano_CNN/blob/main/CNN_architecture.png?raw=true)


The result below is a comparison of volcano detection (binary classification) with the CNN model and VGG16 model using transfer learning, and both solutions to imbalanced dataset are compared as well:


![alt text](https://github.com/xpada001/volcano_CNN/blob/main/volcano_detection_result.png?raw=true)

### Uncertainty Prediction

To predict the uncertainty of volcano detection, we used 4 types of uncertainty plus "no volcanoes" as the 5th label. Below is the result:

![alt text](https://github.com/xpada001/volcano_CNN/blob/main/volcano_uncertainty_result.png?raw=true)

Our model gives high accuracy but poor f1-score in the classes of certainty. Overall, assigning class weights gives slightly better result.


### Number of Volcanoes Prediction

The number of volcanoes in an image is also of interest. We trained our model with the number of volcanoes (including zero volcanoes) as labels and obtain 94.95% accuracy in
the test set. However, for the images with volcanoes only, i.e. we excluded the case of zero volcanoes, the accuracy is about 76%. Furthermore, we also tried different models to make predictions (i.e. Support Vector Machine, Random Forest, XGBoost). Our classification model has already extracted a lot of representative features of the surface of Venus. So we extracted the output before the very last dense layer and take each row as the features of an image. Then we fitted the features to the classifiers mentioned earlier. The results are shown below:

![alt text](https://github.com/xpada001/volcano_CNN/blob/main/classifiers_result.png?raw=true)


These three classifiers yield higher accuracy than the CNN model. Hence it is suggested to combine CNN and one of the three classifiers above for the number of volcanoes prediction.


### Radius of Volcanoes
We found that estimating the radius of volcanoes is the most difficult task. First, we conducted regression to predict radius with various regressors, such as Gradient Boost and multilayer perceptron model. We tried to use the original images or the output before the last dense layer of the binary classification model as the input
to the regressors, but we did not obtain good results. The models overfit the training data severely. Parameter tuning still gives us no satisfactory results. Then, we relaxed the regression problem to a classification problem – dividing the data into 4 groups by thresholding the radius. However, such relaxation still does not give a desired solution as expected. More investigations would be needed to find a good model for the radius prediction.


## Data Augmentation using Generative Adversarial Networks (GAN)

Beside the typical data augmentation of producing new samples by flipping the images, I also tried to use GAN to do it. Even though the discrimnator and the generator are working properly with decreasing loss duing model training. The computation power of my machine is not sufficient to complete the training. You can view the notebook for more details.
