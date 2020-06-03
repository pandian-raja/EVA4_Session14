# EVA4_Session15
## Plan for completing the assignment.
Predict Mask from the background and background with the person.
Predict Depth and Mask from the background and background with the person.

## 1. Predict Mask from the background and background with the person.
My method is to subtract (the background with the person) with background image). The result was better than the concating of two images.
After 10 Epoch, The Training Loss for subtraction method is 0.017246 and concating method's loss is 0.022197

## 2. Predict Depth and Mask images from the background and background with the person.
  Model: My intuition is to combine ResNet and U-nets. From https://towardsdatascience.com/u-nets-with-resnet-encoders-and-cross-connections-d8ba94125a2c. My model has 40,866,048 parameters. 
  ![U-net model](http://deeplearning.net/tutorial/_images/unet.jpg)
  For Practice, I started with 160k images(40k images each ) total. My approach is to start with a small dataset and get the proof of the working model and tune up the data augmentation and, at last, train the whole dataset.
  Data Augmentation: I converted all images into 128x128 to avoid GPU memory out of error. And I used ColorJitter to change the contrast, brightness, and saturation of images randomly. 

## Trail 1: Predicting only Depth Image(RGB)
	For the trail, I first started with predicting only the depth image from background and background with the person images, but the result was not working, and all images were blank.

## Trail 2:  Predicting Mask and Depth Image(RGB)
  I tried to predict the mask and depth and assume that adding the mask will help to predict the depth image. Only the mask was predicting, and depth was not predicting at all. Almost the same result as the (Trail 1).

## Trial 3: Predicting Mask and Depth Image(Grayscale)
  Since the depth image is black and white. I tried two methods.
    1. Only Depth images as Grayscale.
    2. Both Mask and Depth Images as GrayScale.
  Unfortunately, both methods didn't work. 

> Few minor other trials like resizing input size, changing model architecture, and tried loss functions like MSELoss, SSIM, but all my trials didn't work. My intuition is the base model is wrong, and I've to work on base model to predict depth image.
