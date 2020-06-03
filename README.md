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
  ```
  ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 128, 128]             864
       BatchNorm2d-2         [-1, 32, 128, 128]              64
              ReLU-3         [-1, 32, 128, 128]               0
            Conv2d-4         [-1, 32, 128, 128]             288
            Conv2d-5         [-1, 32, 128, 128]           1,024
       BatchNorm2d-6         [-1, 32, 128, 128]              64
              ReLU-7         [-1, 32, 128, 128]               0
            Conv2d-8         [-1, 32, 128, 128]             864
       BatchNorm2d-9         [-1, 32, 128, 128]              64
             ReLU-10         [-1, 32, 128, 128]               0
           Conv2d-11         [-1, 32, 128, 128]             288
           Conv2d-12         [-1, 32, 128, 128]           1,024
      BatchNorm2d-13         [-1, 32, 128, 128]              64
             ReLU-14         [-1, 32, 128, 128]               0
           Conv2d-15         [-1, 64, 128, 128]          36,864
      BatchNorm2d-16         [-1, 64, 128, 128]             128
           Conv2d-17        [-1, 128, 128, 128]          73,728
      BatchNorm2d-18        [-1, 128, 128, 128]             256
             ReLU-19        [-1, 128, 128, 128]               0
           Conv2d-20          [-1, 1, 128, 128]           1,152
           Conv2d-21         [-1, 64, 128, 128]          36,864
      BatchNorm2d-22         [-1, 64, 128, 128]             128
           Conv2d-23         [-1, 64, 128, 128]          36,864
      BatchNorm2d-24         [-1, 64, 128, 128]             128
       BasicBlock-25         [-1, 64, 128, 128]               0
           Conv2d-26          [-1, 128, 64, 64]          73,728
      BatchNorm2d-27          [-1, 128, 64, 64]             256
           Conv2d-28          [-1, 128, 64, 64]         147,456
      BatchNorm2d-29          [-1, 128, 64, 64]             256
           Conv2d-30          [-1, 128, 64, 64]           8,192
      BatchNorm2d-31          [-1, 128, 64, 64]             256
       BasicBlock-32          [-1, 128, 64, 64]               0
           Conv2d-33          [-1, 256, 32, 32]         294,912
      BatchNorm2d-34          [-1, 256, 32, 32]             512
           Conv2d-35          [-1, 256, 32, 32]         589,824
      BatchNorm2d-36          [-1, 256, 32, 32]             512
           Conv2d-37          [-1, 256, 32, 32]          32,768
      BatchNorm2d-38          [-1, 256, 32, 32]             512
       BasicBlock-39          [-1, 256, 32, 32]               0
           Conv2d-40          [-1, 512, 16, 16]       1,179,648
      BatchNorm2d-41          [-1, 512, 16, 16]           1,024
           Conv2d-42          [-1, 512, 16, 16]       2,359,296
      BatchNorm2d-43          [-1, 512, 16, 16]           1,024
           Conv2d-44          [-1, 512, 16, 16]         131,072
      BatchNorm2d-45          [-1, 512, 16, 16]           1,024
       BasicBlock-46          [-1, 512, 16, 16]               0
           Conv2d-47           [-1, 1024, 8, 8]       4,718,592
      BatchNorm2d-48           [-1, 1024, 8, 8]           2,048
           Conv2d-49           [-1, 1024, 8, 8]       9,437,184
      BatchNorm2d-50           [-1, 1024, 8, 8]           2,048
           Conv2d-51           [-1, 1024, 8, 8]         524,288
      BatchNorm2d-52           [-1, 1024, 8, 8]           2,048
       BasicBlock-53           [-1, 1024, 8, 8]               0
           Conv2d-54          [-1, 512, 16, 16]       4,718,592
      BatchNorm2d-55          [-1, 512, 16, 16]           1,024
           Conv2d-56          [-1, 512, 16, 16]       2,359,296
      BatchNorm2d-57          [-1, 512, 16, 16]           1,024
           Conv2d-58          [-1, 512, 16, 16]         524,288
      BatchNorm2d-59          [-1, 512, 16, 16]           1,024
       BasicBlock-60          [-1, 512, 16, 16]               0
           Conv2d-61          [-1, 512, 16, 16]       2,359,296
      BatchNorm2d-62          [-1, 512, 16, 16]           1,024
           Conv2d-63          [-1, 512, 16, 16]       2,359,296
      BatchNorm2d-64          [-1, 512, 16, 16]           1,024
       BasicBlock-65          [-1, 512, 16, 16]               0
           Conv2d-66          [-1, 512, 16, 16]       4,718,592
      BatchNorm2d-67          [-1, 512, 16, 16]           1,024
             ReLU-68          [-1, 512, 16, 16]               0
           Conv2d-69          [-1, 256, 32, 32]       1,179,648
      BatchNorm2d-70          [-1, 256, 32, 32]             512
           Conv2d-71          [-1, 256, 32, 32]         589,824
      BatchNorm2d-72          [-1, 256, 32, 32]             512
           Conv2d-73          [-1, 256, 32, 32]         131,072
      BatchNorm2d-74          [-1, 256, 32, 32]             512
       BasicBlock-75          [-1, 256, 32, 32]               0
           Conv2d-76          [-1, 256, 32, 32]       1,179,648
      BatchNorm2d-77          [-1, 256, 32, 32]             512
             ReLU-78          [-1, 256, 32, 32]               0
           Conv2d-79          [-1, 128, 64, 64]         294,912
      BatchNorm2d-80          [-1, 128, 64, 64]             256
           Conv2d-81          [-1, 128, 64, 64]         147,456
      BatchNorm2d-82          [-1, 128, 64, 64]             256
           Conv2d-83          [-1, 128, 64, 64]          32,768
      BatchNorm2d-84          [-1, 128, 64, 64]             256
       BasicBlock-85          [-1, 128, 64, 64]               0
           Conv2d-86          [-1, 128, 64, 64]         294,912
      BatchNorm2d-87          [-1, 128, 64, 64]             256
             ReLU-88          [-1, 128, 64, 64]               0
           Conv2d-89         [-1, 64, 128, 128]          73,728
      BatchNorm2d-90         [-1, 64, 128, 128]             128
           Conv2d-91         [-1, 64, 128, 128]          36,864
      BatchNorm2d-92         [-1, 64, 128, 128]             128
           Conv2d-93         [-1, 64, 128, 128]           8,192
      BatchNorm2d-94         [-1, 64, 128, 128]             128
       BasicBlock-95         [-1, 64, 128, 128]               0
           Conv2d-96        [-1, 128, 128, 128]         147,456
      BatchNorm2d-97        [-1, 128, 128, 128]             256
             ReLU-98        [-1, 128, 128, 128]               0
           Conv2d-99          [-1, 1, 128, 128]           1,152
================================================================
Total params: 40,866,048
Trainable params: 40,866,048
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 391.75
Params size (MB): 155.89
Estimated Total Size (MB): 547.83
----------------------------------------------------------------
```  
  
  For Practice, I started with 160k images(40k images each ) total. My approach is to start with a small dataset and get the proof of the working model and tune up the data augmentation and, at last, train the whole dataset.
  Data Augmentation: I converted all images into 128x128 to avoid GPU memory out of error. And I used ColorJitter to change the contrast, brightness, and saturation of images randomly. 
  > transforms.Resize((128, 128)),
  > transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue= 0.15)
  

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
