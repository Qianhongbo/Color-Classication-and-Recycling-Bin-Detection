# Color Classiﬁcation and Recycling Bin Detection

## Summary

This project used Na¨ıve Bayes Model for color segmentation and bin detection. In this work, First train on the provided labeled color dataset, and test the results through the validation set. Due to the abundant color data provided, a very high accuracy rate can be obtained in the end. After that, bin detection is performed in a similar way. The difference is that we need to extract color data from the training set by ourselves, and finally select the detected bin box through opencv, and the final accuracy rate reaches about ninety percent. The accuracy rate can be increased by transforming the color from RGB space to YUV space, which will reduce the impact of sunlight on the surface of the object.

## Data

Under the corresponding folder.

## Requirement

opencv-python>=3.4

matplotlib>=2.2

numpy>=1.14

scikit-image>=0.14.0

glob2



## Result

![0062_img](https://tva1.sinaimg.cn/large/e6c9d24egy1h0nsb59d7dj20m80gomxb.jpg)

![0062_img](https://tva1.sinaimg.cn/large/e6c9d24egy1h0nsao9qrij20m80gojrz.jpg)

