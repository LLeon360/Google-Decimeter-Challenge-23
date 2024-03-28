# Google-Decimeter-Challenge-23
## Blurb
**Goal of the Competition**

The goal of this competition is to determine the limit of smartphone GNSS positioning accuracy: that could be down to the decimeter or even centimeter level. You will develop a model based on raw GNSS measurements from Android phones.

Your work will help produce better positions, bridging the connection between the geo-spatial information of finer human behavior and mobile internet with improved granularity. Additionally, the more precise data could lead to new navigation methods.

# Approach
Used a Transformer Encoder, like BERT to do bidirectional self-attention on a sequence of Weighted Least Squares BLH positions and estimated Ionospheric and Tropospheric delay. Then predict residuals, the difference between GT and WLS positions from GNSS.

# IPYNB on Kaggle:
https://www.kaggle.com/code/leonliu360/gnss-smartphone-positioning-transformer

### Competition Link:
https://www.kaggle.com/competitions/smartphone-decimeter-2023

