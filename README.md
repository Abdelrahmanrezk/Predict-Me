# Predict-Me

This repo to predict me using different model from logistic regression to shallow NN, to Deep NN using FastAPi, and let other clone the project to train their images and predict themselves.

Because of any model need a fixed size input we have manipulate an image into fixed size, and you can control these sizes using two variable:
- resize_width
- resize_height 

## What we have Done
- Read all images inside dataset/mixed_size
	- resize the image inside me dierction into fixed size
	- resize the image inside other dierction into fixed size
- scale_image
	- scale image from intensity of 0 to 255 with different ranges into 0 - 1

## To train your images
- Clone the project
- remove images inside dataset/mixed_size/me
- remove images inside dataset/fixed_sized/me
- Leave other dierction it contain 

