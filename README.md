# Predict-Me

This repo to predict me using different model from logistic regression to shallow NN, to Deep NN using FastAPi, and let other clone the project to train their images and predict themselves.

Because of any model need a fixed size input we have manipulate an image into fixed size, and you can control these sizes using two variable:
- resize_width
- resize_height 

![Predict Me](images/me.png)

## To train your images
- git clone https://github.com/Abdelrahmanrezk/Predict-Me.git
- make new direction inside dataset with name "mixed_size"
- put inside this new direction two direction "me" and "other"
- put your image inside me direction
- put other images inside other direction
- Leave other dierction it contain 
- search for read_mixed_width_and_height_images() method and uncomment it
- Run the notebook
	- It will first resize your image into fixed size
	- Then load dataframe and handle labels for you
	- Then shuffle the data
	- Split the data
	- Just follow the instructure

**Ensure you have these direction:**

- SAVE_DIR = "images_uploaded/"
- DATA_MIXED_SIZE='dataset/mixed_size/'
- DATA_FIXED_SIZE="dataset/fixed_sized/"


## Use FastApi
- run FastAPI_for_prediction notebook
- Head over http://localhost:5008/docs
- Click try it
- choose model (logistic regression for now)
- Upload image
- Click Execute

**see "predict-me.mp4" for more details**  