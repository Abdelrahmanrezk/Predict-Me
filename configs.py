import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import os
import time


SAVE_DIR = "images_uploaded/"
DATA_MIXED_SIZE='dataset/mixed_size/'
DATA_FIXED_SIZE="dataset/fixed_sized"

TENSOR_DIR = os.path.join(os.curdir, 'tensor_logs')
MODELS_DIR = os.path.join(os.curdir, 'saved_model_weights') 

## This is control your image width and height
resized_dim = 100 

# at the end and because it rgb images result will be 100 * 100 * 3

def scale_image(image_0_255):
    """Return a numpy array with scaled image from 0 to 1"""
    
    # Scale pixel intensity to be from 0 - 1
    image_0_1 = np.asarray(image_0_255) / 255
    
    # use just 16 bit for each pixel
    image_0_1 = image_0_1.astype(np.float16)
    
    return image_0_1

def save_fixed_width_and_height_image(open_image, which_dir, image_name, resized_dim=100,  img_size="_100_100_3/"):
    '''
    The function used to take a path of image, then compress into some resize_width and resize_height,
    then save the new image into the DATA_FIXED_SIZE (dierction contain just image of specific width and height).
    
    Because we have images of me and other so we have to save into specific "which_dir" direction
    '''
    
    # Read the image
    img = Image.open(open_image)
    
    # Resize the opened image
    img = img.resize((resized_dim, resized_dim), Image.ANTIALIAS)
    
    # Save the image based on its your image or other image
    img.save(DATA_FIXED_SIZE + img_size + which_dir  + '/' + image_name)
    
    return img



def read_mixed_width_and_height_images(resized_dim=100,  img_size="_100_100_3/"):
    '''
    The function used to loop over our images, and other images, which have different sizes, then call the method,
    save fixed size image to resize these images to have all the same height and width, to use later in our model, 
    this because any model need specific number of input at the end.
    '''
    try:
        # Get all directions inside this direction actually 2 direction me and other
        dirs                          = os.listdir(DATA_MIXED_SIZE)
        # loop over these direction contain our images and other images
        for dir_path in dirs:
            # Get all images inside the direction
            images      = os.listdir(DATA_MIXED_SIZE + dir_path)
            # Loop over these images to resize each of them then save these resized ones.
            for image_name in images:
                # First get image full path
                open_image    =  DATA_MIXED_SIZE + dir_path + '/' + image_name
                # Call the method that resize and save the image
                save_fixed_width_and_height_image(open_image, dir_path, image_name, resized_dim, img_size)
                
    # In case of have error throw it into some files in logs direction
    except Exception as e:
        print(e)
        file = open("logs/direction_and_file_handleing_file.log","+a")
        file.write("This error related to function read_mixed_size_images of file Predict Me \n"
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines
    return True



def one_image_reshape_scale(fixed_resized_image_path):
    '''
    The function used to take fixed resized image, then reshape into vectorized version, 
    instead of shape (resize width , resize height , 3) it will be (resize_width*resize_width*3, 1) as new shape.
    '''
    # Open the image
    image_3d_0_255 = mpimg.imread(fixed_resized_image_path)
    # reshape it into (resize_width*resize_width*3, 1)
    image_0_255_vector = image_3d_0_255.reshape(-1, 1)
    # scale it as it 3d
    image_3d_0_1 = scale_image(image_3d_0_255)
    
    # scale it after reshape
    image_0_1_vector = image_3d_0_1.reshape(-1, 1)
    
    return image_0_1_vector, image_0_255_vector


def load_reshape_scale_all_resized_images(dir_path, resized_dim=100, img_size="_100_100_3/"):
    '''
    The images we have save are all of fixed size, then we need to load these images,
    but we need to first scale it into 0-1 pixel intensity, reshape(vecotrize all images),
    built 2d matrix, which is of shape:  number of images * number of features(width*height*3).
    '''
    try:
        image_features = resized_dim * resized_dim * 3
        # Get all saved images from dir_path inside fixed_sized dierction
        images      = os.listdir(DATA_FIXED_SIZE + img_size + dir_path)
        
        # Get number of images inside this direction
        dir_m_data = len(images)
        
        # Create 2d matrix of number of images we have and number of features
        all_images = np.zeros((dir_m_data, image_features))
        
        # save each image in one row 
        indx = 0
        
        # loop over these images
        for image_name in images:
            # Read the image using its full path
            fixed_resized_image_path = DATA_FIXED_SIZE + img_size + dir_path + '/' + image_name
            
            # reshape and scale the image
            image_0_1_vector, image_0_255_vector = one_image_reshape_scale(fixed_resized_image_path)
            # if some image goes to the fixed image direction wrongly skip them
            if image_0_1_vector.shape[0] ==  image_features:
                all_images[indx] = image_0_1_vector.reshape(-1)
            # To save next image in next row
            indx +=1
    # In case of have error throw it into some files in logs direction
    except Exception as e:
        print("Errrrrrrrrrrror")
        print(e)
        file = open("logs/direction_and_file_handleing_file.log","+a")
        file.write("This error related to function load_reshape_scale_all_resized_images of file Predict Me \n"
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines
    return all_images, dir_m_data



def images_pipeline(resized_dim=100,  img_size="_100_100_3/"):
    '''
    The function used to put all of the work into one place, and return one dataframe at the end, 
    this dataframe contain all of your images vectorized with labels.
    '''


    # Second reshape 3_d images into vector(flatten) and rescale pixel intensity
    print("="*50)
    print(resized_dim)
    load_vectorized_images_of_me_images, dir_m_me       = load_reshape_scale_all_resized_images("me", 
                                                            resized_dim,  img_size)
    load_vectorized_images_of_other_images, dir_m_other = load_reshape_scale_all_resized_images("other", 
                                                            resized_dim,  img_size)

    # Third convert numpy 2d array into dataframe
    df_load_vectorized_images_of_me_images               = pd.DataFrame(load_vectorized_images_of_me_images)
    df_load_vectorized_images_of_other_images            = pd.DataFrame(load_vectorized_images_of_other_images)

    # Label images of me as 1 and other as 0
    df_load_vectorized_images_of_me_images['class']      = 1 
    df_load_vectorized_images_of_other_images['class']   = 0
    
    print("I have " + str(len(df_load_vectorized_images_of_me_images)) + " image of me")
    print("="*50)
    print("I have " + str(len(df_load_vectorized_images_of_other_images)) + " image of others")
    
    # combine the two dataframe into one list
    frames                                               = [df_load_vectorized_images_of_me_images , 
                                                            df_load_vectorized_images_of_other_images]

    # concat the frames inside list into dataframe
    df_all_images                                        = pd.concat(frames)
    
    # Shuffle the images
    df_all_images = df_all_images.sample(frac=1).reset_index(drop=True)
    
    print("Now total number of images is " + str(len(df_all_images)))
    
    return df_all_images





def load_train_validation_set(df_all_images):
    '''
    The function used to get the data into train and validation set.
    '''
    # Take y outside the dataframe after shuffle
    y = df_all_images['class']
    y = np.array(y)
    y = y.reshape(1, -1)
    print("The shape of y is: ", y.shape)
    
    # Then drop and return the dataframe with out labels (y)
    df_all_images = df_all_images.drop(['class'], axis=1)
    
    # Take part for training and other for validation
    X_train, X_val, y_train, y_val = df_all_images.iloc[:600], df_all_images.iloc[600:], y[:,:600], y[:,600:]
    print("===================Before===================")
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    # Convert dataframe into numpy array then transpose into features * instances(n*m)
    X_train, X_val = np.array(X_train).T, np.array(X_val).T # T for transpose
    
    print("===================After===================")
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    
    return X_train, X_val, y_train, y_val





def tensor_logs_dir(TENSOR_DIR, run_hyper_params):
    '''
    The function used to create dierction with the time we have run the model in, beside of that,
    concat to this time which hyperparameters we have used in this run, this time along with hyperparameters, 
    will help us compare result from different run with different hyperparamters, 
    as we used the tensorboard server as our vislization tool to help decide which model we can use.
    
    Argument:
    TENSOR_DIR: the tensor logs direction to be our direction for different runs.
    run_hyper_params: which hyper params we have used for this run.
    return
    TENSOR_DIR + run id(which run along with hyperparams to create subdirectory for)
    '''
    
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S_" + run_hyper_params)
    return os.path.join(TENSOR_DIR, run_id)


def model_hyper_parameters(img_size, api_type, n_hidden, learning_rate, epochs):

    learning_rate        = "_lr=" + str(learning_rate)
    num_of_epochs        = "_epochs=" + str(epochs)
    num_of_hidden_layers = "_num of hidden lr=" + str(n_hidden)
    
    params               = img_size + api_type + learning_rate + num_of_epochs + num_of_hidden_layers
    

    return params



def image_display(image):
    image = mpimg.imread(image)
    imgplot = plt.imshow(image)
    return True


def init_2d_graphs(*colors):
    '''
        Just graph initialize in good way
    '''
    plt.style.use(colors) # color of your 2d graph
    return True

init_2d_graphs('ggplot', 'dark_background' )



def display_some_images(X_train, resized_dim=100):

    # X.shape[0] to get number of images we have, then take size of 20 image
    random_indices = np.random.randint(X_train.shape[1], size=20)

    # 50 image with 30000 for each of them
    random_digits_rows = X_train[:, random_indices]
    random_digits_rows = random_digits_rows.T

    # Reshape
    random_images_rows = random_digits_rows.reshape(20, resized_dim, resized_dim, 3)

    # Display the images
    fig, axes = plt.subplots(5,4, figsize=(12,8))

    for i,ax in enumerate(axes.flat):
        ax.imshow(random_images_rows[i])
        ax.axis("off")



def get_losses(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    return acc, val_acc, loss, val_loss

def graph_1(acc, loss, val_loss):

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def graph_2(acc, val_acc):  
    plt.clf()   # clear figure
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()