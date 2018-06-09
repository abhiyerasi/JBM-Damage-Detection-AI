
# coding: utf-8
# Predict the Image classes Using the weightes stored in the folder

# Import the Required Libraries
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
from keras.preprocessing import image


# Basic Settings
img_width, img_height = 150, 150
model_path = 'model.h5'
model_weights_path = 'modelConv2d.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


# Rescale the Image and store it in a folder
test_datagen = ImageDataGenerator(rescale=1./255)


test_set = test_datagen.flow_from_directory(
    './test-data/test_61326',
    target_size=(150, 150),
    batch_size=16,
    class_mode=None,
shuffle=False)


# List all the files in the directory
list_file=os.listdir("./test-data/test_61326/")



##Predicting using test data
for i in list_file:
    stri = str(i)
    test_image = image.load_img('./test-data/test_61326/' + stri, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    result = model.predict_classes(test_image,verbose=1) # Predicts the class labels
    
    # Convert the predicted class into label and preidct it
    if result==0:
        print('61326_ok_back')
    elif result==1:
        print('61326_ok_front')
    elif result==2:
        print('61326_scratch_mark')
    elif result==3:
        print('61326_slot_damage')
    elif result==4:
        print('61326_thinning')
    elif result==5:
        print('61326_wrinkle')
    else:
        print('none')

