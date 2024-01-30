import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import io
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# Function to process TIFF files
padding_globalTEMP=[]
padding_global=np.array(padding_globalTEMP)

snippet=128
def extract_snippets(image, snippet_size=(snippet, snippet)):
    snippets = []
    height=image.shape[0]
    width=image.shape[1]
    
    for i in range(0, height, snippet_size[0]):
        for j in range(0, width, snippet_size[1]):
            snippet = image[i:i+snippet_size[0], j:j+snippet_size[1],:]
            snippets.append(snippet)
    return snippets

def process_tiff_files(tiff_files,model):
    bands=[ 'B2.tif', 'B3.tif', 'B4.tif', 'B8.tif']
    # folder_path = "/kaggle/input/sih-dataset-2/multiclass"
    # all_loc = os.listdir(folder_path)
    x_train=np.zeros(shape=(0,snippet,snippet,len(bands)))
    # y_train=np.zeros(shape=(0,snippet,snippet))
    # fold='loc4'
    # temp=folder_path+"/"+fold
    # tif_files = [file for file in os.listdir(temp) if file.endswith('.tif')]
    # tif_files=[ 'B2.tif', 'B3.tif', 'B4.tif', 'B8.tif']
    # print(tif_files)
    processed_images = []
    # x_train =[]
    # x_train.shape=(0,0,0)
    for uploaded_file in tiff_files:
        image = Image.open(uploaded_file)
        processed_images.append(np.array(image))

    data=np.zeros((processed_images[0].shape[0],processed_images[0].shape[1],len(bands)))
    i=0
    for im in processed_images:
        # im = Image.open(temp+"/"+file_name)
        imarray = np.array(im)
        data[:,:,i]=imarray
        i+=1
    data=np.array(data)
    print(data.shape)
    height=data.shape[0]
    width=data.shape[1]
    pad_height = (height // snippet + 1) * snippet - height
    pad_width = (width // snippet + 1) * snippet - width
    pad = np.pad(data, ((0, pad_height), (0, pad_width),(0,0)), mode='constant')
    # padding_global=pad
    print(padding_global)
    tmp1=np.array(extract_snippets(pad))
    x_train=np.concatenate((x_train,tmp1),axis=0)
    x_train/=np.mean(x_train)
    x_train*=1000
    print(x_train.shape)
    ans=model.predict(x_train)
    ans=np.argmax(ans,axis=-1)
    ans=ans.reshape(-1,128,128)
    output=np.zeros(shape=(pad.shape[0],pad.shape[1]))
    ans.shape
    height,width=pad.shape[0:2]
    k=height//128
    l=width//128
    for i in range(0, height, 128):
        for j in range(0, width, 128):
            output[i:i+128, j:j+128]=ans[(i*l)//(128)+(j//128)]
    plt.imshow(output)
    plt.savefig("Image_before.jpeg")
    return output

# Function to load model from H5 and JSON files


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Change the output layer for multi-class classification
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


def load_model(model_path, weights_path):
    # Load JSON model architecture
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    

    # Load model weights
    # loaded_model=build_unet(input_shape=(128,128,4),num_classes=7)
    loaded_model.load_weights(weights_path)
    return loaded_model

# Function to make predictions using the loaded model
# def predict_with_model(model, x_train):
#     # Preprocess images (if needed)
#     # Replace this with your image preprocessing logic
#     ans=model.predict(x_train)
#     ans=np.argmax(ans,axis=-1)
#     ans=ans.reshape(-1,128,128)
#     output=np.zeros(shape=(padding_global.shape[0],padding_global.shape[1]))
#     ans.shape
#     height,width=padding_global.shape[0:2]
#     k=height//128
#     l=width//128
#     for i in range(0, height, 128):
#         for j in range(0, width, 128):
#             output[i:i+128, j:j+128]=ans[(i*l)//(128)+(j//128)]
#     return output

def main():
    
    # st.set_page_config(layout="wide", page_title="Streamlit App", page_icon=":rocket:", 
    #                    initial_sidebar_state="expanded")
    st.title('TIFF Files Processing and Prediction with Pre-trained Model')
    st.write("Upload 4 TIFF files")


    tiff_files = []
    tiff_files2 = []

    tiff_files = st.file_uploader("Upload 4 TIFF Images", accept_multiple_files=True, type=["tif", "tiff"],key="file_uploader_1")
    tiff_files2 = st.file_uploader("Upload 4 TIFF Images", accept_multiple_files=True, type=["tif", "tiff"],key="file_uploader_2")
    processed_images1=[]
    processed_images2=[]
    # print(tiff_files)
    if len(tiff_files) == 4:
        st.write("TIFF files uploaded successfully!")
        st.title('Display 4 TIF Files')
        for uploaded_file in tiff_files:
            image = Image.open(uploaded_file)
            processed_images1.append(np.array(image))
        if (len(processed_images1)==4):
            composite_image = np.stack([processed_images1[0], processed_images1[1], processed_images1[2]], axis=-1)
        normalized_image = composite_image / np.max(composite_image)
        # st.image(normalized_image, caption='Composite Image', vmin=0.0,vmax=1.0)

    if len(tiff_files2) == 4:
        st.write("TIFF files uploaded successfully!")
        st.title('Display 4 TIF Files')
        for uploaded_file in tiff_files2:
            image = Image.open(uploaded_file)
            processed_images2.append(np.array(image))
        if (len(processed_images2)==4):
            composite_image = np.stack([processed_images2[0], processed_images2[1], processed_images2[2]], axis=-1)
        normalized_image = composite_image / np.max(composite_image)
        print(composite_image.shape)
        st.image(normalized_image, caption='Composite Image',use_column_width=True)
                
    model_path = 'model_architecture.json'
    weights_path = 'best_weight.h5'
    
    loaded_model = load_model(model_path, weights_path)
    if len(processed_images1)!=0:
        predictions1=np.zeros(shape=(processed_images1[0].shape))
    if len(processed_images2)!=0:
        predictions2=np.zeros(shape=(processed_images2[0].shape))

    if len(tiff_files) == 4 and len(tiff_files) == 4:
        predictions1 = process_tiff_files(tiff_files,loaded_model)
        # predictions1 = predict_with_model(loaded_model, x_train)
        predictions2 = process_tiff_files(tiff_files,loaded_model)
        # predictions2 = predict_with_model(loaded_model, x_train)
        npimg1 = np.array(predictions1)
        npimg2 = np.array(predictions2)
        array_shape = npimg1.shape
        forward = np.zeros(array_shape)
        backward = np.zeros(array_shape)
        foward=forward.astype(int)
        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                if npimg1[i, j] != npimg2[i, j]:
                    forward[i, j] = npimg2[i, j]
                    backward[i, j] = npimg1[i, j]
        # st.image(predictions1, caption='Composite Image',use_column_width=True)
        # st.download_button("Mask1", predictions1, "Mask1")
        # st.download_button("Mask2", predictions2, "Mask2")
        # st.download_button("Augmentation", forward, "Augmentation")
        # st.download_button("Depletion", backward, "Depletion")
        st.image(predictions1/np.max(predictions1), caption='Mask 1')
        st.image(predictions2/np.max(predictions2), caption='Mask 2')
        st.image(forward/np.max(forward), caption='Augmentation')
        st.image(backward/np.max(backward), caption='Depletion')

    ###foward,backward,predictions1,predictions2...

if __name__ == '__main__':
        main()