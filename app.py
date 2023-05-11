from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, flash
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from datetime import datetime
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import cv2
import time
import pathlib

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import tensorflow as tf

from skimage.io import imread, imshow
from skimage.transform import resize


app = Flask(__name__)

app.secret_key = "pgc@METIS"

today = datetime.now()
folder_name = today.strftime("%b-%d-%Y-%H-%M-%S")
uploads_dir = os.path.join(app.instance_path, 'uploads', folder_name)


os.makedirs(uploads_dir, exist_ok=True)
dest=os.path.join('static',folder_name)
pred_folder=os.path.join(dest,"PredictionMasks")
pred_list=[]

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_model():
    start = time.time()
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    end=time.time()
    print("Time taken to load the model:",(end-start)/60)
    return model

def predictions(model):
    image_id_list=os.listdir(uploads_dir)
    image_id_list.sort()
    start=time.time()
    X_test = np.zeros((len(image_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for i, image_id in enumerate(image_id_list):
    
        path_image = os.path.join(uploads_dir,image_id)
        # read the image using skimage
        image = imread(path_image)[:,:,:IMG_CHANNELS]
        
        # resize the image
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[i]=image

    end=time.time()
    print("Time taken to read the images:",(end-start)/60)

    model.load_weights('model7.h5')
    y_pred=model.predict(X_test)
    y_pred_thresholded = (y_pred >= 0.5).astype(np.uint8)
    scratch_list=[]
    non_scratch_list=[]
    start=time.time()
    for x in range(len(y_pred_thresholded)):
        if True in y_pred_thresholded[x].flatten():
            scratch_list.append(image_id_list[x])
            test_mask = y_pred_thresholded[x, :, :, 0]*255
            test_mask=cv2.cvtColor(test_mask,cv2.COLOR_GRAY2RGB)
            test_mask = resize(test_mask, (400, 400), mode='constant', preserve_range=True)
            img = os.path.join(uploads_dir,image_id_list[x])
            img=cv2.imread(img)
            img = resize(img, (400, 400), mode='constant', preserve_range=True)
            h_img = cv2.hconcat([img, test_mask])
            cv2.imwrite(os.path.join(pred_folder,image_id_list[x]),h_img)
        else:
            non_scratch_list.append(image_id_list[x])
    end=time.time()
    print("Time taken to combine original image and prediction mask:",(end-start)/60)
    print(scratch_list)
    return scratch_list,non_scratch_list

def createzip():
    examplezip = zipfile.ZipFile(
        'static/'+folder_name+'/labels/labels.zip', 'w')
    for x in os.listdir('static/'+folder_name+'/labels/'):
        examplezip.write('static/'+folder_name+'/labels/'+x,
                         compress_type=zipfile.ZIP_DEFLATED)
    examplezip.close()

def labels(path):
    glue = []
    broken = []
    foobar = []
    one_wire = []
    two_wire = []
    all_wire = []
    good = []

    for x in os.listdir(path):
        image_name = x.replace(".txt", ".jpg")
        file1 = open(path+x, 'r')
        Lines = file1.readlines()
        category = [int(line.strip()[0]) for line in Lines]
        if(1 not in category and 2 not in category and 3 not in category):
            good.append(image_name)
        else:
            if 1 in category:
                broken.append(image_name)
            if 2 in category:
                glue.append(image_name)
            if 3 in category:
                foobar.append(image_name)
        if category.count(0) == 1:
            one_wire.append(image_name)
        elif category.count(0) == 2:
            two_wire.append(image_name)
        elif category.count(0) == 3:
            all_wire.append(image_name)
    return glue, broken, foobar, one_wire, two_wire, all_wire, good

@app.route("/yolo/")
def yolo_uploader():
    global today
    global folder_name
    global uploads_dir
    if folder_name in os.listdir('static/'):
        print("Inside Loop")
        path = 'static/'+folder_name
        # Sorting as per image upload date and time
        uploads = sorted(os.listdir(path))
        # uploads = os.listdir('static/uploads')
        uploads = [folder_name+"/" + file for file in uploads]
        gluelist, brokenlist, foobarlist, one_wirelist, two_wirelist, all_wirelist, goodlist = labels(
            'static/'+folder_name+'/labels/')
        gluelist3, brokenlist3, foobarlist3, one_wirelist3, two_wirelist3, all_wirelist3, goodlist3 = labels(
            'database/Module3/labels/')
        gluelist4, brokenlist4, foobarlist4, one_wirelist4, two_wirelist4, all_wirelist4, goodlist4 = labels(
            'database/Module4/labels/')
        gluelist5, brokenlist5, foobarlist5, one_wirelist5, two_wirelist5, all_wirelist5, goodlist5 = labels(
            'database/Module5/labels/')
        gluelist6, brokenlist6, foobarlist6, one_wirelist6, two_wirelist6, all_wirelist6, goodlist6 = labels(
            'database/Module6/labels/')
        glue = [folder_name+"/" + file for file in gluelist]
        broken = [folder_name+"/" + file for file in brokenlist]
        foobar = [folder_name+"/" + file for file in foobarlist]
        one_wire = [folder_name+"/" + file for file in one_wirelist]
        two_wire = [folder_name+"/" + file for file in two_wirelist]
        all_wire = [folder_name+"/" + file for file in all_wirelist]
        good = [folder_name+"/" + file for file in goodlist]
        labelnames = [folder_name+"/labels/labels.zip"]
        createzip()
        
        today = datetime.now()
        folder_name = today.strftime("%b-%d-%Y-%H-%M-%S")
        uploads_dir = os.path.join(app.instance_path, 'uploads', folder_name)
        
        
        os.makedirs(uploads_dir, exist_ok=True)

        return render_template("yolo.html", uploads=uploads, glue=glue, broken=broken, foobar=foobar,
                               one_wire=one_wire, two_wire=two_wire, all_wire=all_wire, good=good, labelnames=labelnames, gluelist=gluelist, brokenlist=brokenlist, foobarlist=foobarlist, one_wirelist=one_wirelist, two_wirelist=two_wirelist, all_wirelist=all_wirelist, goodlist=goodlist)
    else:
        return render_template('yolo.html')
    

    
@app.route("/unet/")
def unet_uploader():
    global uploads_dir
    global dest
    global pred_folder
    global today
    global folder_name
    global pred_list
    
    if "/yolo" in request.referrer:
        today = datetime.now()
        folder_name = today.strftime("%b-%d-%Y-%H-%M-%S")
        uploads_dir = os.path.join(app.instance_path, 'uploads', folder_name)
        os.makedirs(uploads_dir, exist_ok=True)
        dest=os.path.join('static',folder_name)
        pred_folder=os.path.join(dest,"PredictionMasks")
        return render_template('unet.html')

    if len(os.listdir(uploads_dir))!=0:
        # folder_name=max(pathlib.Path(os.path.join(app.instance_path, 'uploads')).glob('*/'), key=os.path.getmtime)
        # folder_name=str(folder_name).split('/')[-1]
        # uploads_dir = os.path.join(app.instance_path, 'uploads', folder_name)
        # print(folder_name)
        print("Inside Loop")
        start=time.time()
        destination = shutil.copytree(uploads_dir, dest,dirs_exist_ok=True)
        end=time.time()
        print("Time taken to copy the folder to static:",(end-start)/60)
        os.makedirs(pred_folder,exist_ok=True)
        uploads=os.listdir(uploads_dir)
        model=load_model()
        scratch,nonscratch=predictions(model)
        uploads_list=[folder_name+"/" + file for file in uploads]
        scratch_list=[folder_name+"/" + file for file in scratch]
        nonscratch_list=[folder_name+"/" + file for file in nonscratch]
        pred_list=os.listdir(pred_folder)
        pred_list=[folder_name+"/PredictionMasks/"+file for file in pred_list]
        today = datetime.now()
        folder_name = today.strftime("%b-%d-%Y-%H-%M-%S")
        uploads_dir = os.path.join(app.instance_path, 'uploads', folder_name)
        
        
        os.makedirs(uploads_dir, exist_ok=True)
        dest=os.path.join('static',folder_name)
        pred_folder=os.path.join(dest,"PredictionMasks")
        return render_template("unet.html",uploads=uploads,scratch=scratch,nonscratch=nonscratch
                               ,uploads_list=uploads_list,scratch_list=scratch_list,nonscratch_list=nonscratch_list)
    else:
        return render_template('unet.html')


@app.route("/upload", methods=["POST", "GET"])
def unet_upload():
    if request.method == "POST":
        files = request.files.getlist('files[]')
        print(files)
        for f in files:
            filename = secure_filename(f.filename)
            f.save(os.path.join(uploads_dir, filename))
            #print(filename)
        flash('File(s) successfully uploaded')
    if "/unet" in request.referrer:
        return redirect(request.referrer)
    else:
        return detect()


@app.route("/unet/predictionsmask")
def predictionsmasks():
    global pred_list
    return render_template('predictions.html',scratch_list=pred_list)

@app.route("/detect")
def detect():
    subprocess.run("ls")
    print(uploads_dir)
    subprocess.run(['python3', 'detect.py', '--weights', 'updated.pt', '--source', uploads_dir, '--img', '640',
                   '--conf', '0.5', '--project', 'static', '--name', folder_name, '--save-txt'])
    return redirect(request.referrer)

@app.route("/")
def home():
    return render_template('index.html')