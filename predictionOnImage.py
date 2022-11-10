import os
import json
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil
import cv2
from keras.preprocessing import image                  
from tqdm.notebook import tqdm
# from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import io


THU_MUC_GOC = os.getcwd()
# BASE_DIR = "D:\Code\Project\DistractedDriverDetection"
PICKLE_DIR = os.path.join(THU_MUC_GOC,"pickle_files")
BEST_MODEL = os.path.join(THU_MUC_GOC,"distracted.hdf5")
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:
    labels_id = pickle.load(handle)

def path_to_tensor(img_path):
    img = load_img(img_path, target_size=(128, 128))
    x = img_to_array(img)
    return np.expand_dims(x, axis=0)

# def paths_to_tensor(img_paths):
    # list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    # return np.vstack(list_of_tensors)

def return_prediction(filename):
    buffer = io.BytesIO()
    filename.save(buffer, 'jpeg')
    buffer.seek(0)
    filename = buffer
    test_tensors = path_to_tensor(filename).astype('float32')/255 - 0.5

    ypred_test = model.predict(test_tensors,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)

    print(ypred_class)
    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    print(id_labels)
    ypred_class = int(ypred_class)
    res = id_labels[ypred_class]

    with open(os.path.join(os.getcwd(),'class_name_map.json')) as secret_input:
        info = json.load(secret_input)
    prediction_result = info[res]
    return prediction_result

if __name__=='__main__':
    pass
