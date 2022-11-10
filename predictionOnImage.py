import os
import json
from keras.models import load_model
import pickle
import numpy as np
import cv2
import io


THU_MUC_GOC = os.getcwd()
# BASE_DIR = "D:\Code\Project\DistractedDriverDetection"
PICKLE_DIR = os.path.join(THU_MUC_GOC, "pickle_files")
BEST_MODEL = os.path.join(THU_MUC_GOC, "distracted.hdf5")
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR, "labels_list.pkl"), "rb") as handle:
    labels_id = pickle.load(handle)


def path_to_tensor(img_path):
    file_bytes = np.asarray(bytearray(img_path.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    x = cv2.resize(img, (128, 128))
    return np.expand_dims(x, axis=0)


def return_prediction(filename):
    buffer = io.BytesIO()
    filename.save(buffer, 'jpeg')
    buffer.seek(0)
    filename = buffer
    test_tensors = path_to_tensor(filename).astype('float32') / 255

    ypred_test = model.predict(test_tensors, verbose=1)
    ypred_class = np.argmax(ypred_test, axis=1)

    print(ypred_class)
    id_labels = dict()
    for class_name, idx in labels_id.items():
        id_labels[idx] = class_name
    print(id_labels)
    ypred_class = int(ypred_class)
    res = id_labels[ypred_class]

    with open(os.path.join(os.getcwd(), 'class_name_map.json')) as secret_input:
        info = json.load(secret_input)
    prediction_result = info[res]
    return prediction_result


if __name__ == '__main__':
    pass
