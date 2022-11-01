import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
import uuid
import os
from collections import deque
import time


from keras.models import load_model
restored_model = load_model('tensorflow_model/Mymodel_tf_original/1/')

SEQUENCE_LENGTH = 30
# Image size of each frame will resize
Image_height,Image_width = 224, 224

# Specify the Classes list
CLASSES_LIST = ["Falling","Loitering","Voilence"]

app = FastAPI()



def predict_on_video(video_file_path, SEQUENCE_LENGTH):
    # start = time.time()
    video_reader = cv2.VideoCapture(video_file_path)


    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    list_prediction = []
    while video_reader.isOpened():
        ok, frame = video_reader.read()

        if not ok:
            break

        resized_frame = cv2.resize(frame, (Image_height, Image_width))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = restored_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predict_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predict_label]
            # HRA

            list_prediction.append(predicted_class_name)




    video_reader.release()
    return list_prediction


@app.post("/predict/")
async def predict_api(file: UploadFile=File(...) ):

    abs_file_path = r'C:\Users\NH1088\Downloads\{}_{}'.format(uuid.uuid1(), file.filename)

    with open(abs_file_path, 'wb') as f:
        # print("$$$$$$$$$$$$$$$$$$",f)
        f.write(file.file.read())
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&",f)

    # frames = read_video(abs_file_path, SEQUENCE_LENGTH)
    # prediction = predict_model(frames)

    prediction = predict_on_video(abs_file_path,SEQUENCE_LENGTH)
    os.remove(abs_file_path)

    return prediction
    # return {'success': True}





if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=12200, debug=True)