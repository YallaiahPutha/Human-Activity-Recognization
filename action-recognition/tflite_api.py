import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
import uuid
import os
import tensorflow.lite as tflite
import time
from collections import deque
# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='tflite_models/model_to1.tflite')
#allocate the tensors
interpreter.allocate_tensors()

# Checking the input and output details from model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
SEQUENCE_LENGTH = 30
# Image size of each frame will resize
Image_height,Image_width = 224, 224

# Specify the Classes list
CLASSES_LIST = ["Falling","Loitering","Voilence"]

app = FastAPI()


def tflite_model_prediction(input_video_path, SEQUENCE_LENGTH):
    #     start = time.time()
    # Capturing the video from the input
    video_reader = cv2.VideoCapture(input_video_path)
    # Getting the frames and make as a queue with sequence length
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    list_prediction = []
    # we are checking the condition for video
    while video_reader.isOpened():
        # read the frames from video
        ok, frame = video_reader.read()

        if not ok:
            break

        # resizing the frames
        resized_frame = cv2.resize(frame, (Image_height, Image_width))
        # Normalizing the frames in the range of 0 to 1
        normalized_frame = resized_frame / 255
        # appending the normalize frames into frames_queue
        frames_queue.append(normalized_frame)
        # converting to frames_queue from float64 to float32 to support the model
        frames_que = np.float32(frames_queue)

        if len(frames_que) == SEQUENCE_LENGTH:
            # Passing the all input data with dimensions like X for model
            input_tensor = np.array(np.expand_dims(frames_que, 0))
            # getting the input indexes like y for model
            input_index = interpreter.get_input_details()[0]['index']
            # Both input_tensor and input_index passing through the model
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            # getting the output details
            output_details = interpreter.get_output_details()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            # Getting prediction probabilities of the output
            pred = np.squeeze(output_data)
            # Getting highest probabilities of the value as our output and index
            predict_label = np.argmax(pred)
            # Getting the class label based on the index
            predicted_class_name = CLASSES_LIST[predict_label]

            #             end = time.time()
            #             print("$$$$$$$$$$$$$$$$$$$$$$$$$$$",end-start)

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

    prediction = tflite_model_prediction(abs_file_path,SEQUENCE_LENGTH)
    os.remove(abs_file_path)

    return prediction
    # return {'success': True}





if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=12201, debug=True)