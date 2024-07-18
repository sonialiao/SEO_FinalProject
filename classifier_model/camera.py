import cv2
import numpy as np
import onnxruntime as ort

def main():
    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # create runnable session of the exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")

    # TODO: here code pulls for webcam (check number to make sure it links to the right camera)
    cap = cv2.VideoCapture(1)

    # Continuous capture frame-by-frame
    while True:
        ret, frame = cap.read()

        # preprocess data: crop center and gray-scale to 0-255
        frame = center_crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # reshape & scale into the right format for model 
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std
        x = x.reshape(1, 1, 28, 28).astype(np.float32)

        # run model to predict letter
        y = ort_session.run(None, {'input': x})[0]

        # get model output and convert to letter
        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        # TODO: this displays the characters on screen --> pass letter to frontend page?
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        # exit if the q key is hit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release memory & windows after exiting loop
    cap.release()
    cv2.destroyAllWindows()


# crop to the center of the frame
def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        frame = frame[start: start + w]
    else:
        frame = frame[:, start: start + h]
    return frame

if __name__ == '__main__':
    main()