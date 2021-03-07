import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox
alpha=0.5

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


vid = cv2.VideoCapture(r'C:\Users\admin\Desktop\sample_traffic_scene.mp4')


while(vid.isOpened()):
    status, frame = vid.read()
    #Rescaling the output image as its in high resolution
    frame=rescale_frame(frame,60)
    print(frame.shape)

    overlay = frame.copy()
    output = frame.copy()

    # roi = cv2.selectROI(output)
    # roi_cropped = output[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    # print(roi_cropped)
    # cv2.imshow('ROI',roi_cropped)

    cv2.rectangle(overlay, (406,421), (862, 543),(0, 0, 255), -1)

    # cv2.line(overlay,(422,394),(328,579),(0, 0, 255),5)
    # cv2.line(overlay, (422, 394), (726, 384), (0, 0, 255), 5)
    # cv2.line(overlay, (328, 579), (882, 555), (0, 0, 255), 5)
    # cv2.line(overlay, (882, 555), (726, 384), (0, 0, 255), 5)


    #pts = np.array([(422, 394), (882, 555), (328, 579),(726, 384) ], np.int32)
    #cv2.polylines(overlay, [pts], True, (0, 255, 255), 3)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)

    if not status:
        break

    # applying object detection
    bbox, label, conf = cv.detect_common_objects(output, confidence=0.80, model='yolov4-tiny')

    #print(bbox, label, conf)

    # draw bounding box over detected objects
    out = draw_bbox(output, bbox, label, conf, write_conf=True)

    cv2.imshow("Real-time object detection", out)


    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

