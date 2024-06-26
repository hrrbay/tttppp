import sys

import cv2
import numpy as np

video_filename = sys.argv[1]
out_filename = sys.argv[2]
print('* 1 \t- Point left\n* 2 \t- Point right\n* 3 \t- Flip last point\n* UP \t- Increase speed\n* DOWN \t- Decrease speed\n*')

# store labels
labels = []

# load video
cap = cv2.VideoCapture(video_filename)
while not cap.isOpened():
    cap = cv2.VideoCapture(video_filename)
    cv2.waitKey(1000)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_num = 0
delay = 1
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break
    cv2.putText(frame, f'Frame {frame_num}/{length} ({frame_num*100/length:.0f}%)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    if len(labels) > 0:
        cv2.putText(frame, f'Last label: {labels[-1][0]}', (frame.shape[1]-300,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('video', frame)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    if key == 82:
        # UP
        delay -= 1
    elif key == 84:
        # DOWN
        delay += 1
    elif key == 49:
        # 1 - point left
        labels.append([0, frame_num])
        print("Point left " + str(frame_num))
    elif key == 50:
        # 2 - point right
        labels.append([1, frame_num])
        print("Point right " + str(frame_num))
    elif key == 51:
        # 3 - flip last label
        print("Flip last label")
        if len(labels) > 0:
            labels[-1][0] = 1 - labels[-1][0]
    elif key == 115:
        # s - start
        labels.append([0, frame_num])
        print("Start " + str(frame_num))
    
    delay = max(delay, 1)
    frame_num += 1

if len(labels) > 0:
    np.savetxt(out_filename, labels, ('%d', '%07d'))
cap.release()
cv2.destroyAllWindows()