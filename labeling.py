import cv2
import sys
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
speed = 5
while cap.isOpened():
    ret, frame = cap.read()
    cv2.putText(frame, f'Frame {frame_num}/{length} ({frame_num*100/length:.0f}%)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    if len(labels) > 0:
        cv2.putText(frame, f'Last label: {labels[-1][0]}', (frame.shape[1]-300,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('video', frame)

    key = cv2.waitKey(speed) & 0xFF
    if key == ord('q'):
        break
    if key == 82:
        # UP
        speed -= 1
    elif key == 84:
        # DOWN
        speed += 1
    elif key == 49:
        # 1 - point left
        labels.append([0, frame_num])
    elif key == 50:
        # 2 - point right
        labels.append([1, frame_num])
    elif key == 51:
        # 3 - flip last label
        if len(labels) > 0:
            labels[-1][0] = 1 - labels[-1][0]
    
    speed = max(speed, 1)
    frame_num += 1

np.savetxt(out_filename, labels, ('%d', '%07d'))
cap.release()
cv2.destroyAllWindows()