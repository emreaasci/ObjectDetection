import cv2

original_video = cv2.VideoCapture("/Users/emreasci/Downloads/CarParkProject/carPark.mp4")

lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

_, prev_frame = original_video.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

frame_width = int(original_video.get(3))
frame_height = int(original_video.get(4))
stabilized_out = cv2.VideoWriter('stabilized_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                 (frame_width, frame_height))

while True:
    ret, original_frame = original_video.read()
    if not ret:
        break
    original_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original Video', original_frame)

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, original_frame_gray, prev_pts, None, **lk_params)

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    m = cv2.estimateAffinePartial2D(good_old, good_new)

    stabilized_frame = cv2.warpAffine(original_frame, m[0], (original_frame.shape[1], original_frame.shape[0]))

    cv2.imshow('Stabilized Video', stabilized_frame)

    stabilized_out.write(stabilized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = original_frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

original_video.release()
stabilized_out.release()
cv2.destroyAllWindows()
