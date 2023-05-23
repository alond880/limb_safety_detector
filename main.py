import cv2
import os

current_directory = os.getcwd()

# Change the path and filename of your input video with its extension
video_path = os.path.join(current_directory, 'me.mp4')
vs = cv2.VideoCapture(video_path)

output_video = None

while True:

    (frame_exists, frame) = vs.read()

    if not frame_exists:
        break

    else:
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    key = cv2.waitKey(1) & 0xFF

    # Write the both outputs video to a local folders
    if output_video is None:
        fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")

        # Change the path and filename of your output video with .avi extension
        output_video = cv2.VideoWriter(os.path.join(current_directory, 'footage_detected_corrected.avi'), fourcc1, 25,
                                       (new_frame.shape[1], new_frame.shape[0]), True)

    elif output_video is not None:
        output_video.write(new_frame)

    cv2.imshow("video", frame)

    if key == ord("q"):
        break

vs.release()
output_video.release()
cv2.destroyAllWindows()