import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from line_boundary_check import *

# ----------------------------------------------------------------------------

g_mouse_pos      = [0 ,0]

# Mouse event handler
def onMouse(event, x, y, flags, param):
    global g_mouse_pos
    g_mouse_pos = [x, y]

# ----------------------------------------------------------------------------

class boundaryLine:
    def __init__(self, line=(0,0,0,0)):
        self.p0 = (line[0], line[1])
        self.p1 = (line[2], line[3])
        self.color = (0,255,255)
        self.lineThinkness = 4
        self.textColor = (0,255,255)
        self.textSize = 4
        self.textThinkness = 2
        self.count1 = 0
        self.count2 = 0

# Draw single boundary line
def drawBoundaryLine(img, line):
    x1, y1 = line.p0
    x2, y2 = line.p1
    cv2.line(img, (x1, y1), (x2, y2), line.color, line.lineThinkness)
    cv2.putText(img, str(line.count1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.putText(img, str(line.count2), (x2, y2), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.drawMarker(img, (x1, y1),line.color, cv2.MARKER_TRIANGLE_UP, 16, 4)
    cv2.drawMarker(img, (x2, y2),line.color, cv2.MARKER_TILTED_CROSS, 16, 4)

# Draw multiple boundary lines
def drawBoundaryLines(img, boundaryLines):
    for line in boundaryLines:
        drawBoundaryLine(img, line)

# in: boundary_line = boundaryLine class object
#     trajectory   = (x1, y1, x2, y2)
def checkLineCross(boundary_line, trajectory_line):
    global audio_enable_flag
    global sound_welcome, sound_thankyou
    traj_p0  = trajectory_line[0]                                       # Trajectory of an object
    traj_p1  = trajectory_line[1]
    bLine_p0 = (boundary_line.p0[0], boundary_line.p0[1])               # Boundary line
    bLine_p1 = (boundary_line.p1[0], boundary_line.p1[1])
    intersect = checkIntersect(traj_p0, traj_p1, bLine_p0, bLine_p1)    # Check if intersect or not
    if intersect == True:
        angle = calcVectorAngle(traj_p0, traj_p1, bLine_p0, bLine_p1)   # Calculate angle between trajectory and boundary line
        if angle<180:
            boundary_line.count1 += 1
        else:
            boundary_line.count2 += 1
        #cx, cy = calcIntersectPoint(traj_p0, traj_p1, bLine_p0, bLine_p1) # Calculate the intersect coordination

#------------------------------------
# Area intrusion detection
class area:
    def __init__(self, contour):
        self.contour  = np.array(contour, dtype=np.int32)
        self.count    = 0

# Draw areas (polygons)
def drawAreas(img, areas):
    for area in areas:
        if area.count>0:
            color=(0,0,255)
        else:
            color=(255,0,0)
        cv2.polylines(img, [area.contour], True, color,4)
        cv2.putText(img, str(area.count), (area.contour[0][0], area.contour[0][1]), cv2.FONT_HERSHEY_PLAIN, 4, color, 2)

# Area intrusion check
def checkAreaIntrusion(area, points):
    global audio_enable_flag
    global sound_warning
    area.count = 0
    for pt in points:
        if pointPolygonTest(area.contour, pt):
            print("intrusion")
            area.count += 1


# ----------------------------------------------------------------------------

# boundary lines
boundaryLines = [
    boundaryLine([300, 0, 150, 350]),
    boundaryLine([640, 0, 640, 350])
]

# Areas
areas = [
    #      up left, up right, up middle, bottom right, bottom left
    area([ [300,0], [640,0], [640,352], [300,352], [300,352] ])
]

def main():
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture('me.mp4')
    pTime = 0

    cv2.namedWindow('test')
    cv2.setMouseCallback('test', onMouse)
    prev_mouse_pos = [0, 0]
    trace = []
    trace_length = 25

    #video_width = frame1.shape[1]
    #video_hight = frame1.shape[0]

    key = -1
    credentials = []
    while True:
        success, image = cap.read()
        image.shape[1]

        frame_edge = cv2.GaussianBlur(image, (7, 7), 1.41)
        frame_edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edge = cv2.Canny(frame_edge, 25, 75)

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        #print(results.pose_landmarks)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                g_mouse_pos = [cx, cy]
                #print(g_mouse_pos)
                for line in boundaryLines:
                    checkLineCross(line, (prev_mouse_pos, g_mouse_pos))
                for area in areas:
                    checkAreaIntrusion(area, (g_mouse_pos,))
                drawAreas(image, areas)


            #img = np.zeros((600, 800, 3), dtype=np.uint8)
            #for line in boundaryLines:
            #    g_mouse_pos = [lm.x, lm.y]
            #    print(g_mouse_pos)
            #    checkLineCross(line, (prev_mouse_pos, g_mouse_pos))
            #for area in areas:
            #    checkAreaIntrusion(area, (g_mouse_pos,))
            #drawAreas(image, areas)

            #trace.append(g_mouse_pos)
            #if len(trace) > trace_length:
            #    trace = trace[-trace_length:]
            #cv2.polylines(image, np.array([trace], dtype=np.int32), False, (255, 255, 0), 1, cv2.LINE_AA)
            #print(image.shape[1] = 640, image.shape[0] = 352)
            #cv2.rectangle(image, (150, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
            prev_mouse_pos = [150, 0]
            cv2.imshow('test', image)
            key = cv2.waitKey(50)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #cv2.putText(image, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", image)
        cv2.imshow('Canny Edge', edge)
        cv2.waitKey(1)


if __name__ == '__main__':
    sys.exit(main())
