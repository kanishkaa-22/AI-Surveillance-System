"""
Run this on any system (no GPU needed) to find the pixel coordinates
of your zone line or restricted-area rectangle.

Usage:
    python backend/get_zone_coords.py path/to/zone_crossing_01.mp4

Click on the image that opens:
  - For a LINE zone: click the two endpoints of your tape line.
  - For a RECTANGLE zone: click the four corners.
Coordinates print to the terminal as you click. Press any key to close once done.
"""

import sys
import cv2

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_zone_coords.py data/detection_tracking_clips/zone_crossing_01.mp4")
        return

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read a frame from that video.")
        return

    clicked_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            print(f"Clicked at: x={x}, y={y}")
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(clicked_points) > 1:
                cv2.line(frame, clicked_points[-2], clicked_points[-1], (0, 255, 0), 2)
            cv2.imshow("Click zone points - press any key when done", frame)

    cv2.imshow("Click zone points - press any key when done", frame)
    cv2.setMouseCallback("Click zone points - press any key when done", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nAll clicked points:", clicked_points)
    print("\nPaste these into backend/zone_and_trigger.py:")
    print(f"ZONE_POINTS = {clicked_points}")


if __name__ == "__main__":
    main()