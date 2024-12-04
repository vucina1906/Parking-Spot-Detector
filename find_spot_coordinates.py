import cv2
import numpy as np

parking_spots = []  # List to store all parking spots
current_spot = []  # Temporarily stores 4 points for one parking spot
spot_count = 1  # Counter for spot naming


def select_points(event, x, y, flags, param):
    global current_spot, parking_spots, spot_count
    frame = param[0]  
    if event == cv2.EVENT_LBUTTONDOWN:  
        current_spot.append((x, y))  
        print(f"Point selected: {x, y}")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  

        if len(current_spot) == 4:  
            parking_spots.append((f"spot{spot_count}", current_spot))
            print(f"Saved spot{spot_count}: {current_spot}")
            current_spot = []  
            spot_count += 1  


cap = cv2.VideoCapture('camera.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

cv2.namedWindow('Video')
ret, frame = cap.read()
frame = cv2.resize(frame, (1020, 500))


frame_container = [frame]
cv2.setMouseCallback('Video', select_points, frame_container)

while cap.isOpened():
    temp_frame = frame.copy()  
    for spot_name, spot_points in parking_spots:
        cv2.polylines(temp_frame, [np.array(spot_points, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(temp_frame, spot_name, spot_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Video', temp_frame)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):  
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        frame = cv2.resize(frame, (1020, 500))
        frame_container[0] = frame  
    elif key == 27:  
        print("Saving parking spots to file...")
        with open("parking_spots.txt", "w") as file:
            for spot_name, spot_points in parking_spots:
                file.write(f"{spot_name}={spot_points}\n")
        print("Parking spots saved to 'parking_spots.txt'.")
        break

cap.release()
cv2.destroyAllWindows()
