import cv2
import numpy as np
import cvzone

# Load parking spots from a text file
def load_parking_spots(file_path):
    parking_spots = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or "=" not in line: 
                continue
            key, value = line.split("=", 1)  
            key = key.strip()  
            value = value.strip()   
            parking_spots[key] = eval(value)  
    return parking_spots


parking_spots = load_parking_spots("parking_spots.txt")

# Conditions for specific spots
spot_conditions = {
    4: 600, # Spot 4 has a threshold of 600
    7: 600,  # Spot 7 has a threshold of 600
    13: 600,  # Spot 13 has a threshold of 600
    27:400,  # Spot 27 has a threshold of 400
    28:450, # Spot 28 has a threshold of 450
    29:450, # Spot 29 has a threshold of 450
    31: 300  # Spot 31 has a threshold of 1000
}


cap = cv2.VideoCapture('camera.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

spaceCounter = 0  

while True:
    # Check if the end of the video is reached
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    
    ret, img = cap.read()
    if not ret:
        print("End of video.")
        break

    # Resize frame for consistency
    img = cv2.resize(img, (1020, 500))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    spaces = 0  

    # Loop through parking spots and check for occupancy
    for key, points in parking_spots.items():
        # Crop each parking spot from the transformed image
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask = np.zeros_like(imgDilate)
        cv2.fillPoly(mask, [pts], 255)  # Mask the parking spot area
        cropped = cv2.bitwise_and(imgDilate, mask)  # Apply mask to the image

        # Count non-zero pixels (indicating presence of car)
        count = cv2.countNonZero(cropped)

        spot_id = int(''.join(filter(str.isdigit, key))) 

        # Check for specific threshold for each spot
        if spot_id in spot_conditions:
            threshold = spot_conditions[spot_id]  # Use custom threshold for specific spots
        else:
            threshold = 500  # Default threshold for all other spots

        # Criteria for determining if the spot is occupied
        if count < threshold:
            color = (0, 255, 0)  
            thickness = 5
            spaces += 1  
        else:
            color = (0, 0, 255)  
            thickness = 2

        cv2.polylines(img, [pts], True, color, thickness)


    # Display the number of free parking spots at the top of the image
    cvzone.putTextRect(img, f'Free: {spaces}/{len(parking_spots)}', (50, 60), thickness=3, offset=20, colorR=(0, 200, 0))

    cv2.imshow('Parking Spots', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total free parking spots: {spaces}")
