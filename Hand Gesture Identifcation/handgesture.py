import cv2
import time
import mediapipe as mp

# Initialize the mediapipe hands, drawing and drawing styles classes.
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the variable for the FPS number
pTime = 0

def detectHandsLandmarks(image, hands, draw=True):

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:
        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                   thickness=3, circle_radius=4 ),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255),
                                                                                     thickness=3, circle_radius=2))

    return output_image, results

def countFingers(image, results, draw=True):

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()

    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}

    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label

        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:

            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]

            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y:
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper() + "_" + finger_name] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper() + "_THUMB"] = True

            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
# 
    # Return the output image, the status of each finger and the count of the fingers up of both hands.
    return output_image, fingers_statuses, count

def recognizeGestures(image, fingers_statuses, count, draw=True):

    gesture = 0

    # Create a copy of the input image.
    output_image = image.copy()

    # Store the labels of both hands in a list.
    hands_labels = ['RIGHT', 'LEFT']

    # Initialize a dictionary to store the gestures of both hands in the image.
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}

    # Iterate over the left and right hand.
    for hand_index, hand_label in enumerate(hands_labels):

        # Initialize a variable to store the color we will use to write the hands gestures on the image.
        # Initially it is red which represents that the gesture is not recognized.
        color = (0, 0, 255)

        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################

        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        if count[hand_label] == 2 and fingers_statuses[hand_label + '_MIDDLE'] and fingers_statuses[
            hand_label + '_INDEX']:

            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "V SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

        # Check if the person is making the 'SPIDERMAN' gesture with the hand.
        ##########################################################################################################################################################

        # Check if the number of fingers up is 3 and the fingers that are up, are the thumb, index and the pinky finger.
        elif count[hand_label] == 3 and fingers_statuses[hand_label + '_THUMB'] and fingers_statuses[
            hand_label + '_INDEX'] and fingers_statuses[hand_label + '_PINKY']:

            # Update the gesture value of the hand that we are iterating upon to SPIDERMAN SIGN.
            hands_gestures[hand_label] = "SPIDERMAN SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

        # Check if the person is making the 'HIGH-FIVE' gesture with the hand.

        # Check if the number of fingers up is 5, which means that all the fingers are up.
        elif count[hand_label] == 5:

            # Update the gesture value of the hand that we are iterating upon to HIGH-FIVE SIGN.
            hands_gestures[hand_label] = "HIGH-FIVE SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

            # Check if the person is making the 'V' gesture with the hand.
            ####################################################################################################################

            # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
            if count[hand_label] == 2 and fingers_statuses[hand_label + '_MIDDLE'] and fingers_statuses[
                hand_label + '_INDEX']:
                # Update the gesture value of the hand that we are iterating upon to V SIGN.
                hands_gestures[hand_label] = "V SIGN"

                # Update the color value to green.
                color = (0, 255, 0)

        # Check if the person is making the 'TELEPHONE' gesture with the hand.

        # Check if the number of fingers up is 2 and the fingers that are up, are the thumb and the pinky finger.
        elif count[hand_label] == 2 and fingers_statuses[hand_label + '_THUMB'] and \
                fingers_statuses[hand_label + '_PINKY']:

            # Update the gesture value of the hand that we are iterating upon to TELEPHONE SIGN.
            hands_gestures[hand_label] = "TELEPHONE SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

        # Check if the person is making the 'L' gesture with the hand.

        # Check if the number of fingers up is 2 and the fingers that are up, are the thumb and the index finger.
        elif count[hand_label] == 2 and fingers_statuses[hand_label + '_THUMB'] and \
                fingers_statuses[hand_label + '_INDEX']:

            # Update the gesture value of the hand that we are iterating upon to L SIGN.
            hands_gestures[hand_label] = "L SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

        # Check if the person is making the 'OK' gesture with the hand.

        # Check if the number of fingers up is 4 and the fingers that are up, are the middle, index, ring and pinky.
        elif count[hand_label] == 4 and fingers_statuses[hand_label + '_MIDDLE'] and \
                fingers_statuses[hand_label + '_RING'] and fingers_statuses[hand_label + '_PINKY'] and \
                fingers_statuses[hand_label + '_THUMB']:

            # Update the gesture value of the hand that we are iterating upon to OK SIGN.
            hands_gestures[hand_label] = "OK SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

        # Check if the person is making the 'ROCK' gesture with the hand.

        # Check if the number of fingers up is 2 and the fingers that are up, are the index and pinky.
        elif count[hand_label] == 2 and fingers_statuses[hand_label + '_INDEX'] and \
                fingers_statuses[hand_label + '_PINKY']:

            # Update the gesture value of the hand that we are iterating upon to ROCK SIGN.
            hands_gestures[hand_label] = "ROCK SIGN"

            # Update the color value to green.
            color = (0, 255, 0)

        # Check if the hands gestures are specified to be written.
        if draw:
            # Write the hand gesture on the output image.
            cv2.putText(output_image, hand_label + ': ' + hands_gestures[hand_label], (10, (hand_index + 1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 5)

    # Return the output image and the gestures of the both hands.
    return output_image, hands_gestures, gesture


# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Create named window for resizing purposes.
cv2.namedWindow('Gesture Recognition', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_videos)

    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:

        output_image, fingers_statuses, count = countFingers(frame, results, draw=False)
        frame, hands_gestures, gesture = recognizeGestures(frame, fingers_statuses, count)

    # Display the frame.
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (1050, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Gesture Recognition', frame)

    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1)

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()