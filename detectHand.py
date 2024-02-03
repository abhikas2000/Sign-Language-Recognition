import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, 
                       min_detection_confidence=0.7, max_num_hands=1)

while True:
    try:
        _, frame = cap.read()
        test_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_results=hands.process(test_img_rgb)

        if test_results.multi_hand_landmarks:
            # for hand_landmarks in test_results.multi_hand_landmarks:
            #     for i in range(len(hand_landmarks.landmark)):
            #         mp_drawing.draw_landmarks(
            #                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            #                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))
            cv2.putText(frame,'hand found' , (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame,'hand not found' , (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1,
                        cv2.LINE_AA)
                
        cv2.imshow('handFeed',frame)
        cv2.waitKey(1)
    except Exception as e:
         print(e)
cap.release()
cv2.destroyAllWindows()