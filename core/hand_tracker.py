import cv2
import mediapipe as mp


class HandTracker:

    def __init__(
        self,
        max_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.7,
    ):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame):
        """
        Returns:
            annotated_frame
            hands_data
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        hands_data = {
            "left": None,
            "right": None
        }

        if results.multi_hand_landmarks and results.multi_handedness:

            for landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                label = handedness.classification[0].label.lower()

                self.mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                hand_info = self.extract_hand_info(
                    frame,
                    landmarks
                )

                hands_data[label] = hand_info

        return frame, hands_data

    def extract_hand_info(self, frame, landmarks):

        h, w, _ = frame.shape

        points = []

        for lm in landmarks.landmark:

            x = int(lm.x * w)
            y = int(lm.y * h)

            points.append((x, y))

        thumb_tip = points[4]
        index_tip = points[8]

        center = (
            (thumb_tip[0] + index_tip[0]) // 2,
            (thumb_tip[1] + index_tip[1]) // 2
        )

        return {
            "landmarks": points,
            "thumb_tip": thumb_tip,
            "index_tip": index_tip,
            "center": center
        }

    def close(self):
        self.hands.close()