import cv2
import dlib
import numpy as np
import requests

class AccessoriesImposer:

    def __init__(self):
        predictor_path = "shape_predictor_81_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)
        self.detector = dlib.get_frontal_face_detector()

    def download_image(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image = np.array(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        else:
            print(f"Error: Could not download image from {url}")
            return None

    def overlay_image_alpha(self, img, img_overlay, x, y):
        """Overlay img_overlay on top of img at (x, y) and blend using alpha channel."""
        if img_overlay.shape[2] == 4:  # Check if the overlay image has an alpha channel
            # Extract the alpha mask of the RGBA image, resize it to match img_overlay
            alpha_mask = img_overlay[:, :, 3] / 255.0
            img_overlay = img_overlay[:, :, :3]

            # Image ranges
            y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
            x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

            y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
            x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

            # Exit if there's no overlap
            if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
                return

            # Blend overlay within the determined ranges
            img_crop = img[y1:y2, x1:x2]
            img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
            alpha = alpha_mask[y1o:y2o, x1o:x2o, None]

            img[y1:y2, x1:x2] = (1.0 - alpha) * img_crop + alpha * img_overlay_crop

    def try_on_accessories(self, image_url, left_earring_url, right_earring_url, necklace_url, sunglasses_url):
        image = self.download_image(image_url)
        left_earring = self.download_image(left_earring_url) if left_earring_url is not None else None
        right_earring = self.download_image(right_earring_url) if right_earring_url is not None else None
        necklace_img = self.download_image(necklace_url) if necklace_url is not None else None
        sunglasses = self.download_image(sunglasses_url) if sunglasses_url is not None else None

        if image is None:
            print("Error: Could not load input image.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            if sunglasses is not None:
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                left_eye_center = np.mean(left_eye, axis=0).astype(int)
                right_eye_center = np.mean(right_eye, axis=0).astype(int)
                eye_width = np.linalg.norm(right_eye_center - left_eye_center)

                factor = (eye_width / sunglasses.shape[1]) * 2.5
                new_sunglasses_w = int(sunglasses.shape[1] * factor)
                new_sunglasses_h = int(sunglasses.shape[0] * factor)
                resized_sunglasses = cv2.resize(sunglasses, (new_sunglasses_w, new_sunglasses_h),
                                                interpolation=cv2.INTER_AREA)

                y_offset = left_eye_center[1] - resized_sunglasses.shape[0] // 2
                x_offset = left_eye_center[0] - resized_sunglasses.shape[1] // 3

                self.overlay_image_alpha(image, resized_sunglasses, x_offset, y_offset)

            if left_earring is not None:
                left_ear_lobe = (landmarks.part(2).x, landmarks.part(2).y)
                left_ear_w = int(0.05 * image.shape[1])
                left_ear_h = int(0.1 * image.shape[0])
                left_ear_x = left_ear_lobe[0] - left_ear_w // 2
                left_ear_y = left_ear_lobe[1]

                resized_left_earring = cv2.resize(left_earring, (left_ear_w, left_ear_h))
                image = self.overlay_image(image, resized_left_earring, (left_ear_y, left_ear_x))

            if right_earring is not None:
                right_ear_lobe = (landmarks.part(14).x, landmarks.part(14).y)
                right_ear_w = int(0.05 * image.shape[1])
                right_ear_h = int(0.1 * image.shape[0])
                right_ear_x = right_ear_lobe[0] - right_ear_w // 2
                right_ear_y = right_ear_lobe[1]

                resized_right_earring = cv2.resize(right_earring, (right_ear_w, right_ear_h))
                image = self.overlay_image(image, resized_right_earring, (right_ear_y, right_ear_x))

            if necklace_img is not None:
                jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(6, 11)]
                chin_point = (landmarks.part(8).x, landmarks.part(8).y)

                x_min = min(point[0] for point in jaw_points)
                x_max = max(point[0] for point in jaw_points)
                y_min = chin_point[1]
                y_max = chin_point[1] + 150

                neck_y = y_min
                neck_height = y_max - y_min
                neck_width = int((x_max - x_min) * 2.5)
                neck_x = x_min - (neck_width - (x_max - x_min)) // 2

                if neck_y + neck_height <= image.shape[0] and neck_x + neck_width <= image.shape[1] and neck_x >= 0:
                    necklace_resized = cv2.resize(necklace_img, (neck_width, neck_height))

                    if necklace_resized.shape[2] == 4:  # With alpha channel
                        alpha_channel = necklace_resized[:, :, 3] / 255.0
                        overlay_color = necklace_resized[:, :, :3]

                        for c in range(0, 3):
                            image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] = (
                                    overlay_color[:, :, c] * alpha_channel +
                                    image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] * (
                                                1 - alpha_channel)
                            )
                    else:  # Without alpha channel
                        image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width] = necklace_resized

        return image

    def overlay_image(self, background, overlay, position):
        y, x = position
        h, w = overlay.shape[0], overlay.shape[1]

        if y + h > background.shape[0] or x + w > background.shape[1]:
            return background

        if overlay.shape[2] == 4:  # With alpha channel
            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                background[y:y + h, x:x + w, c] = (alpha_s * overlay[:, :, c] +
                                                   alpha_l * background[y:y + h, x:x + w, c])
        else:  # Without alpha channel
            background[y:y + h, x:x + w] = overlay

        return background





















































































































# import cv2
# import dlib
# import numpy as np
# import requests
#
# class AccessoriesImposer:
#
#     def __init__(self):
#         predictor_path = "shape_predictor_81_face_landmarks.dat"
#         self.predictor = dlib.shape_predictor(predictor_path)
#         self.detector = dlib.get_frontal_face_detector()
#
#     def download_image(self, url):
#         response = requests.get(url)
#         if response.status_code == 200:
#             image = np.array(bytearray(response.content), dtype=np.uint8)
#             return cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
#         else:
#             print(f"Error: Could not download image from {url}")
#             return None
#
#     def overlay_image_alpha(self,img, img_overlay, x, y):
#         """Overlay img_overlay on top of img at (x, y) and blend using alpha channel."""
#         # Extract the alpha mask of the RGBA image, resize it to match img_overlay
#         alpha_mask = img_overlay[:, :, 3] / 255.0
#         img_overlay = img_overlay[:, :, :3]
#
#         # Image ranges
#         y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
#         x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
#
#         y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
#         x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
#
#         # Exit if there's no overlap
#         if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
#             return
#
#         # Blend overlay within the determined ranges
#         img_crop = img[y1:y2, x1:x2]
#         img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
#         alpha = alpha_mask[y1o:y2o, x1o:x2o, None]
#
#         img[y1:y2, x1:x2] = (1.0 - alpha) * img_crop + alpha * img_overlay_crop
#
#     def try_on_accessories(self, image_url, left_earring_url, right_earring_url, necklace_url, sunglasses_url):
#         image = self.download_image(image_url)
#         left_earring = self.download_image(left_earring_url) if left_earring_url is not None else None
#         right_earring = self.download_image(right_earring_url) if right_earring_url is not None else None
#         necklace_img = self.download_image(necklace_url) if necklace_url is not None else None
#         sunglasses = self.download_image(sunglasses_url) if sunglasses_url is not None else None
#
#         if image is None:
#             print("Error: Could not load input image.")
#             return
#
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = self.detector(gray)
#
#         for face in faces:
#             landmarks = self.predictor(gray, face)
#
#             if sunglasses is not None:
#                 left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
#                 right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
#
#                 left_eye_center = np.mean(left_eye, axis=0).astype(int)
#                 right_eye_center = np.mean(right_eye, axis=0).astype(int)
#                 eye_width = np.linalg.norm(right_eye_center - left_eye_center)
#
#                 factor = (eye_width / sunglasses.shape[1]) * 2.5
#                 new_sunglasses_w = int(sunglasses.shape[1] * factor)
#                 new_sunglasses_h = int(sunglasses.shape[0] * factor)
#                 resized_sunglasses = cv2.resize(sunglasses, (new_sunglasses_w, new_sunglasses_h),
#                                                 interpolation=cv2.INTER_AREA)
#
#                 y_offset = left_eye_center[1] - resized_sunglasses.shape[0] // 2
#                 x_offset = left_eye_center[0] - resized_sunglasses.shape[1] // 3
#
#                 self.overlay_image_alpha(image, resized_sunglasses, x_offset, y_offset)
#
#             if left_earring is not None:
#                 left_ear_lobe = (landmarks.part(2).x, landmarks.part(2).y)
#                 left_ear_w = int(0.05 * image.shape[1])
#                 left_ear_h = int(0.1 * image.shape[0])
#                 left_ear_x = left_ear_lobe[0] - left_ear_w // 2
#                 left_ear_y = left_ear_lobe[1]
#
#                 resized_left_earring = cv2.resize(left_earring, (left_ear_w, left_ear_h))
#                 image = self.overlay_image(image, resized_left_earring, (left_ear_y, left_ear_x))
#
#             if right_earring is not None:
#                 right_ear_lobe = (landmarks.part(14).x, landmarks.part(14).y)
#                 right_ear_w = int(0.05 * image.shape[1])
#                 right_ear_h = int(0.1 * image.shape[0])
#                 right_ear_x = right_ear_lobe[0] - right_ear_w // 2
#                 right_ear_y = right_ear_lobe[1]
#
#                 resized_right_earring = cv2.resize(right_earring, (right_ear_w, right_ear_h))
#                 image = self.overlay_image(image, resized_right_earring, (right_ear_y, right_ear_x))
#
#             if necklace_img is not None:
#                 jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(6, 11)]
#                 chin_point = (landmarks.part(8).x, landmarks.part(8).y)
#
#                 x_min = min(point[0] for point in jaw_points)
#                 x_max = max(point[0] for point in jaw_points)
#                 y_min = chin_point[1]
#                 y_max = chin_point[1] + 150
#
#                 neck_y = y_min
#                 neck_height = y_max - y_min
#                 neck_width = int((x_max - x_min) * 2.5)
#                 neck_x = x_min - (neck_width - (x_max - x_min)) // 2
#
#                 if neck_y + neck_height <= image.shape[0] and neck_x + neck_width <= image.shape[1] and neck_x >= 0:
#                     necklace_resized = cv2.resize(necklace_img, (neck_width, neck_height))
#
#                     if necklace_resized.shape[2] == 4:  # With alpha channel
#                         alpha_channel = necklace_resized[:, :, 3] / 255.0
#                         overlay_color = necklace_resized[:, :, :3]
#
#                         for c in range(0, 3):
#                             image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] = (
#                                     overlay_color[:, :, c] * alpha_channel +
#                                     image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] * (
#                                                 1 - alpha_channel)
#                             )
#                     else:  # Without alpha channel
#                         image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width] = necklace_resized
#
#         return image
#
#     def overlay_image(self, background, overlay, position):
#         y, x = position
#         h, w = overlay.shape[0], overlay.shape[1]
#
#         if y + h > background.shape[0] or x + w > background.shape[1]:
#             return background
#
#         if overlay.shape[2] == 4:  # With alpha channel
#             alpha_s = overlay[:, :, 3] / 255.0
#             alpha_l = 1.0 - alpha_s
#             for c in range(0, 3):
#                 background[y:y + h, x:x + w, c] = (alpha_s * overlay[:, :, c] +
#                                                    alpha_l * background[y:y + h, x:x + w, c])
#         else:  # Without alpha channel
#             background[y:y + h, x:x + w] = overlay
#
#         return background
