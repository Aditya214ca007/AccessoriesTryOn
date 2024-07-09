import cvzone
import dlib
import numpy as np
import requests
import cv2
import os
from rembg import remove
from PIL import Image


class AccessoriesImposer:

    def __init__(self):
        predictor_path = "shape_predictor_81_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)
        self.detector = dlib.get_frontal_face_detector()

    def download_image(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image_data = response.content
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Error: Could not download image from {url}")
            return None

    def background_remover(self):
        current_dir = os.getcwd()
        image_filename = 'input_image.jpeg'
        input_path = os.path.join(current_dir, image_filename)
        output_path = os.path.join(current_dir, 'output_image.png')
        input_image = Image.open(input_path)
        output_image = remove(input_image)
        output_image.save(output_path)

    def overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
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
        alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0

        img_crop[:] = (1.0 - alpha) * img_crop + alpha * img_overlay_crop

    def try_on_accessories(self, image_url, left_earring_url, necklace_url, sunglasses_url):
        image = self.download_image(image_url)
        left_earring = None
        right_earring = None
        necklace_img = None
        sunglasses = None

        if left_earring_url is not None:
            cv2.imwrite("input_image.jpeg", self.download_image(left_earring_url))
            self.background_remover()
            left_earring = cv2.imread("output_image.png", cv2.IMREAD_UNCHANGED)
            right_earring = left_earring

        if necklace_url is not None:
            cv2.imwrite("input_image.jpeg", self.download_image(necklace_url))
            self.background_remover()
            necklace_img = cv2.imread("output_image.png", cv2.IMREAD_UNCHANGED)

        if sunglasses_url is not None:
            cv2.imwrite("input_image.jpeg", self.download_image(sunglasses_url))
            self.background_remover()
            sunglasses = cv2.imread("output_image.png", cv2.IMREAD_UNCHANGED)

        if image is None:
            print("Error: Could not load input image.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            shape = self.predictor(gray, face)
            landmarks = self.predictor(gray, face)
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

            if sunglasses is not None:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces using Haar cascades
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                # Adjustment factors for x and y positions
                x_adjust = 8  # Change this value to adjust the horizontal position
                y_adjust = 0  # Change this value to adjust the vertical position

                # Use dlib for more precise facial landmarks within detected faces
                for (x, y, w, h) in faces:
                    # Create a dlib rectangle object for the face
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

                    # Detect landmarks using dlib
                    shape = predictor(gray, dlib_rect)
                    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

                    # Extract eye coordinates
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]

                    # Calculate the center and distance between eyes
                    left_eye_center = np.mean(left_eye, axis=0).astype(int)
                    right_eye_center = np.mean(right_eye, axis=0).astype(int)
                    eye_width = np.linalg.norm(right_eye_center - left_eye_center)

                    # Scale the sunglasses image to fit the width of the eyes
                    scale_factor = 2.42  # Adjust this factor as needed
                    factor = (eye_width / sunglasses.shape[1]) * scale_factor
                    new_sunglasses_w = int(sunglasses.shape[1] * factor)
                    new_sunglasses_h = int(sunglasses.shape[0] * factor)
                    resized_sunglasses = cv2.resize(sunglasses, (new_sunglasses_w, new_sunglasses_h),
                                                    interpolation=cv2.INTER_AREA)

                    # Position the sunglasses above the eyes
                    y_offset = left_eye_center[1] - resized_sunglasses.shape[0] // 2 + y_adjust
                    x_offset = left_eye_center[0] - resized_sunglasses.shape[1] // 3 + x_adjust

                    # Split the image into its color and alpha channels
                    if resized_sunglasses.shape[2] == 4:
                        sunglass_color = resized_sunglasses[:, :, :3]
                        alpha_mask = resized_sunglasses[:, :, 3]
                    else:
                        sunglass_color = resized_sunglasses
                        alpha_mask = np.ones((resized_sunglasses.shape[0], resized_sunglasses.shape[1]),
                                             dtype=resized_sunglasses.dtype) * 255

                    # Overlay the sunglasses on the image
                    self.overlay_image_alpha(image, sunglass_color, x_offset, y_offset, alpha_mask)



            # landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            # if sunglasses is not None:
            #     left_eye = landmarks[36:42]
            #     right_eye = landmarks[42:48]
            #
            #     left_eye_center = np.mean(left_eye, axis=0).astype(int)
            #     right_eye_center = np.mean(right_eye, axis=0).astype(int)
            #     eye_width = np.linalg.norm(right_eye_center - left_eye_center)
            #
            #     factor = (eye_width / sunglasses.shape[1]) * 2.5
            #     new_sunglasses_w = int(sunglasses.shape[1] * factor)
            #     new_sunglasses_h = int(sunglasses.shape[0] * factor)
            #     resized_sunglasses = cv2.resize(sunglasses, (new_sunglasses_w, new_sunglasses_h),
            #                                     interpolation=cv2.INTER_AREA)
            #
            #     y_offset = left_eye_center[1] - resized_sunglasses.shape[0] // 2
            #     x_offset = left_eye_center[0] - resized_sunglasses.shape[1] // 3
            #
            #     if resized_sunglasses.shape[2] == 4:
            #         sunglass_color = resized_sunglasses[:, :, :3]
            #         alpha_mask = resized_sunglasses[:, :, 3]
            #     else:
            #         sunglass_color = resized_sunglasses
            #         alpha_mask = np.ones((resized_sunglasses.shape[0], resized_sunglasses.shape[1]),
            #                              dtype=resized_sunglasses.dtype) * 255
            #
            #     self.overlay_image_alpha(image, sunglass_color, x_offset, y_offset, alpha_mask)

        return image

    # def try_on_accessories(self, image_url, left_earring_url, right_earring_url, necklace_url, sunglasses_url):
    #     image = self.download_image(image_url)
    #     left_earring = None
    #     right_earring = None
    #     necklace_img = None
    #     sunglasses = None
    #
    #     if(left_earring_url is not None):
    #         background_remover(self.download_image(left_earring_url))
    #         left_earring = cv2.imread("Background_Remover/Images/output_image.png", cv2.IMREAD_UNCHANGED)
    #         right_earring=left_earring
    #
    #         # left_earring = cv2.imread("output_images.png")
    #         # right_earring = left_earring
    #     elif(necklace_url is not None):
    #         background_remover(self.download_image(necklace_url))
    #         necklace_img = cv2.imread("Background_Remover/Images/output_image.png", cv2.IMREAD_UNCHANGED)
    #
    #         # necklace_img = cv2.imread("output_images.png")
    #     elif(sunglasses_url is not None):
    #         background_remover(self.download_image(sunglasses_url))
    #         sunglasses = cv2.imread("Background_Remover/Images/output_image.png", cv2.IMREAD_UNCHANGED)
    #
    #         # sunglasses = cv2.imread("output_images.png")
    #     if image is None:
    #         print("Error: Could not load input image.")
    #         return
    #
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     faces = self.detector(gray)
    #
    #     for face in faces:
    #         shape = self.predictor(gray, face)
    #         landmarks = self.predictor(gray, face)
    #         if left_earring is not None:
    #             left_ear_lobe = (landmarks.part(2).x, landmarks.part(2).y)
    #             left_ear_w = int(0.05 * image.shape[1])
    #             left_ear_h = int(0.1 * image.shape[0])
    #             left_ear_x = left_ear_lobe[0] - left_ear_w // 2
    #             left_ear_y = left_ear_lobe[1]
    #
    #             resized_left_earring = cv2.resize(left_earring, (left_ear_w, left_ear_h))
    #             image = self.overlay_image(image, resized_left_earring, (left_ear_y, left_ear_x))
    #
    #         if right_earring is not None:
    #             right_ear_lobe = (landmarks.part(14).x, landmarks.part(14).y)
    #             right_ear_w = int(0.05 * image.shape[1])
    #             right_ear_h = int(0.1 * image.shape[0])
    #             right_ear_x = right_ear_lobe[0] - right_ear_w // 2
    #             right_ear_y = right_ear_lobe[1]
    #
    #             resized_right_earring = cv2.resize(right_earring, (right_ear_w, right_ear_h))
    #             image = self.overlay_image(image, resized_right_earring, (right_ear_y, right_ear_x))
    #
    #         if necklace_img is not None:
    #             jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(6, 11)]
    #             chin_point = (landmarks.part(8).x, landmarks.part(8).y)
    #
    #             x_min = min(point[0] for point in jaw_points)
    #             x_max = max(point[0] for point in jaw_points)
    #             y_min = chin_point[1]
    #             y_max = chin_point[1] + 150
    #
    #             neck_y = y_min
    #             neck_height = y_max - y_min
    #             neck_width = int((x_max - x_min) * 2.5)
    #             neck_x = x_min - (neck_width - (x_max - x_min)) // 2
    #
    #             if neck_y + neck_height <= image.shape[0] and neck_x + neck_width <= image.shape[1] and neck_x >= 0:
    #                 necklace_resized = cv2.resize(necklace_img, (neck_width, neck_height))
    #
    #                 if necklace_resized.shape[2] == 4:  # With alpha channel
    #                     alpha_channel = necklace_resized[:, :, 3] / 255.0
    #                     overlay_color = necklace_resized[:, :, :3]
    #
    #                     for c in range(0, 3):
    #                         image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] = (
    #                                 overlay_color[:, :, c] * alpha_channel +
    #                                 image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] * (
    #                                             1 - alpha_channel)
    #                         )
    #                 else:  # Without alpha channel
    #                     image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width] = necklace_resized
    #
    #         landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    #         if sunglasses is not None:
    #             # left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    #             # right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    #
    #             left_eye = landmarks[36:42]
    #             right_eye = landmarks[42:48]
    #
    #             left_eye_center = np.mean(left_eye, axis=0).astype(int)
    #             right_eye_center = np.mean(right_eye, axis=0).astype(int)
    #             eye_width = np.linalg.norm(right_eye_center - left_eye_center)
    #
    #             factor = (eye_width / sunglasses.shape[1]) * 2.5
    #             new_sunglasses_w = int(sunglasses.shape[1] * factor)
    #             new_sunglasses_h = int(sunglasses.shape[0] * factor)
    #             resized_sunglasses = cv2.resize(sunglasses, (new_sunglasses_w, new_sunglasses_h),
    #                                         interpolation=cv2.INTER_AREA)
    #
    #             y_offset = left_eye_center[1] - resized_sunglasses.shape[0] // 2
    #             x_offset = left_eye_center[0] - resized_sunglasses.shape[1] // 3
    #
    #             if resized_sunglasses.shape[2] == 4:
    #                 sunglass_color = resized_sunglasses[:,  :, :3]
    #                 alpha_mask = resized_sunglasses[:, :, 3]
    #             else:
    #                 sunglass_color = resized_sunglasses
    #                 alpha_mask = np.ones((resized_sunglasses.shape[0], resized_sunglasses.shape[1]),
    #                                  dtype=resized_sunglasses.dtype) * 255
    #
    #
    #             self.overlay_image_alpha(image, resized_sunglasses, x_offset, y_offset, alpha_mask)
    #
    #     return image

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

    # def try_on_accessories(self, image_url, left_earring_url, right_earring_url, necklace_url, sunglasses_url):
    #     image = self.download_image(image_url)
    #     left_earring = None
    #     right_earring = None
    #     necklace_img = None
    #     sunglasses = None
    #
    #     if(left_earring_url is not None):
    #         background_remover(self.download_image(left_earring_url))
    #         left_earring = cv2.imread("output_images.png")
    #         right_earring = left_earring
    #     elif(necklace_url is not None):
    #         background_remover(self.download_image(necklace_url))
    #         necklace_img = cv2.imread("output_images.png")
    #     elif(sunglasses_url is not None):
    #         background_remover(self.download_image(sunglasses_url))
    #         sunglasses = cv2.imread("output_images.png")
    #     if image is None:
    #         print("Error: Could not load input image.")
    #         return
    #
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     faces = self.detector(gray)
    #
    #     for face in faces:
    #         shape = self.predictor(gray, face)
    #         landmarks = self.predictor(gray, face)
    #         if left_earring is not None:
    #             left_ear_lobe = (landmarks.part(2).x, landmarks.part(2).y)
    #             left_ear_w = int(0.05 * image.shape[1])
    #             left_ear_h = int(0.1 * image.shape[0])
    #             left_ear_x = left_ear_lobe[0] - left_ear_w // 2
    #             left_ear_y = left_ear_lobe[1]
    #
    #             resized_left_earring = cv2.resize(left_earring, (left_ear_w, left_ear_h))
    #             image = self.overlay_image(image, resized_left_earring, (left_ear_y, left_ear_x))
    #
    #         if right_earring is not None:
    #             right_ear_lobe = (landmarks.part(14).x, landmarks.part(14).y)
    #             right_ear_w = int(0.05 * image.shape[1])
    #             right_ear_h = int(0.1 * image.shape[0])
    #             right_ear_x = right_ear_lobe[0] - right_ear_w // 2
    #             right_ear_y = right_ear_lobe[1]
    #
    #             resized_right_earring = cv2.resize(right_earring, (right_ear_w, right_ear_h))
    #             image = self.overlay_image(image, resized_right_earring, (right_ear_y, right_ear_x))
    #
    #         if necklace_img is not None:
    #             jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(6, 11)]
    #             chin_point = (landmarks.part(8).x, landmarks.part(8).y)
    #
    #             x_min = min(point[0] for point in jaw_points)
    #             x_max = max(point[0] for point in jaw_points)
    #             y_min = chin_point[1]
    #             y_max = chin_point[1] + 150
    #
    #             neck_y = y_min
    #             neck_height = y_max - y_min
    #             neck_width = int((x_max - x_min) * 2.5)
    #             neck_x = x_min - (neck_width - (x_max - x_min)) // 2
    #
    #             if neck_y + neck_height <= image.shape[0] and neck_x + neck_width <= image.shape[1] and neck_x >= 0:
    #                 necklace_resized = cv2.resize(necklace_img, (neck_width, neck_height))
    #
    #                 if necklace_resized.shape[2] == 4:  # With alpha channel
    #                     alpha_channel = necklace_resized[:, :, 3] / 255.0
    #                     overlay_color = necklace_resized[:, :, :3]
    #
    #                     for c in range(0, 3):
    #                         image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] = (
    #                                 overlay_color[:, :, c] * alpha_channel +
    #                                 image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width, c] * (
    #                                             1 - alpha_channel)
    #                         )
    #                 else:  # Without alpha channel
    #                     image[neck_y:neck_y + neck_height, neck_x:neck_x + neck_width] = necklace_resized
    #
    #         landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    #         if sunglasses is not None:
    #             # left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    #             # right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    #
    #             left_eye = landmarks[36:42]
    #             right_eye = landmarks[42:48]
    #
    #             left_eye_center = np.mean(left_eye, axis=0).astype(int)
    #             right_eye_center = np.mean(right_eye, axis=0).astype(int)
    #             eye_width = np.linalg.norm(right_eye_center - left_eye_center)
    #
    #             factor = (eye_width / sunglasses.shape[1]) * 2.5
    #             new_sunglasses_w = int(sunglasses.shape[1] * factor)
    #             new_sunglasses_h = int(sunglasses.shape[0] * factor)
    #             resized_sunglasses = cv2.resize(sunglasses, (new_sunglasses_w, new_sunglasses_h),
    #                                         interpolation=cv2.INTER_AREA)
    #
    #             y_offset = left_eye_center[1] - resized_sunglasses.shape[0] // 2
    #             x_offset = left_eye_center[0] - resized_sunglasses.shape[1] // 3
    #
    #             if resized_sunglasses.shape[2] == 4:
    #                 sunglass_color = resized_sunglasses[:,  :, :3]
    #                 alpha_mask = resized_sunglasses[:, :, 3]
    #             else:
    #                 sunglass_color = resized_sunglasses
    #                 alpha_mask = np.ones((resized_sunglasses.shape[0], resized_sunglasses.shape[1]),
    #                                  dtype=resized_sunglasses.dtype) * 255
    #
    #
    #             self.overlay_image_alpha(image, resized_sunglasses, x_offset, y_offset, alpha_mask)
    #
    #     return image
    #
    # def overlay_image(self, background, overlay, position):
    #     y, x = position
    #     h, w = overlay.shape[0], overlay.shape[1]
    #
    #     if y + h > background.shape[0] or x + w > background.shape[1]:
    #         return background
    #
    #     if overlay.shape[2] == 4:  # With alpha channel
    #         alpha_s = overlay[:, :, 3] / 255.0
    #         alpha_l = 1.0 - alpha_s
    #         for c in range(0, 3):
    #             background[y:y + h, x:x + w, c] = (alpha_s * overlay[:, :, c] +
    #                                                alpha_l * background[y:y + h, x:x + w, c])
    #     else:  # Without alpha channel
    #         background[y:y + h, x:x + w] = overlay
    #
    #     return background
