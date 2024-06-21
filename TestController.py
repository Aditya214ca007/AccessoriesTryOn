import io

from fastapi import FastAPI
import cv2
import traceback
from fastapi.responses import StreamingResponse
from baseModal import TryOnRequest
from en import AccessoriesImposer
import base64


app = FastAPI()
accessories_imposer = AccessoriesImposer()

@app.post("/try_on_accessories")
async def try_on_accessories_endpoint(request_body: TryOnRequest):
    try:
        person_image_url = request_body.person_image_url
        article_url=request_body.article_url
        type=request_body.type

        # left_earring_url = request_body.left_earring_url
        # right_earring_url = request_body.right_earring_url
        # necklace_url = request_body.necklace_url
        # sunglasses_url = request_body.sunglasses_url

        left_earring_url = None
        right_earring_url = None
        necklace_url = None
        sunglasses_url = None
        if(type == "glasses"):
            sunglasses_url = article_url

        elif(type=="necklace"):
            necklace_url = article_url

        elif(type == "earring"):
            left_earring_url = article_url
            right_earring_url = article_url

        result_image = accessories_imposer.try_on_accessories(image_url=person_image_url,
                                                              left_earring_url=left_earring_url,
                                                              right_earring_url=right_earring_url,
                                                              necklace_url=necklace_url,
                                                              sunglasses_url=sunglasses_url)

        if result_image is None:
            return {"error": "Error processing image"}, 500

        # Convert result image to JPEG bytes
        ret, jpeg = cv2.imencode('.jpg', result_image)


        # return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")

        # Encode JPEG bytes to base64 string
        base64_image = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        return base64_image

    except Exception as e:
        traceback.print_exc()  # Print traceback to see where the error occurred
        return {"error": str(e)}, 500  # Return an error response with status code 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

























# async def try_on_accessories_endpoint(request_body: TryOnRequest):
#     try:
#         person_image_url = request_body.person_image_url
#         left_earring_url = request_body.left_earring_url
#         necklace_url = request_body.necklace_url
#         right_earring_url = request_body.right_earring_url
#
#         result_image = accessories_imposer.try_on_accessories(image_url=person_image_url, left_earring_url=left_earring_url,
#                                                               right_earring_url=right_earring_url,
#                                                               necklace_url=necklace_url)
#
#         if result_image is None:
#             return {"error": "Error processing image"}, 500
#
#         # Convert result image to JPEG bytes
#         ret, jpeg = cv2.imencode('.jpg', result_image)
#
#         # Encode JPEG bytes to base64 string
#         base64_image = base64.b64encode(jpeg.tobytes()).decode('utf-8')
#
#         return base64_image
#
#     except Exception as e:
#         traceback.print_exc()  # Print traceback to see where the error occurred
#         return {"error": str(e)}, 500  # Return an error response with status code 500
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# async def try_on_accessories_endpoint(person_image: UploadFile = File(None),
#                                       left_earring: UploadFile = File(None),
#                                       right_earring: UploadFile = File(None),
#                                       necklace_img: UploadFile = File(None)):
#     try:
#         left_earring_img = None
#         right_earring_img = None
#         necklace_image = None
#         image = None
#
#         if person_image is not None:
#             file_bytes = await person_image.read()
#             image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
#
#         # Load left earring image if provided
#         if left_earring is not None:
#             left_earring_bytes = await left_earring.read()
#             left_earring_img = cv2.imdecode(np.frombuffer(left_earring_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
#
#         # Load right earring image if provided
#         if right_earring is not None:
#             right_earring_bytes = await right_earring.read()
#             right_earring_img = cv2.imdecode(np.frombuffer(right_earring_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
#
#         # Load necklace image if provided
#         if necklace_img is not None:
#             necklace_bytes = await necklace_img.read()
#             necklace_img = cv2.imdecode(np.frombuffer(necklace_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
#
#         result_image =accessories_imposer.try_on_accessories(image=image, left_earring=left_earring_img,
#                                                  right_earring=right_earring_img, necklace_img=necklace_img)
#
#         if result_image is None:
#             return {"error": "Error processing image"}, 500
#
#             # Convert result image to JPEG bytes
#         ret, jpeg = cv2.imencode('.jpg', result_image)
#
#         # Encode JPEG bytes to base64 string
#         base64_image = base64.b64encode(jpeg.tobytes()).decode('utf-8')
#
#         return base64_image
#
#     except Exception as e:
#         traceback.print_exc()  # Print traceback to see where the error occurred
#         return {"error": str(e)}, 500  # Return an error response with status code 500

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)