from fastapi import FastAPI
import cv2
import traceback
from baseModal import TryOnRequest
from en import AccessoriesImposer
from fastapi.responses import StreamingResponse
import io

import base64

app = FastAPI()
accessories_imposer = AccessoriesImposer()


@app.post("/try_on_accessories")
async def try_on_accessories_endpoint(request_body: TryOnRequest):
    try:
        person_image_url = request_body.person_image_url
        article_url = request_body.article_url
        body_type = request_body.type
        left_earring_url = None
        right_earring_url = None
        necklace_url = None
        sunglasses_url = None
        if body_type == "Sunglasses_Or_ReadingGlasses":
            sunglasses_url = article_url

        elif body_type == "Necklace":
            necklace_url = article_url

        elif body_type == "Earring":
            left_earring_url = article_url
            right_earring_url = article_url

        result_image = accessories_imposer.try_on_accessories(image_url=person_image_url,
                                                              left_earring_url=left_earring_url,
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

    uvicorn.run(app, host="0.0.0.0", port=8001)
