from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
from typing import Optional
from app.afro_captioner import generate_caption
import pyfiglet

app = FastAPI(
    title="AfroVision Captioning API",
    description="🖼️ Generate captions for African **fashion** 👗 and **food** 🍲 using powerful AI models (BLIP).",
    version="1.0.0",
    contact={
        "name": "Yémalin Morel KPAVODE",
        "url": "https://github.com/yemal22",
        "email": "yemalem03@gmail.com",
    },
    openapi_tags=[
        {
            "name": "Home",
            "description": "Welcome and general information about the API."
        },
        {
            "name": "Fashion",
            "description": "Generate captions for African fashion images 👘👗🧥"
        },
        {
            "name": "Food",
            "description": "Generate captions for African food images 🍛🍲🥘"
        }
    ]
)


@app.get("/", include_in_schema=False, response_class=PlainTextResponse, tags=["Home"])
async def welcome():
    """
    Welcome Message
    """
    ascii_art = pyfiglet.figlet_format("Afro Captioner")
    message = f"""{ascii_art}
👋 Welcome to the Afro Caption Generator API!

Use:
➡️  POST /afro/fashion    - for African fashion image captioning
➡️  POST /afro/food       - for African food image captioning
➡️  GET /docs             - to view the API documentation
➡️  GET /redoc            - to view the Redoc documentation
➡️  GET /openapi.json     - to view the OpenAPI schema

"""
    return message

@app.post("/afro/fashion", response_class=JSONResponse, tags=["Fashion"], summary="Caption for fashion image")
async def fashion_caption(
    file: Optional[UploadFile] = File(None, description="Upload an image file"),
    image_url: Optional[str] = Form(None, description="Provide a direct image URL")
):
    """
    Generate a caption for an **African fashion image**.

    You can either upload an image file **or** provide a direct image URL.

    - `file`: Upload a `.jpg`, `.png`, etc.
    - `image_url`: A link to an image on the web
    """
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide either an image file or image URL.")

    try:
        input_source = file if file else image_url
        caption = generate_caption(input_source, model_type="fashion")
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/afro/food", response_class=JSONResponse, tags=["Food"], summary="Caption for food image")
async def food_caption(
    file: Optional[UploadFile] = File(None, description="Upload an image file"),
    image_url: Optional[str] = Form(None, description="Provide a direct image URL")
):
    """
    Generate a caption for an **African food image**.

    You can either upload an image file **or** provide a direct image URL.

    - `file`: Upload a `.jpg`, `.png`, etc.
    - `image_url`: A link to an image on the web
    """
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide either an image file or image URL.")

    try:
        input_source = file if file else image_url
        caption = generate_caption(input_source, model_type="food")
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally (optional)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
