"""
Reference - https://github.com/facebookresearch/nougat/blob/main/app.py
"""

import argparse
import sys
from functools import partial
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pypdfium2
import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible, close_envs
from nougat.utils.dataset import ImageDataset
from nougat.utils.checkpoint import get_checkpoint
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.device import move_to_device
from tqdm import tqdm


BATCHSIZE = 4
NOUGAT_CHECKPOINT = get_checkpoint()
if NOUGAT_CHECKPOINT is None:
    print(
        "Set environment variable 'NOUGAT_CHECKPOINT' with a path to the model checkpoint!"
    )
    sys.exit(1)

app = FastAPI(title="Nougat API")
origins = ["http://localhost", "http://127.0.0.1"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None


@app.on_event("startup")
async def load_model(
    checkpoint: str = NOUGAT_CHECKPOINT,
):
    global model, BATCHSIZE
    if model is None:
        model = NougatModel.from_pretrained(checkpoint)
        model = move_to_device(model, cuda=BATCHSIZE > 0)
        if BATCHSIZE <= 0:
            BATCHSIZE = 1
        model.eval()


@app.get("/")
def root():
    """Health check."""
    response = {
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...), start: int = None, stop: int = None
) -> str:
    """
    Perform predictions on a PDF document and return the extracted text in Markdown format.

    Args:
        file (UploadFile): The uploaded PDF file to process.
        start (int, optional): The starting page number for prediction.
        stop (int, optional): The ending page number for prediction.

    Returns:
        str: The extracted text in Markdown format.
    """
    pdfbin = file.file.read()
    pdf = pypdfium2.PdfDocument(pdfbin)

    # Page selection
    if start is not None and stop is not None:
        pages = list(range(start - 1, stop))
    else:
        pages = list(range(len(pdf)))

    # Rasterize pages
    images = rasterize_paper(pdf, pages=pages)
    global model

    dataset = ImageDataset(
        images,
        partial(model.encoder.prepare_input, random_padding=False),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        pin_memory=True,
        shuffle=False,
    )

    predictions = [""] * len(pages)

    # Run inference
    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        model_output = model.inference(image_tensors=sample)

        for j, output in enumerate(model_output["predictions"]):
            if model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    disclaimer = "\n\n+++ ==WARNING: Truncated because of repetitions==\n%s\n+++\n\n"
                else:
                    disclaimer = (
                        "\n\n+++ ==ERROR: No output for this page==\n%s\n+++\n\n"
                    )
                rest = close_envs(model_output["repetitions"][j]).strip()
                if len(rest) > 0:
                    disclaimer = disclaimer % rest
                else:
                    disclaimer = ""
            else:
                disclaimer = ""

            predictions[idx * BATCHSIZE + j] = (
                markdown_compatible(output) + disclaimer
            )

    final = "".join(predictions).strip()
    return final



def main():
    import uvicorn

    # Add argument parser for port number
    parser = argparse.ArgumentParser(description = 'Nougat API Server')
    parser.add_argument('--port', type = int, default = 8001, help = 'Port number to run the server on')
    args = parser.parse_args()

    # Use the port number from command line arguments
    uvicorn.run("app:app", port = args.port, host = "0.0.0.0")


if __name__ == "__main__":
    main()