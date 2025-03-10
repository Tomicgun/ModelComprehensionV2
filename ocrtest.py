import pymupdf
import os
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

def pdf2img(filepath, outdir):
    try:
        with pymupdf.open(filepath) as pdf:
            png_filename = os.path.splitext(os.path.basename(filepath))[0] + '.png'
            png_filepath = os.path.join(outdir, png_filename)
            pdf.load_page(0).get_pixmap(dpi=300).save(png_filepath)
            return png_filepath
    except:
        print(f'failed to convert {filepath} to image')
        return None

def findtext(filepath, predictor):
    result = predictor(DocumentFile.from_images(filepath))
    print(result)

if __name__ == '__main__':
    predictor = ocr_predictor(pretrained=True)
    directory = 'diagrams/raw'
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if not os.path.isfile(file):
            continue
        print(f'starting {file}')
        findtext(pdf2img(file, 'diagrams/png'), predictor)
        print(f'completed {file}')
