import io

import numpy as np
from PIL import Image

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, 'jpeg')
    return buf.getvalue()
