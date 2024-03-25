from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import base64
from io import BytesIO
from threading import Lock
from .demo import get_demo, get_visualization

app = Flask(__name__)
demo_lock = Lock()
demo = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global demo, demo_lock
    
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        input_image = Image.open(file.stream).convert("RGB")
        
        with demo_lock:
            predictions, visualized_output = demo.run_on_image(
                cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
            )
        
        output_image = Image.fromarray(get_visualization(predictions["masks"]))
        # output_image.save("online/static/tmp.png")
        output_image_data = BytesIO()
        output_image.save(output_image_data, format='PNG')
        output_image_base64 = base64.b64encode(output_image_data.getvalue()).decode('utf-8')
        
        input_image_data = BytesIO()
        input_image.save(input_image_data, format='PNG')
        input_image_base64 = base64.b64encode(input_image_data.getvalue()).decode('utf-8')
        return render_template('index.html', input_image_base64=input_image_base64, output_image_base64=output_image_base64)
    return render_template('index.html')

if __name__ == '__main__':
    with demo_lock:
        demo = get_demo()
    app.run("localhost", port=10000, debug=False)
