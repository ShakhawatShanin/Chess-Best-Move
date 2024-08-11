from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from utils.fen_generate import ChessboardFENGenerator  
from utils.best_move import BestMovePredictor  

app = Flask(__name__)

# Paths to model and Stockfish
model_path = "C:\\Users\\Shanin\\Desktop\\SHANIN\\DRF\\all_code\\Chess\\models\\best.pt"
stockfish_path = "C:\\Users\\Shanin\\Desktop\\SHANIN\\DRF\\all_code\\Chess\\models\\stockfish_16_1.exe"

# Class mappings and FEN dictionary
classes = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
class_fen_dict = {'wr': "R", 'wn': "N", 'wk': "K", 'wq': "Q",
                  'wb': "B", 'wp': "P", 'bp': "p", 'br': "r",
                  'bn': "n", 'bb': "b", 'bk': "k", 'bq': "q"}

# Image frame shape
frame_shape = (640, 640) 

# Initialize FEN generator and Stockfish
fen_generator = ChessboardFENGenerator(model_path, class_fen_dict, classes, frame_shape[0], frame_shape[1])
chess_engine = BestMovePredictor(stockfish_path)  

# Function to decode base64 image data
def decode_image(image_data):
    image_data = image_data.split(",")[1]  # Remove the "data:image/png;base64," part
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to classify the chessboard image
@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.get_json()
    image_data = data['image']

    # Decode the image
    image = decode_image(image_data)
    image = cv2.resize(image, (frame_shape[0], frame_shape[1]))

    # Generate the FEN string
    fen_line, board_matrix = fen_generator.generate_fen(image)

    # Use Stockfish to find the best move
    chess_engine.set_fen_position(fen_line)
    best_move = chess_engine.get_best_move()

    return jsonify({"result": f"{best_move}"})

if __name__ == '__main__':
    app.run(debug=True)
