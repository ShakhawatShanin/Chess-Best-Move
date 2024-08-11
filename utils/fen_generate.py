import cv2
import numpy as np
import os
from ultralytics import YOLO

class ChessboardFENGenerator:
    def __init__(self, model_path, class_fen_dict, classes, width, height, confidence_threshold=0.8):
        self.model = YOLO(model_path)
        self.class_fen_dict = class_fen_dict
        self.classes = classes
        self.width = width
        self.height = height
        self.confidence_threshold = confidence_threshold
        self.grid_size = (self.width // 8, self.height // 8)

    def detect_pieces(self, image):
        results = self.model(image)
        boxes, classes, confidences = [], [], []
        for result in results:
            for box in result.boxes:
                if box.conf[0] > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    boxes.append((x1, y1, x2, y2))
                    classes.append(self.classes[class_id])
                    confidences.append(box.conf[0])
        return boxes, classes

    def get_board_grid(self):
        return np.array([[
            [x * self.grid_size[0], y * self.grid_size[1]],
            [(x + 1) * self.grid_size[0], y * self.grid_size[1]],
            [(x + 1) * self.grid_size[0], (y + 1) * self.grid_size[1]],
            [x * self.grid_size[0], (y + 1) * self.grid_size[1]]
        ] for y in range(8) for x in range(8)], dtype=np.int32)

    def generate_fen(self, image, who_turn='w'):
        boxes, detected_classes = self.detect_pieces(image)
        grid_contours = self.get_board_grid()
        board = np.full((8, 8), '.')

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for grid_index, grid_box in enumerate(grid_contours):
                if cv2.pointPolygonTest(grid_box, (cx, cy), False) > 0:
                    row, col = divmod(grid_index, 8)
                    board[row, col] = self.class_fen_dict.get(detected_classes[i], '.')
                    break

        return self.board_to_fen(board, who_turn), board

    @staticmethod
    def board_to_fen(board, who_turn):
        def rank_to_fen(rank):
            result = ''
            empty_count = 0
            for cell in rank:
                if cell == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        result += str(empty_count)
                        empty_count = 0
                    result += cell
            if empty_count > 0:
                result += str(empty_count)
            return result

        return '/'.join(map(rank_to_fen, board.tolist())) + f' {who_turn} KQkq - 0 1'


# class_fen_dict = {
#     'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
#     'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
#     'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
#     'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p'
# }

# classes = list(class_fen_dict.keys())
# model_path = "C:\\Users\\Shanin\\Desktop\\SHANIN\\DRF\\all_code\\Chess\\models\\best.pt"
# image_path = "C:\\Users\\Shanin\\Desktop\\SHANIN\\DRF\\all_code\\Chess\\board.png" 
# fen_generator = ChessboardFENGenerator(model_path=model_path,
#                                        class_fen_dict=class_fen_dict,
#                                        classes=classes,
#                                        width=640, height=640)  # Replace with your image size

# image = cv2.imread(image_path)
# fen, board = fen_generator.generate_fen(image)
# print("Generated FEN:", fen)
