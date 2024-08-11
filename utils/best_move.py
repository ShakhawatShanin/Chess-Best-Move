from stockfish import Stockfish

class BestMovePredictor:
    def __init__(self, stockfish_path="path/to/stockfish"):
        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.update_engine_parameters({"Threads": 2, "Minimum Thinking Time": 100})

    def set_fen_position(self, fen):
        self.stockfish.set_fen_position(fen)

    def get_best_move(self):
        return self.stockfish.get_best_move()

# # Example usage
# if __name__ == "__main__":
#     predictor = BestMovePredictor(stockfish_path="C:\\Users\\Shanin\\Desktop\\SHANIN\\DRF\\all_code\\Chess\\models\\stockfish_16_1.exe")
#     fen_string = "r1bqkbnr/pppppppp/n7/8/8/N7/PPPPPPPP/R1BQKBNR w KQkq - 2 2"
#     predictor.set_fen_position(fen_string)
#     best_move = predictor.get_best_move()
#     print(f"The best move is: {best_move}")
