from __future__ import annotations

import chess
import torch

PIECE_CHANNELS = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

PROMOTION_TO_INDEX = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
INDEX_TO_PROMOTION = {value: key for key, value in PROMOTION_TO_INDEX.items()}

BOARD_CHANNELS = 18
ACTION_SIZE = 64 * 64 * len(PROMOTION_TO_INDEX)


def square_to_row_col(square: chess.Square) -> tuple[int, int]:
    """Return tensor coordinates with rank 8 at row 0 and rank 1 at row 7."""
    return 7 - chess.square_rank(square), chess.square_file(square)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board as C x 8 x 8 planes.

    The first 12 planes are piece planes. Extra planes encode side to move,
    castling rights, and en-passant availability so the model is not blind to
    legal-state details that do not appear on piece planes.
    """
    tensor = torch.zeros((BOARD_CHANNELS, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        row, col = square_to_row_col(square)
        tensor[PIECE_CHANNELS[(piece.piece_type, piece.color)], row, col] = 1.0

    tensor[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    tensor[13].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    tensor[14].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    tensor[15].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    tensor[16].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    if board.ep_square is not None:
        row, col = square_to_row_col(board.ep_square)
        tensor[17, row, col] = 1.0

    return tensor


def move_to_index(move: chess.Move) -> int:
    promotion_index = PROMOTION_TO_INDEX.get(move.promotion)
    if promotion_index is None:
        raise ValueError(f"Unsupported promotion piece in move {move.uci()}")
    return ((move.from_square * 64) + move.to_square) * len(PROMOTION_TO_INDEX) + promotion_index


def index_to_move(index: int) -> chess.Move:
    base, promotion_index = divmod(index, len(PROMOTION_TO_INDEX))
    from_square, to_square = divmod(base, 64)
    return chess.Move(from_square, to_square, promotion=INDEX_TO_PROMOTION[promotion_index])


def legal_action_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask

