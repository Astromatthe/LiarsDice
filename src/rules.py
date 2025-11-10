from typing import List

def count_face_total(
        dice: List[List[int]],
        face: int
) -> int:
    """Counts the total number of faces showing in the given dice."""
    count = 0
    for player_dice in dice:
        for die in player_dice:
            if die == face:
                count += 1
    return count

def is_bid_higher(
        bid_old: List[int],
        bid_new: List[int]
) -> bool:
    """Compares two bids and returns True if the new bid is higher."""
    # Standard ordering: compare quantity first; ties break by face
    old_q, old_face = bid_old
    new_q, new_face = bid_new
    if old_q == 0:
        return True # any bid is higher than no bid
    if new_q > old_q:
        return True # higher quantity
    if new_q == old_q and new_face > old_face:
        return True # higher face
    return False