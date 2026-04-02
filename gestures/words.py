def detect_word(lm, up, d):
    """Returns (word, 1.0) or (None, 0.0). Words take priority over letters."""
    thumb, index, middle, ring, pinky = up
    ti = d["thumb_index"]
    tp = d["thumb_pinky"]
    im = d["index_middle"]

    # All-five-up group — discriminate by spread
    if up == [1, 1, 1, 1, 1]:
        if ti < 0.15:                    return ("PLEASE", 1.0)
        if tp > 0.50:                    return ("HELLO", 1.0)
        if 0.25 < tp <= 0.50:            return ("THANKS", 1.0)

    # I LOVE YOU — thumb + index + pinky
    if (thumb == 1 and index == 1 and middle == 0
            and ring == 0 and pinky == 1 and tp > 0.40):
        return ("I_LOVE_YOU", 1.0)

    # STOP — four fingers up, thumb folded
    if up == [0, 1, 1, 1, 1]:           return ("STOP", 1.0)

    # NO — index + middle close together pointing
    if up == [0, 1, 1, 0, 0] and im < 0.06 and ti > 0.20:
        return ("NO", 1.0)

    # YES — tight fist
    if up == [0, 0, 0, 0, 0] and d["wrist_middle"] < 0.12:
        return ("YES", 1.0)

    return (None, 0.0)
