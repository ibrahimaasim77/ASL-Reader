def detect_letter(lm, up, d):
    """Returns (letter, 1.0) or (None, 0.0). J/Q/Z handled in dynamic.py."""
    thumb, index, middle, ring, pinky = up
    ti  = d["thumb_index"]
    tm  = d["thumb_middle"]
    tr  = d["thumb_ring"]
    tp  = d["thumb_pinky"]
    im  = d["index_middle"]
    tty = d["thumb_tip_y"]
    imy = d["index_mcp_y"]
    mmy = d["middle_mcp_y"]
    rmy = d["ring_mcp_y"]

    # ── Unique finger states ────────────────────────────────────────────────
    if up == [0, 0, 0, 0, 1]:                                    return ("I", 1.0)
    if up == [0, 1, 1, 1, 0]:                                    return ("W", 1.0)

    # Y — thumb + pinky, wide spread
    if up == [1, 0, 0, 0, 1] and tp > 0.35:                     return ("Y", 1.0)

    # B — four fingers up, thumb folded
    if up == [0, 1, 1, 1, 1] and ti > 0.25:                     return ("B", 1.0)

    # K — three up, thumb nestled between index+middle
    if up == [1, 1, 1, 0, 0] and ti < 0.20 and tm < 0.20:       return ("K", 1.0)

    # F — middle+ring+pinky up, index+thumb pinch
    if up == [0, 0, 1, 1, 1] and ti < 0.12:                     return ("F", 1.0)

    # ── Two-finger up states [0,1,1,0,0] ───────────────────────────────────
    if up == [0, 1, 1, 0, 0]:
        if im < 0.06:                                            return ("R", 1.0)
        if 0.06 < im < 0.14 and ti > 0.20:                      return ("U", 1.0)
        if im >= 0.14:                                           return ("V", 1.0)
        if im < 0.12 and ti > 0.15:                             return ("H", 1.0)

    # ── Thumb+index up [1,1,0,0,0] ─────────────────────────────────────────
    if up == [1, 1, 0, 0, 0]:
        if ti > 0.30:                                            return ("L", 1.0)
        if tm < 0.18 and 0.10 < ti < 0.28:                      return ("P", 1.0)
        if 0.10 < ti < 0.30:                                     return ("G", 1.0)
        if 0.15 < ti < 0.35:                                     return ("C", 1.0)

    # ── Thumb only up [1,0,0,0,0] ──────────────────────────────────────────
    if up == [1, 0, 0, 0, 0] and ti > 0.30:                     return ("A", 1.0)

    # ── Index only up [0,1,0,0,0] ──────────────────────────────────────────
    if up == [0, 1, 0, 0, 0]:
        if tm < 0.15 and ti > 0.15:                             return ("D", 1.0)
        if 0.08 < ti < 0.25:                                     return ("X", 1.0)

    # ── All fingers down [0,0,0,0,0] — critical disambiguation ────────────
    if up == [0, 0, 0, 0, 0]:
        # O: all fingertips pinch close to thumb — check first (most specific)
        if ti < 0.12 and tm < 0.14 and tr < 0.16 and tp < 0.22: return ("O", 1.0)
        # S: thumb IN FRONT of fist (tip above knuckles in image coords = lower y)
        if tty < imy and 0.06 < ti < 0.20:                      return ("S", 1.0)
        # T: thumb inserted between index+middle (very close to middle)
        if tm < 0.10 and ti < 0.12 and tty > imy:               return ("T", 1.0)
        # M: thumb under all three fingers (index+middle+ring)
        if tty > imy and tty > rmy and ti < 0.14:               return ("M", 1.0)
        # N: thumb under index+middle but NOT ring
        if tty > imy and tty > mmy and tty < rmy and ti < 0.14: return ("N", 1.0)
        # E: fingers partially curled, thumb mid-range
        if 0.12 < ti < 0.24:                                    return ("E", 1.0)

    return (None, 0.0)
