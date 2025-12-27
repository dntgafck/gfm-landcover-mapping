def test_window_enumeration_logic():
    # Synthetic H, W, P, S
    H, W = 1000, 1000
    P = 256
    S = 256

    windows = []
    for row_off in range(0, H - P + 1, S):
        for col_off in range(0, W - P + 1, S):
            windows.append((row_off, col_off))

    # H - P + 1 = 1000 - 256 + 1 = 745
    # range(0, 745, 256) -> 0, 256, 512
    # 3x3 = 9 windows
    assert len(windows) == 9
    assert windows[0] == (0, 0)
    assert windows[1] == (0, 256)
    assert windows[3] == (256, 0)


def test_patch_id_format():
    tile_id = "test_tile"
    row_off = 0
    col_off = 256
    P = 256
    S = 256
    patch_id = f"{tile_id}_r{row_off}_c{col_off}_p{P}_s{S}"
    assert patch_id == "test_tile_r0_c256_p256_s256"
