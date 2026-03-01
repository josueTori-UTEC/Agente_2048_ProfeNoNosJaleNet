from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np

from game_2048 import Game2048  # noqa: F401

ActionStr = str


def _to_exp(v: int) -> int:
    return 0 if v <= 0 else (v.bit_length() - 1)


def _encode_board(board: np.ndarray) -> int:
    b = 0
    shift = 0
    for r in range(4):
        for c in range(4):
            e = _to_exp(int(board[r, c]))
            if e > 15:
                e = 15
            b |= (e & 0xF) << shift
            shift += 4
    return b


def _get_nibble(b: int, idx: int) -> int:
    return (b >> (idx * 4)) & 0xF


def _set_nibble(b: int, idx: int, val: int) -> int:
    sh = idx * 4
    return (b & ~(0xF << sh)) | ((val & 0xF) << sh)


def _row_at(b: int, r: int) -> int:
    return (b >> (r * 16)) & 0xFFFF


def _col_state(b: int, c: int) -> int:
    i0 = c
    i1 = 4 + c
    i2 = 8 + c
    i3 = 12 + c
    return (_get_nibble(b, i0) |
            (_get_nibble(b, i1) << 4) |
            (_get_nibble(b, i2) << 8) |
            (_get_nibble(b, i3) << 12))


def _set_col_from_state(b: int, c: int, col: int) -> int:
    i0 = c
    i1 = 4 + c
    i2 = 8 + c
    i3 = 12 + c
    b = _set_nibble(b, i0,  col        & 0xF)
    b = _set_nibble(b, i1, (col >> 4)  & 0xF)
    b = _set_nibble(b, i2, (col >> 8)  & 0xF)
    b = _set_nibble(b, i3, (col >> 12) & 0xF)
    return b


_CORNERS = (0, 3, 12, 15)


class Agent:


    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

        self._init_tables()

        self._h_cache: Dict[Tuple[int, int], float] = {}
        self._h_cache_limit = 220000

        self._move_cache: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
        self._move_cache_limit = 260000

        # Corner objetivo fijado cuando aparece nuevo max tile
        self.max_seen_exp = 0
        self.corner = 0  # {0,3,12,15}

        self.tight_depth2 = 2  
        self.p_two = 0.9
        self.p_four = 0.1

    def _init_tables(self) -> None:
        self.row_left = np.zeros(65536, dtype=np.uint16)
        self.row_left_reward = np.zeros(65536, dtype=np.int32)
        self.row_right = np.zeros(65536, dtype=np.uint16)
        self.row_right_reward = np.zeros(65536, dtype=np.int32)

        self.row_empty = np.zeros(65536, dtype=np.uint8)
        self.row_emptymask = np.zeros(65536, dtype=np.uint8)  # 4-bit mask por fila
        self.row_smooth = np.zeros(65536, dtype=np.int16)      # negativo
        self.row_mono = np.zeros(65536, dtype=np.int16)        # positivo
        self.row_merges = np.zeros(65536, dtype=np.uint8)
        self.row_max = np.zeros(65536, dtype=np.uint8)

        self.rev_row = np.zeros(65536, dtype=np.uint16)

        W = (
            (65536, 32768, 16384, 8192),
            (512,   1024,  2048,  4096),
            (256,   128,   64,    32),
            (2,     4,     8,     16),
        )
        self.row_wscore = np.zeros((4, 65536), dtype=np.int64)

        for row in range(65536):
            t0 = (row      ) & 0xF
            t1 = (row >> 4 ) & 0xF
            t2 = (row >> 8 ) & 0xF
            t3 = (row >> 12) & 0xF
            tiles = [t0, t1, t2, t3]

            self.rev_row[row] = np.uint16(((row & 0xF) << 12) | ((row & 0xF0) << 4) | ((row & 0xF00) >> 4) | ((row & 0xF000) >> 12))

            nz = [t for t in tiles if t != 0]
            reward = 0
            merged = []
            i = 0
            while i < len(nz):
                if i + 1 < len(nz) and nz[i] == nz[i + 1]:
                    ne = nz[i] + 1
                    if ne > 15:
                        ne = 15
                    merged.append(ne)
                    reward += (1 << ne)
                    i += 2
                else:
                    merged.append(nz[i])
                    i += 1
            merged += [0] * (4 - len(merged))
            new_row = (merged[0] | (merged[1] << 4) | (merged[2] << 8) | (merged[3] << 12))
            self.row_left[row] = np.uint16(new_row)
            self.row_left_reward[row] = np.int32(reward)

            self.row_empty[row] = tiles.count(0)
            mask = 0
            if t0 == 0: mask |= 1
            if t1 == 0: mask |= 2
            if t2 == 0: mask |= 4
            if t3 == 0: mask |= 8
            self.row_emptymask[row] = np.uint8(mask)

            self.row_max[row] = max(tiles)

            smooth = 0
            for k in range(3):
                if tiles[k] and tiles[k+1]:
                    smooth -= abs(tiles[k] - tiles[k+1])
            self.row_smooth[row] = np.int16(smooth)

            inc = 0
            dec = 0
            for k in range(3):
                x, y = tiles[k], tiles[k+1]
                if x == 0 or y == 0:
                    continue
                if x > y:
                    dec += x - y
                else:
                    inc += y - x
            self.row_mono[row] = np.int16(max(inc, dec))

            merges = 0
            for k in range(len(nz) - 1):
                if nz[k] == nz[k+1]:
                    merges += 1
            self.row_merges[row] = np.uint8(merges)

            for r in range(4):
                w = W[r]
                v0 = (1 << tiles[0]) if tiles[0] else 0
                v1 = (1 << tiles[1]) if tiles[1] else 0
                v2 = (1 << tiles[2]) if tiles[2] else 0
                v3 = (1 << tiles[3]) if tiles[3] else 0
                self.row_wscore[r, row] = v0*w[0] + v1*w[1] + v2*w[2] + v3*w[3]

        for row in range(65536):
            rr = int(self.rev_row[row])
            self.row_right[row] = np.uint16(self.rev_row[int(self.row_left[rr])])
            self.row_right_reward[row] = np.int32(self.row_left_reward[rr])

    def _move(self, b: int, action: ActionStr) -> Tuple[int, int, bool]:
        aid = 0 if action == "up" else 1 if action == "down" else 2 if action == "left" else 3
        key = (b, aid)
        got = self._move_cache.get(key)
        if got is not None:
            return got

        if action == "left":
            nb = 0
            reward = 0
            moved = False
            for r in range(4):
                row = _row_at(b, r)
                new_row = int(self.row_left[row])
                nb |= new_row << (r * 16)
                reward += int(self.row_left_reward[row])
                moved = moved or (new_row != row)
            out = (nb, reward, moved)

        elif action == "right":
            nb = 0
            reward = 0
            moved = False
            for r in range(4):
                row = _row_at(b, r)
                new_row = int(self.row_right[row])
                nb |= new_row << (r * 16)
                reward += int(self.row_right_reward[row])
                moved = moved or (new_row != row)
            out = (nb, reward, moved)

        elif action == "up":
            nb = 0
            reward = 0
            moved = False
            for c in range(4):
                col = _col_state(b, c)
                new_col = int(self.row_left[col])
                reward += int(self.row_left_reward[col])
                moved = moved or (new_col != col)
                nb = _set_col_from_state(nb, c, new_col)
            out = (nb, reward, moved)

        else:  # down
            nb = 0
            reward = 0
            moved = False
            for c in range(4):
                col = _col_state(b, c)
                new_col = int(self.row_right[col])
                reward += int(self.row_right_reward[col])
                moved = moved or (new_col != col)
                nb = _set_col_from_state(nb, c, new_col)
            out = (nb, reward, moved)

        if len(self._move_cache) > self._move_cache_limit:
            for k in list(self._move_cache.keys())[: self._move_cache_limit // 2]:
                del self._move_cache[k]
        self._move_cache[key] = out
        return out

    def _board_empty_mask(self, b: int) -> int:
        r0 = _row_at(b, 0); r1 = _row_at(b, 1); r2 = _row_at(b, 2); r3 = _row_at(b, 3)
        return (int(self.row_emptymask[r0]) |
                (int(self.row_emptymask[r1]) << 4) |
                (int(self.row_emptymask[r2]) << 8) |
                (int(self.row_emptymask[r3]) << 12))

    def _sample_positions_from_mask(self, mask: int, k: int, seed: int) -> List[int]:
        n = mask.bit_count()
        if n <= k:
            out = []
            m = mask
            while m:
                lsb = m & -m
                out.append(lsb.bit_length() - 1)
                m ^= lsb
            return out

        start = seed & 15
        out = []
        for j in range(16):
            i = (start + j) & 15
            if (mask >> i) & 1:
                out.append(i)
                if len(out) == k:
                    break
        return out

    def _spawned(self, b: int, idx: int, exp: int) -> int:
        return b | ((exp & 0xF) << (idx * 4))

    def _wscore_for_corner(self, b: int, corner: int) -> int:
        r0 = _row_at(b, 0); r1 = _row_at(b, 1); r2 = _row_at(b, 2); r3 = _row_at(b, 3)

        if corner == 0:  # TL
            return int(self.row_wscore[0, r0] + self.row_wscore[1, r1] + self.row_wscore[2, r2] + self.row_wscore[3, r3])
        if corner == 3:  # TR
            rr0 = int(self.rev_row[r0]); rr1 = int(self.rev_row[r1]); rr2 = int(self.rev_row[r2]); rr3 = int(self.rev_row[r3])
            return int(self.row_wscore[0, rr0] + self.row_wscore[1, rr1] + self.row_wscore[2, rr2] + self.row_wscore[3, rr3])
        if corner == 12:  # BL
            return int(self.row_wscore[0, r3] + self.row_wscore[1, r2] + self.row_wscore[2, r1] + self.row_wscore[3, r0])
        rr0 = int(self.rev_row[r0]); rr1 = int(self.rev_row[r1]); rr2 = int(self.rev_row[r2]); rr3 = int(self.rev_row[r3])
        return int(self.row_wscore[0, rr3] + self.row_wscore[1, rr2] + self.row_wscore[2, rr1] + self.row_wscore[3, rr0])

    def _heuristic(self, b: int) -> float:
        key = (b, self.corner)
        cached = self._h_cache.get(key)
        if cached is not None:
            return cached

        r0 = _row_at(b, 0); r1 = _row_at(b, 1); r2 = _row_at(b, 2); r3 = _row_at(b, 3)
        empty = int(self.row_empty[r0] + self.row_empty[r1] + self.row_empty[r2] + self.row_empty[r3])

        smooth = int(self.row_smooth[r0] + self.row_smooth[r1] + self.row_smooth[r2] + self.row_smooth[r3])
        mono = int(self.row_mono[r0] + self.row_mono[r1] + self.row_mono[r2] + self.row_mono[r3])
        merges = int(self.row_merges[r0] + self.row_merges[r1] + self.row_merges[r2] + self.row_merges[r3])

        c0 = _col_state(b, 0); c1 = _col_state(b, 1); c2 = _col_state(b, 2); c3 = _col_state(b, 3)
        smooth += int(self.row_smooth[c0] + self.row_smooth[c1] + self.row_smooth[c2] + self.row_smooth[c3])
        mono   += int(self.row_mono[c0]   + self.row_mono[c1]   + self.row_mono[c2]   + self.row_mono[c3])
        merges += int(self.row_merges[c0] + self.row_merges[c1] + self.row_merges[c2] + self.row_merges[c3])

        max_e = int(max(self.row_max[r0], self.row_max[r1], self.row_max[r2], self.row_max[r3]))
        max_tile = float(1 << max_e) if max_e else 0.0

        wscore = float(self._wscore_for_corner(b, self.corner))
        corner_ok = 1.0 if (_get_nibble(b, self.corner) == max_e) else 0.0

        empty_w = 320.0 if empty <= 6 else 260.0

        val = (
            empty_w * empty
            + 0.00011 * wscore
            + 82.0 * merges
            + 26.0 * mono
            + 18.0 * smooth
            + 1.6 * max_tile
            + (0.9 * corner_ok) * max_tile
        )

        if len(self._h_cache) > self._h_cache_limit:
            for k in list(self._h_cache.keys())[: self._h_cache_limit // 2]:
                del self._h_cache[k]
        self._h_cache[key] = float(val)
        return float(val)

    def _action_order(self, legal: List[str]) -> List[str]:
        if self.corner == 0:
            pref = ("up", "left", "right", "down")
        elif self.corner == 3:
            pref = ("up", "right", "left", "down")
        elif self.corner == 12:
            pref = ("down", "left", "right", "up")
        else:
            pref = ("down", "right", "left", "up")

        out = [a for a in pref if a in legal]
        for a in legal:
            if a not in out:
                out.append(a)
        return out

    def _action_bias(self, action: str) -> float:
        if self.corner == 0:
            return -80.0 if action in ("down", "right") else 0.0
        if self.corner == 3:
            return -80.0 if action in ("down", "left") else 0.0
        if self.corner == 12:
            return -80.0 if action in ("up", "right") else 0.0
        return -80.0 if action in ("up", "left") else 0.0

    def _second_ply_actions(self) -> Tuple[str, str]:
        
        if self.corner == 0:
            return ("up", "left")
        if self.corner == 3:
            return ("up", "right")
        if self.corner == 12:
            return ("down", "left")
        return ("down", "right")

    def act(self, board: np.ndarray, legal_actions: List[str]) -> str:
        if not legal_actions:
            return "up"

        b = _encode_board(board)

        max_e = 0
        max_pos = 0
        for i in range(16):
            e = (b >> (i * 4)) & 0xF
            if e > max_e:
                max_e = int(e)
                max_pos = i
        if max_e > self.max_seen_exp:
            self.max_seen_exp = max_e
            r, c = divmod(max_pos, 4)
            best = 0
            bestd = 999
            for corner in _CORNERS:
                rr, cc = divmod(corner, 4)
                d = abs(r - rr) + abs(c - cc)
                if d < bestd:
                    bestd = d
                    best = corner
            self.corner = best

        actions1 = self._action_order(legal_actions)

        empties_now = self._board_empty_mask(b).bit_count()
        use_depth2 = empties_now <= self.tight_depth2

        best_a = actions1[0]
        best_v = -1e30

        for a1 in actions1:
            b1, r1, moved1 = self._move(b, a1)
            if not moved1:
                continue

            mask1 = self._board_empty_mask(b1)
            nE = mask1.bit_count()
            if nE == 0:
                v = float(r1) + self._heuristic(b1) + self._action_bias(a1)
                if v > best_v:
                    best_v = v
                    best_a = a1
                continue

            k = 2 if nE >= 7 else 3
            seed = (b1 ^ (b1 >> 17) ^ (b1 >> 43)) & 0xFFFFFFFF
            spawns = self._sample_positions_from_mask(mask1, k, int(seed))

            inv = 1.0 / float(len(spawns))
            exp_v = 0.0

            if not use_depth2:
                for pos in spawns:
                    b2 = self._spawned(b1, pos, 1)
                    b4 = self._spawned(b1, pos, 2)
                    exp_v += inv * (self.p_two * self._heuristic(b2) + self.p_four * self._heuristic(b4))
                v = float(r1) + exp_v + self._action_bias(a1)
                if v > best_v:
                    best_v = v
                    best_a = a1
            else:
                for pos in spawns:
                    b2 = self._spawned(b1, pos, 1)
                    best2 = -1e30
                    for a2 in self._second_ply_actions():
                        b3, r2, moved2 = self._move(b2, a2)
                        if moved2:
                            vv = float(r2) + self._heuristic(b3) + 0.35 * self._action_bias(a2)
                            if vv > best2:
                                best2 = vv
                    if best2 < -1e20:
                        best2 = self._heuristic(b2)
                    exp_v += inv * (self.p_two * best2)

                    b4 = self._spawned(b1, pos, 2)
                    exp_v += inv * (self.p_four * self._heuristic(b4))

                v = float(r1) + exp_v + self._action_bias(a1)
                if v > best_v:
                    best_v = v
                    best_a = a1


        return best_a if best_a in legal_actions else legal_actions[0]
