MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

_ACTION_DELTAS = {
    0: (0, -1),  # Left
    1: (1, 0),  # Down
    2: (0, 1),  # Right
    3: (-1, 0),  # Up
}


class FrozenLakeEnv:
    def __init__(self):
        self._map = MAP
        self._nrow = 8
        self._ncol = 8
        self._row = 0
        self._col = 0

    def reset(self):
        self._row = 0
        self._col = 0
        return self._state()

    def step(self, action):
        dr, dc = _ACTION_DELTAS[action]
        self._row = max(0, min(self._nrow - 1, self._row + dr))
        self._col = max(0, min(self._ncol - 1, self._col + dc))
        cell = self._map[self._row][self._col]
        done = cell in ("H", "G")
        reward = 1.0 if cell == "G" else 0.0
        return self._state(), reward, done

    def render(self):
        for r in range(self._nrow):
            row_str = ""
            for c in range(self._ncol):
                if r == self._row and c == self._col:
                    row_str += "A "
                else:
                    row_str += self._map[r][c] + " "
            print(row_str)
        print()

    def get_state(self):
        return self._state()

    def is_terminal(self):
        return self._map[self._row][self._col] in ("H", "G")

    def _state(self):
        return self._row * self._ncol + self._col
