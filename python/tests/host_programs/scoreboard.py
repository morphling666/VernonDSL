"""Host classes owning dicts: every dict operation, run alike in Python mode and AOT."""

import vernon_dsl as vd
from vernon_dsl import Vector, f32, i32


@vd.handle
class Player:
    pass


@vd.host
class Scoreboard:
    """Scores by player number, stats by name, and positions by player handle."""

    scores: dict[i32, i32] = {}
    stats: dict[str, f32]
    seen: dict[Player, Vector[f32, 2]]
    flags: dict[bool, i32]
    title: str = "speed"

    def score(self, player: i32, points: i32) -> i32:
        if player in self.scores:
            self.scores[player] += points
        else:
            self.scores[player] = points
        return len(self.scores)

    def points(self, player: i32) -> i32:
        return self.scores[player]

    def points_or(self, player: i32, fallback: i32) -> i32:
        return self.scores.get(player, fallback)

    def has(self, player: i32) -> bool:
        return player not in self.scores

    def total(self) -> i32:
        total = 0
        for points in self.scores.values():
            total += points
        return total

    def weighted(self) -> i32:
        # Keys in insertion order: a player's place counts.
        total = 0
        place = 1
        for player in self.scores:
            total += player * place
            place += 1
        return total

    def best(self) -> i32:
        best_player = -1
        best_points = -1
        for player, points in self.scores.items():
            if points > best_points:
                best_player = player
                best_points = points
        return best_player

    def forget(self, player: i32) -> i32:
        del self.scores[player]
        return len(self.scores)

    def take(self, player: i32) -> i32:
        return self.scores.pop(player)

    def take_or(self, player: i32, fallback: i32) -> i32:
        return self.scores.pop(player, fallback)

    def double_all(self) -> i32:
        # Assigning existing keys keeps the size, so the loop runs on.
        for player in self.scores:
            self.scores[player] *= 2
        return self.total()

    def grow_while_iterating(self) -> i32:
        for player in self.scores:
            self.scores[player + 100] = 0
        return 0

    def record(self, value: f32) -> f32:
        self.stats["speed"] = value
        self.stats[self.title] += f32(1.0)
        self.stats[""] = f32(-1.0)
        return self.stats["speed"] + self.stats.get("height", f32(0.5))

    def retitle(self) -> i32:
        self.title = "héight"
        self.stats[self.title] = f32(2.0)
        return len(self.title) * 10 + len(self.stats)

    def place(self, who: Player, x: f32) -> f32:
        self.seen[who] = Vector([x, x * f32(2.0)])
        return self.seen[who].y

    def flag(self, value: bool) -> i32:
        self.flags[value] = self.flags.get(value, 0) + 1
        return self.flags[True] if True in self.flags else -1

    def reset(self) -> i32:
        self.scores = {}
        self.stats.clear()
        return len(self.scores) + len(self.stats)
