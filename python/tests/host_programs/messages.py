"""str locals and f-strings: messages built in Host code, run alike in Python mode and AOT."""

import vernon_dsl as vd
from vernon_dsl import f32, f64, i32, i64, u64


@vd.handle
class Ship: ...


@vd.extern(symbol="messages_log")
def log(message: str, value: i32) -> None: ...


@vd.extern(symbol="messages_stamp")
def stamp(message: str) -> i32: ...


@vd.host
class Logbook:
    """A captain's log: what each ship reported, and how often each word came up."""

    captain: str = "nobody"
    last: str
    words: dict[str, i32]
    entries: i32 = 0

    def report(self, ship: Ship, speed: f32, fuel: f64, docked: bool) -> i32:
        status = "docked" if docked else "under way"
        line = f"ship {ship} at {speed} m/s, fuel {fuel:.2f} ({status}, docked={docked})"
        log(line, len(line))
        self.last = line
        self.entries += 1
        return len(line)

    def numbers(self, count: i32, mask: u64, delta: i64, ratio: f32) -> i32:
        line = f"[{count:>5}|{count:<5}|{count:^7}|{count:05}|{count:+}]"
        line += f" {mask:x} {mask:X} {mask:o}"
        line = line + f" {mask:b} {delta:+d} {delta: d} {delta:=+8d}"
        line += f" {ratio} {ratio:.3f} {ratio:10.2e} {ratio:*<9.1f} {f64(ratio):E}"
        self.last = line
        log(line, count)
        return len(line)

    def titled(self, title_is_long: bool) -> i32:
        title = "Captain" if not title_is_long else "Grand Admiral of the Fleet"
        self.captain = f"{title:-^30}"
        greeting = "Hello, " + self.captain + "!"
        greeting += " " + f"{self.entries} entries"
        log(greeting, len(self.captain))
        return len(greeting)

    def count_words(self, which: i32) -> i32:
        word = "calm"
        if which == 1:
            word = "storm"
        elif which == 2:
            word = "storm" + "y"
        self.words[word] = self.words.get(word, 0) + 1
        return self.words[word]

    def summary(self) -> i32:
        text = ""
        for word, seen in self.words.items():
            text += f"{word}={seen};"
        log(text, len(self.words))
        self.last = text
        return len(text)

    def forget_while_listing(self) -> i32:
        # The loop's key is a copy: deleting the entry it names leaves the bound key readable.
        kept = 0
        for word in self.words:
            if word == "calm":
                del self.words[word]
                log(f"forgot {word}", len(self.words))
                break
            kept += 1
        return kept

    def field_then_call(self) -> i32:
        # A str field passed before a call that changes it reads as it was before the call.
        self.last = "before"
        log(self.last, self.rewrite())
        return stamp(self.last)

    def rewrite(self) -> i32:
        self.last = "after the rewrite"
        return len(self.last)

    def same_captain(self, other: i32) -> bool:
        name = f"{'Captain':-^30}" if other == 0 else "someone else"
        return name == self.captain

    def early_exit(self, fail: bool) -> i32:
        note = f"entry {self.entries}"
        if fail:
            return stamp(note) - 1000
        note += " closed"
        return stamp(note)

    def unicode(self) -> i32:
        word = "größe"
        line = f"[{word:>8}] [{word:é^9}]"
        log(line, len(word))
        return len(line)
