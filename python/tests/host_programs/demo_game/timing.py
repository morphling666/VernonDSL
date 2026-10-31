from typing import Annotated

import vernon_dsl as vd
from vernon_dsl import f64, i32


@vd.host
class Timer:
    """Fires once per period of accumulated time."""

    period: Annotated[f64, vd.meta(min=0.0, unit="s")] = 0.5
    elapsed: f64 = 0.0
    fired: i32

    @vd.host(effects="pure")
    def tick(self, dt: f64) -> bool:
        self.elapsed += dt
        if self.elapsed >= self.period:
            self.elapsed -= self.period
            self.fired += 1
            return True
        return False
