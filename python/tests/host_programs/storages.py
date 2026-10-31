"""TensorStorage fields and locals: numpy-like containers owned by Host code, run alike in Python mode and AOT."""

import vernon_dsl as vd
from vernon_dsl import Tensor, TensorStorage, dyn, f32, f64, i32, u32


@vd.extern(symbol="storages_log")
def log(value: i32) -> None: ...


@vd.host
class Grid:
    """A height grid sized at run time, a fixed 2x3 tile, and a running histogram."""

    heights: TensorStorage[f32, (dyn, dyn)]
    tile: TensorStorage[i32, (2, 3)]
    histogram: TensorStorage[u32, (dyn,)]
    weights: TensorStorage[f64, (4,)] = vd.zeros(dtype=f64, shape=(4,))
    cells: i32 = 0

    def resize(self, rows: i32, columns: i32) -> i32:
        self.heights = vd.zeros(dtype=f32, shape=(rows, columns))
        self.histogram = vd.empty(dtype=u32, shape=(columns,))
        self.histogram.fill(0)
        self.cells = rows * columns
        return len(self.heights)

    def raise_cell(self, row: i32, column: i32, amount: f32) -> f32:
        self.heights[row, column] += amount
        self.histogram[column] += 1
        return self.heights[row, column]

    def corner(self) -> f32:
        return self.heights[-1, -1]

    def shape_sum(self) -> i32:
        rows, columns = self.heights.shape
        return rows * 100 + columns + self.heights.shape[1]

    def load_tile(self, values: Tensor[i32, 2, 3]) -> i32:
        self.tile.copy_from_values(values)
        total = 0
        for row in range(2):
            for column in range(3):
                total += self.tile[row, column]
        return total

    def read_tile(self) -> Tensor[i32, 2, 3]:
        self.tile[0, 0] = 7
        return self.tile.to_values()

    def load_heights(self, values: Tensor[f32, 2, 2]) -> i32:
        self.heights.copy_from_values(values)
        return len(self.heights)

    def adopt_heights(self, values: Tensor[f32, 3, 2]) -> f32:
        self.heights = vd.from_values(values, dtype=f32)
        rows, columns = self.heights.shape
        return self.heights[-1, 1] + f32(rows * 10 + columns)

    def blurred_total(self) -> f32:
        scratch = vd.zeros_like(self.heights)
        rows, columns = self.heights.shape
        total = f32(0)
        for row in range(rows):
            for column in range(columns):
                scratch[row, column] = self.heights[row, column] * 0.5
                total += scratch[row, column]
        log(len(scratch))
        return total

    def weighted(self) -> f64:
        self.weights[0] = 0.5
        self.weights[3] = 2.0
        local: TensorStorage[f64, (dyn,)] = vd.empty_like(self.weights)
        local.fill(1.5)
        total = f64(0)
        for index in range(4):
            total += local[index] * self.weights[index]
        return total

    def out_of_range(self, row: i32) -> f32:
        return self.heights[row, 0]

    def histogram_total(self) -> u32:
        total = u32(0)
        for index in range(len(self.histogram)):
            total += self.histogram[index]
        return total

    def slab(self, row: i32) -> f32:
        self.heights[row, 0:2] = 1.5
        self.heights[row : row + 2, 1] += self.heights[row : row + 2, 0]
        pair = self.heights[row : row + 2, 1]
        return pair[0] * 10.0 + pair[1]

    def tile_block(self, column: i32) -> i32:
        self.tile[:, column : column + 2] = self.tile[:, 0:2] * 2
        self.tile[1] = -1
        block = self.tile[::-1, 1:]
        return block[0, 0] * 100 + block[1, 1]
