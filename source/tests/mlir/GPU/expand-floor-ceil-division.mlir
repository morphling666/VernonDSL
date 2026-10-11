// RUN: %vernon-opt --vernon-expand-floor-ceil-division %s | %FileCheck %s
// RUN: %vernon-opt --vernon-lower-mul-overflow-checks --vernon-expand-floor-ceil-division %s | %FileCheck %s --check-prefix=ORDERED

// CHECK-LABEL: func.func @floor_division
// CHECK-NOT: arith.floordivsi
// CHECK: arith.divsi
// CHECK: return
func.func @floor_division(%a: i32, %b: i32) -> i32 {
  %quotient = arith.floordivsi %a, %b : i32
  return %quotient : i32
}

// CHECK-LABEL: func.func @floor_division_tensor
// CHECK-NOT: arith.floordivsi
// CHECK: arith.divsi %{{.+}}, %{{.+}} : tensor<4xi32>
// CHECK: return
func.func @floor_division_tensor(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %quotient = arith.floordivsi %a, %b : tensor<4xi32>
  return %quotient : tensor<4xi32>
}

// CHECK-LABEL: func.func @ceil_division
// CHECK-NOT: arith.ceildivsi
// CHECK-NOT: arith.ceildivui
// CHECK: return
func.func @ceil_division(%a: i32, %b: i32) -> (i32, i32) {
  %signed = arith.ceildivsi %a, %b : i32
  %unsigned = arith.ceildivui %a, %b : i32
  return %signed, %unsigned : i32, i32
}

// ORDERED-LABEL: func.func @overflow_check_survives_expansion
// ORDERED-NOT: arith.divsi
// ORDERED-NOT: arith.floordivsi
// ORDERED: return
func.func @overflow_check_survives_expansion(%a: i32, %b: i32) -> i1 {
  %product = arith.muli %a, %b : i32
  %quotient = arith.floordivsi %product, %a : i32
  %fits = arith.cmpi eq, %quotient, %b : i32
  return %fits : i1
}
