// RUN: %vernon-opt --vernon-lower-mul-overflow-checks %s | %FileCheck %s

// CHECK-LABEL: func.func @quotient_equals_cofactor
// CHECK-NOT: arith.divui
// CHECK-DAG: %[[HALF:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[MASK:.+]] = arith.constant 65535 : i32
// CHECK-DAG: %[[ZERO:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[AH:.+]] = arith.shrui %arg0, %[[HALF]] : i32
// CHECK-DAG: %[[BH:.+]] = arith.shrui %arg1, %[[HALF]] : i32
// CHECK-DAG: %[[AHZ:.+]] = arith.cmpi eq, %[[AH]], %[[ZERO]] : i32
// CHECK-DAG: %[[BHZ:.+]] = arith.cmpi eq, %[[BH]], %[[ZERO]] : i32
// CHECK: %[[EITHER:.+]] = arith.ori %[[AHZ]], %[[BHZ]] : i1
// CHECK: %[[FITS:.+]] = arith.andi %[[EITHER]], %{{.+}} : i1
// CHECK: return %[[FITS]] : i1
func.func @quotient_equals_cofactor(%a: i32, %b: i32) -> i1 {
  %product = arith.muli %a, %b : i32
  %quotient = arith.divui %product, %a : i32
  %fits = arith.cmpi eq, %quotient, %b : i32
  return %fits : i1
}

// CHECK-LABEL: func.func @commuted_quotient_not_equal
// CHECK-NOT: arith.divui
// CHECK: %[[FITS:.+]] = arith.andi %{{.+}}, %{{.+}} : i1
// CHECK: %[[WRAPS:.+]] = arith.xori %[[FITS]], %{{.+}} : i1
// CHECK: return %[[WRAPS]] : i1
func.func @commuted_quotient_not_equal(%a: i32, %b: i32) -> i1 {
  %product = arith.muli %b, %a : i32
  %quotient = arith.divui %product, %a : i32
  %wraps = arith.cmpi ne, %b, %quotient : i32
  return %wraps : i1
}

// CHECK-LABEL: func.func @bound_upper_limit
// CHECK-NOT: arith.divui
// CHECK: %[[FITS:.+]] = arith.andi %{{.+}}, %{{.+}} : i1
// CHECK: return %[[FITS]] : i1
func.func @bound_upper_limit(%a: i32, %b: i32) -> i1 {
  %max = arith.constant -1 : i32
  %limit = arith.divui %max, %a : i32
  %fits = arith.cmpi ule, %b, %limit : i32
  return %fits : i1
}

// CHECK-LABEL: func.func @bound_exceeded
// CHECK-NOT: arith.divui
// CHECK: %[[WRAPS:.+]] = arith.xori %{{.+}}, %{{.+}} : i1
// CHECK: return %[[WRAPS]] : i1
func.func @bound_exceeded(%a: i32, %b: i32) -> i1 {
  %max = arith.constant -1 : i32
  %limit = arith.divui %max, %a : i32
  %wraps = arith.cmpi ugt, %b, %limit : i32
  return %wraps : i1
}

// CHECK-LABEL: func.func @signed_quotient_equals_cofactor
// CHECK-NOT: arith.divsi
// CHECK-DAG: %[[SMIN:.+]] = arith.constant -2147483648 : i32
// CHECK-DAG: %[[SMAX:.+]] = arith.constant 2147483647 : i32
// CHECK-DAG: %[[AN:.+]] = arith.cmpi slt, %arg0, %{{.+}} : i32
// CHECK-DAG: %[[BN:.+]] = arith.cmpi slt, %arg1, %{{.+}} : i32
// CHECK-DAG: %[[DIFFER:.+]] = arith.xori %[[AN]], %[[BN]] : i1
// CHECK-DAG: %[[LIMIT:.+]] = arith.select %[[DIFFER]], %[[SMIN]], %[[SMAX]] : i32
// CHECK-DAG: %[[WITHIN:.+]] = arith.cmpi ule, %{{.+}}, %[[LIMIT]] : i32
// CHECK: %[[FITS:.+]] = arith.andi %{{.+}}, %[[WITHIN]] : i1
// CHECK: return %[[FITS]] : i1
func.func @signed_quotient_equals_cofactor(%a: i32, %b: i32) -> i1 {
  %product = arith.muli %a, %b : i32
  %quotient = arith.divsi %product, %a : i32
  %fits = arith.cmpi eq, %quotient, %b : i32
  return %fits : i1
}

// CHECK-LABEL: func.func @signed_floor_quotient_not_equal
// CHECK-NOT: arith.floordivsi
// CHECK-DAG: %[[SMIN:.+]] = arith.constant -2147483648 : i32
// CHECK-DAG: %[[SMAX:.+]] = arith.constant 2147483647 : i32
// CHECK: arith.cmpi ule, %{{.+}}, %{{.+}} : i32
// CHECK: %[[WRAPS:.+]] = arith.xori %{{.+}}, %{{.+}} : i1
// CHECK: return %[[WRAPS]] : i1
func.func @signed_floor_quotient_not_equal(%a: i32, %b: i32) -> i1 {
  %product = arith.muli %b, %a : i32
  %quotient = arith.floordivsi %product, %a : i32
  %wraps = arith.cmpi ne, %quotient, %b : i32
  return %wraps : i1
}

// CHECK-LABEL: func.func @wide_vector
// CHECK-NOT: arith.divui
// CHECK-DAG: arith.constant dense<32> : vector<2xi64>
// CHECK-DAG: arith.constant dense<4294967295> : vector<2xi64>
// CHECK: return
func.func @wide_vector(%a: vector<2xi64>, %b: vector<2xi64>) -> vector<2xi1> {
  %product = arith.muli %a, %b : vector<2xi64>
  %quotient = arith.divui %product, %a : vector<2xi64>
  %fits = arith.cmpi eq, %quotient, %b : vector<2xi64>
  return %fits : vector<2xi1>
}

// CHECK-LABEL: func.func @divisor_is_not_a_factor
// CHECK: arith.divui
func.func @divisor_is_not_a_factor(%a: i32, %b: i32, %c: i32) -> i1 {
  %product = arith.muli %a, %b : i32
  %quotient = arith.divui %product, %c : i32
  %same = arith.cmpi eq, %quotient, %b : i32
  return %same : i1
}

// CHECK-LABEL: func.func @quotient_compared_to_unrelated_value
// CHECK: arith.divsi
func.func @quotient_compared_to_unrelated_value(%a: i32, %b: i32, %c: i32) -> i1 {
  %product = arith.muli %a, %b : i32
  %quotient = arith.divsi %product, %a : i32
  %same = arith.cmpi eq, %quotient, %c : i32
  return %same : i1
}

// CHECK-LABEL: func.func @strict_bound_is_not_an_overflow_check
// CHECK: arith.divui
func.func @strict_bound_is_not_an_overflow_check(%a: i32, %b: i32) -> i1 {
  %max = arith.constant -1 : i32
  %limit = arith.divui %max, %a : i32
  %below = arith.cmpi ult, %b, %limit : i32
  return %below : i1
}
