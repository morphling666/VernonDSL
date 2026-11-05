// RUN: %vernon-opt --vernon-lower-cpu-tensors %s | %FileCheck %s
//
// min, max and clamp select the second operand when the comparison is unordered or equal, as compiled host
// code does, instead of lowering to the NaN-propagating arith.minimumf/maximumf.

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  // CHECK-LABEL: func.func @min_max_clamp
  // CHECK-NOT: arith.minimumf
  // CHECK-NOT: arith.maximumf
  // CHECK: %[[LT:.*]] = arith.cmpf olt, %[[A:.*]], %[[B:.*]] : f64
  // CHECK: arith.select %[[LT]], %[[A]], %[[B]] : f64
  // CHECK: %[[GT:.*]] = arith.cmpf ogt, %[[A]], %[[B]] : f64
  // CHECK: arith.select %[[GT]], %[[A]], %[[B]] : f64
  // CHECK: %[[ABOVE:.*]] = arith.cmpf ogt, %[[A]], %[[B]] : f64
  // CHECK: %[[RAISED:.*]] = arith.select %[[ABOVE]], %[[A]], %[[B]] : f64
  // CHECK: %[[BELOW:.*]] = arith.cmpf olt, %[[RAISED]], %[[C:.*]] : f64
  // CHECK: arith.select %[[BELOW]], %[[RAISED]], %[[C]] : f64
  func.func @min_max_clamp(%a: f64, %b: f64, %c: f64) -> (f64, f64, f64) {
    %min = "vernon.intrinsic"(%a, %b) <{name = "min"}> : (f64, f64) -> f64
    %max = "vernon.intrinsic"(%a, %b) <{name = "max"}> : (f64, f64) -> f64
    %clamp = "vernon.intrinsic"(%a, %b, %c) <{name = "clamp"}> : (f64, f64, f64) -> f64
    return %min, %max, %clamp : f64, f64, f64
  }
}
