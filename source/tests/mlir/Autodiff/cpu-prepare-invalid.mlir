// RUN: %not %vernon-opt --vernon-prepare-cpu-autodiff-signatures %s 2>&1 | %FileCheck %s
//
// CHECK: dynamic CPU autodiff backward has an invalid logical tape signature

module {
  func.func @invalid_backward(%root: !vernon.ad_region_header)
      attributes {vernon.entry, vernon.stage = "compute"} {
    %count = "vernon.ad.read_executed_count"(%root)
        : (!vernon.ad_region_header) -> index
    return
  }
}
