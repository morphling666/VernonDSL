#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::vernon {

enum class AggregateStorageBackend {
    MemRef,
    WorkgroupTensorView,
};

void populateCpuAggregateTensorViewPatterns(TypeConverter &converter, RewritePatternSet &patterns, ModuleOp module);

LogicalResult lowerGpuAggregateWorkgroupStorage(gpu::GPUFuncOp kernel, ModuleOp module);

FailureOr<Value> loadAggregateRecordFromStorages(Type elementType, ValueRange storages, Value recordIndex,
                                                 const ValueAbiLayout &layout, ModuleOp module, OpBuilder &builder,
                                                 Location location, AggregateStorageBackend backend);

LogicalResult storeAggregateRecordToStorages(Type elementType, ValueRange storages, Value recordIndex, Value value,
                                             const ValueAbiLayout &layout, ModuleOp module, OpBuilder &builder,
                                             Location location, AggregateStorageBackend backend);

} // namespace mlir::vernon
