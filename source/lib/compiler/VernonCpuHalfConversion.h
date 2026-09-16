#ifndef VERNON_CPU_HALF_CONVERSION_H
#define VERNON_CPU_HALF_CONVERSION_H

namespace llvm {
class Error;
class Module;
class TargetMachine;
} // namespace llvm

namespace vernon {

llvm::Error lowerCpuHalfConversions(llvm::Module &module, const llvm::TargetMachine &targetMachine);

} // namespace vernon

#endif
