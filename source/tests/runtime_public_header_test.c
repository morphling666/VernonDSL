#include <vernon-c/Runtime.h>

int main(void) {
  VernonRuntimeCreateOptions options = {0};
  VernonCpuInvocation invocation = {0};
  options.struct_size = sizeof(options);
  return options.struct_size == 0 || invocation.arguments != 0;
}
