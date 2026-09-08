#ifndef VERNON_RUNTIME_PROGRAM_INVOCATION_CONTEXT_H
#define VERNON_RUNTIME_PROGRAM_INVOCATION_CONTEXT_H

namespace vernon::runtime::program {
class InvocationSnapshot;
}

namespace vernon::runtime {

struct ProgramInvocationContext {
    const program::InvocationSnapshot &controls;
};

} // namespace vernon::runtime

#endif
