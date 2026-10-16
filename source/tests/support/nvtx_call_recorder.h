#ifndef VERNON_TESTS_SUPPORT_NVTX_CALL_RECORDER_H
#define VERNON_TESTS_SUPPORT_NVTX_CALL_RECORDER_H

#include <string>
#include <vector>

#if defined(_WIN32)
#if defined(VERNON_NVTX_CALL_RECORDER_BUILD)
#define VERNON_NVTX_CALL_RECORDER_API __declspec(dllexport)
#else
#define VERNON_NVTX_CALL_RECORDER_API __declspec(dllimport)
#endif
#else
#define VERNON_NVTX_CALL_RECORDER_API __attribute__((visibility("default")))
#endif

namespace vernon::tests {

// Returns and forgets the NVTX calls made in the "Vernon" domain, as "start <message>", "end <message>" and
// "mark <message>" in call order. NVTX reaches this library only when NVTX_INJECTION64_PATH names it before the
// process makes its first NVTX call.
VERNON_NVTX_CALL_RECORDER_API std::vector<std::string> takeNvtxCalls();

} // namespace vernon::tests

#endif
