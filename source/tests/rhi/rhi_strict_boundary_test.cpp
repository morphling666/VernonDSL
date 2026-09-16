#include "rhi/public_c_boundary.h"
#include "rhi/rhi_pending_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

using Write = std::pair<vernon::rhi::ResourceKind, uint64_t>;

struct ThrowingWrites {
    using value_type = Write;

    void insert(value_type) { throw std::bad_alloc{}; }
};

TEST(RhiPendingWriteTest, AllocationFailureRemainsConservativelyUnknown) {
    ThrowingWrites writes;
    bool unknown = false;

    vernon::rhi::recordPendingWrite(writes, unknown, vernon::rhi::ResourceKind::Buffer, UINT64_C(0x1234));

    EXPECT_TRUE(unknown);
}

TEST(RhiPendingWriteTest, SuccessfulInsertionRestoresKnownState) {
    struct Hash {
        size_t operator()(const Write &write) const noexcept {
            return static_cast<size_t>(write.second) ^ static_cast<size_t>(write.first);
        }
    };
    std::unordered_set<Write, Hash> writes;
    bool unknown = false;

    vernon::rhi::recordPendingWrite(writes, unknown, vernon::rhi::ResourceKind::Image, UINT64_C(7));
    EXPECT_FALSE(unknown);
    EXPECT_EQ(writes.count({vernon::rhi::ResourceKind::Image, UINT64_C(7)}), 1u);
}

TEST(RhiPublicHandleBoundaryTest, PreservesBackendCreationDiagnostic) {
    vernon::rhi::setDeviceCreationError("specific backend diagnostic");
    auto result = vernon::rhi::publicHandleBoundary(
        [] {
            vernon::rhi::setDeviceCreationError("specific backend diagnostic");
            return vernon::Result<uint64_t, vernon::RhiError>{
                vernon::err(vernon::RhiError{vernon::RhiErrorCode::BackendFailure, {"create_test_device", 0, 0}})};
        },
        UINT64_C(0));

    EXPECT_EQ(result, 0u);
    VernonStringView diagnostic = vernon::rhi::deviceCreationError();
    EXPECT_EQ(std::string(diagnostic.data, diagnostic.size), "specific backend diagnostic");
}

TEST(RhiPublicHandleBoundaryTest, SynthesizesStableDiagnosticWhenBackendProvidesNone) {
    auto result = vernon::rhi::publicHandleBoundary(
        [] {
            return vernon::Result<uint64_t, vernon::RhiError>{
                vernon::err(vernon::RhiError{vernon::RhiErrorCode::Unsupported, {"create_test_device", 0, 0}})};
        },
        UINT64_C(0));

    EXPECT_EQ(result, 0u);
    VernonStringView diagnostic = vernon::rhi::deviceCreationError();
    EXPECT_EQ(std::string(diagnostic.data, diagnostic.size), "RHI device creation is unsupported");
}

TEST(RhiPublicHandleBoundaryTest, AllocationExceptionUsesBoundedEmergencyDiagnostic) {
    auto result = vernon::rhi::publicHandleBoundary(
        []() -> vernon::Result<uint64_t, vernon::RhiError> { throw std::bad_alloc{}; }, UINT64_C(0));

    EXPECT_EQ(result, 0u);
    VernonStringView diagnostic = vernon::rhi::deviceCreationError();
    EXPECT_EQ(std::string(diagnostic.data, diagnostic.size), "RHI device creation exhausted resources");
}

TEST(RhiPublicHandleBoundaryTest, UnknownExceptionUsesBoundedEmergencyDiagnostic) {
    auto result = vernon::rhi::publicHandleBoundary(
        []() -> vernon::Result<uint64_t, vernon::RhiError> { throw std::runtime_error("creation escaped"); },
        UINT64_C(0));

    EXPECT_EQ(result, 0u);
    VernonStringView diagnostic = vernon::rhi::deviceCreationError();
    EXPECT_EQ(std::string(diagnostic.data, diagnostic.size), "RHI device creation failed at the C boundary");
}

TEST(RhiPublicDestroyBoundaryTest, AdmissionRefusalsDoNotViolateContract) {
    EXPECT_NO_FATAL_FAILURE(vernon::rhi::publicDestroyBoundary([] {
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"destroy_test_device", 0, 0}})};
    }));
    EXPECT_NO_FATAL_FAILURE(vernon::rhi::publicDestroyBoundary([] {
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure, {"destroy_test_device", 0, 0}})};
    }));
}

TEST(RhiPublicDestroyBoundaryTest, BackendTeardownFailureViolatesContract) {
    EXPECT_DEATH(vernon::rhi::publicDestroyBoundary([] {
                     return vernon::Result<void, vernon::RhiError>{vernon::err(
                         vernon::RhiError{vernon::RhiErrorCode::BackendFailure, {"destroy_test_device", 0, 0}})};
                 }),
                 "");
}

TEST(RhiPublicDestroyBoundaryTest, EscapingExceptionViolatesContract) {
    EXPECT_DEATH(vernon::rhi::publicDestroyBoundary(
                     []() -> vernon::Result<void, vernon::RhiError> { throw std::runtime_error("teardown escaped"); }),
                 "");
}

} // namespace
