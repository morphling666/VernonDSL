#include "compiler_spirv.h"

#include "mlir/Dialect/Vernon/Transforms/VernonSpirvMarkers.h"

#include <gtest/gtest.h>
#include <string>

namespace {

void appendInstruction(llvm::SmallVectorImpl<uint32_t> &words, uint16_t opcode,
                       std::initializer_list<uint32_t> operands) {
    words.push_back((static_cast<uint32_t>(operands.size() + 1) << 16) | opcode);
    words.append(operands);
}

llvm::SmallVector<uint32_t> spirvHeader(uint32_t idBound) { return {0x07230203u, 0x00010000u, 0, idBound, 0}; }

} // namespace

TEST(SpirvImageQuerySizeLodMaterialization, MaterializesOnlyCompleteMarkerSequence) {
    llvm::SmallVector<uint32_t> words = spirvHeader(20);
    appendInstruction(words, 43, {1, 10, mlir::vernon::kImageQueryLodMarker});
    appendInstruction(words, 43, {1, 11, mlir::vernon::kImageQueryResultMarkerA});
    appendInstruction(words, 43, {1, 12, mlir::vernon::kImageQueryResultMarkerB});
    appendInstruction(words, 44, {2, 13, 11, 11});
    appendInstruction(words, 44, {2, 14, 12, 12});
    appendInstruction(words, 100, {3, 15, 16});
    appendInstruction(words, 128, {1, 17, 18, 10});
    const size_t queryOffset = words.size();
    appendInstruction(words, 128, {2, 19, 13, 14});

    std::string diagnostics;
    ASSERT_TRUE(vernon::compiler_detail::materializeImageQuerySizeLod(words, 1, diagnostics)) << diagnostics;
    EXPECT_EQ(static_cast<uint16_t>(words[queryOffset]), 103);
    EXPECT_EQ(words[queryOffset + 3], 15u);
    EXPECT_EQ(words[queryOffset + 4], 18u);
}

TEST(SpirvImageQuerySizeLodMaterialization, DoesNotMaterializeOrdinaryImageAndAdds) {
    llvm::SmallVector<uint32_t> words = spirvHeader(12);
    appendInstruction(words, 100, {1, 2, 3});
    appendInstruction(words, 128, {4, 5, 6, 7});
    appendInstruction(words, 128, {8, 9, 10, 11});
    const llvm::SmallVector<uint32_t> original = words;

    std::string diagnostics;
    EXPECT_TRUE(vernon::compiler_detail::materializeImageQuerySizeLod(words, 0, diagnostics));
    EXPECT_EQ(words, original);

    EXPECT_FALSE(vernon::compiler_detail::materializeImageQuerySizeLod(words, 1, diagnostics));
    EXPECT_NE(diagnostics.find("replacement count mismatch"), std::string::npos);
}

TEST(SpirvImageQuerySizeLodMaterialization, RejectsUnpairedMarkers) {
    llvm::SmallVector<uint32_t> words = spirvHeader(8);
    appendInstruction(words, 43, {1, 2, mlir::vernon::kImageQueryLodMarker});
    appendInstruction(words, 128, {1, 3, 4, 2});

    std::string diagnostics;
    EXPECT_FALSE(vernon::compiler_detail::materializeImageQuerySizeLod(words, 1, diagnostics));
    EXPECT_NE(diagnostics.find("replacement count mismatch"), std::string::npos);
}
