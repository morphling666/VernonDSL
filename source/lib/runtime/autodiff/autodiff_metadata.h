#ifndef VERNON_RUNTIME_AUTODIFF_METADATA_H
#define VERNON_RUNTIME_AUTODIFF_METADATA_H

#include <string>
#include <string_view>
#include <vector>

namespace vernon::runtime {

enum class AutodiffDerivativeRole {
    Gradient,
    Cotangent,
};

struct AutodiffDerivativeGroup {
    AutodiffDerivativeRole role{};
    std::string declaredPath;
    std::vector<std::string> leafPaths;
};

inline std::vector<std::string> autodiffDerivativeLeafPaths(const std::vector<AutodiffDerivativeGroup> &groups,
                                                            AutodiffDerivativeRole role) {
    std::vector<std::string> paths;
    for (const AutodiffDerivativeGroup &group : groups)
        if (group.role == role)
            paths.insert(paths.end(), group.leafPaths.begin(), group.leafPaths.end());
    return paths;
}

bool isCanonicalAutodiffPath(std::string_view path);
bool validateAutodiffDerivativeGroups(const std::vector<AutodiffDerivativeGroup> &groups, std::string &error);

} // namespace vernon::runtime

#endif
