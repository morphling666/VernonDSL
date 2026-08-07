#include "autodiff_metadata.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace vernon::runtime {
namespace {

bool validPathIdentifier(std::string_view component) {
    if (component.empty() || !(std::isalpha(static_cast<unsigned char>(component.front())) || component.front() == '_'))
        return false;
    return std::all_of(component.begin() + 1, component.end(), [](char value) {
        const unsigned char character = static_cast<unsigned char>(value);
        return std::isalnum(character) || character == '_';
    });
}

bool pathContains(std::string_view declared, std::string_view leaf) {
    return leaf == declared || (leaf.size() > declared.size() && leaf.compare(0, declared.size(), declared) == 0 &&
                                leaf[declared.size()] == '.');
}

} // namespace

bool isCanonicalAutodiffPath(std::string_view path) {
    if (path.empty())
        return false;
    size_t begin = 0;
    size_t componentIndex = 0;
    while (begin <= path.size()) {
        const size_t end = path.find('.', begin);
        const std::string_view component =
            path.substr(begin, end == std::string_view::npos ? path.size() - begin : end - begin);
        const bool numeric = componentIndex != 0 && !component.empty() &&
                             std::all_of(component.begin(), component.end(),
                                         [](char value) { return std::isdigit(static_cast<unsigned char>(value)); });
        if (!numeric && !validPathIdentifier(component))
            return false;
        ++componentIndex;
        if (end == std::string_view::npos)
            return true;
        begin = end + 1;
    }
    return false;
}

bool validateAutodiffDerivativeGroups(const std::vector<AutodiffDerivativeGroup> &groups, std::string &error) {
    if (groups.empty()) {
        error = "autodiff derivative groups are empty";
        return false;
    }
    bool seenCotangent = false;
    size_t gradientCount = 0;
    size_t cotangentCount = 0;
    std::string previousGradient;
    std::string previousCotangent;
    std::set<std::string> gradientLeaves;
    std::set<std::string> cotangentLeaves;
    std::vector<std::string> orderedGradientLeaves;
    for (const AutodiffDerivativeGroup &group : groups) {
        if (group.role != AutodiffDerivativeRole::Gradient && group.role != AutodiffDerivativeRole::Cotangent) {
            error = "autodiff derivative group role is invalid";
            return false;
        }
        if (group.role == AutodiffDerivativeRole::Cotangent)
            seenCotangent = true;
        else if (seenCotangent) {
            error = "autodiff derivative groups are not in canonical role order";
            return false;
        }
        std::string &previous = group.role == AutodiffDerivativeRole::Gradient ? previousGradient : previousCotangent;
        std::set<std::string> &leaves =
            group.role == AutodiffDerivativeRole::Gradient ? gradientLeaves : cotangentLeaves;
        size_t &groupCount = group.role == AutodiffDerivativeRole::Gradient ? gradientCount : cotangentCount;
        if (!isCanonicalAutodiffPath(group.declaredPath) || group.leafPaths.empty() ||
            (!previous.empty() && previous >= group.declaredPath)) {
            error = "autodiff derivative group declarations are not canonical";
            return false;
        }
        previous = group.declaredPath;
        ++groupCount;
        for (const std::string &leaf : group.leafPaths) {
            if (!isCanonicalAutodiffPath(leaf) || !pathContains(group.declaredPath, leaf) ||
                !leaves.insert(leaf).second) {
                error = "autodiff derivative group leaves are not canonical";
                return false;
            }
            size_t owners = 0;
            for (const AutodiffDerivativeGroup &candidate : groups)
                if (candidate.role == group.role && pathContains(candidate.declaredPath, leaf))
                    ++owners;
            if (owners != 1) {
                error = "autodiff derivative leaf does not have exactly one declared owner";
                return false;
            }
            if (group.role == AutodiffDerivativeRole::Gradient)
                orderedGradientLeaves.push_back(leaf);
        }
    }
    if (!gradientCount || !cotangentCount) {
        error = "autodiff requires gradient and cotangent derivative groups";
        return false;
    }
    if (!std::is_sorted(orderedGradientLeaves.begin(), orderedGradientLeaves.end())) {
        error = "autodiff gradient leaves are not in canonical order";
        return false;
    }
    return true;
}

} // namespace vernon::runtime
