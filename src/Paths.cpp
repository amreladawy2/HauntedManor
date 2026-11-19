#include "Paths.h"

#include <filesystem>
#include <system_error>

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

std::string resolvePath(const std::string& relative) {
    std::error_code ec;
    const fs::path rel(relative);

    fs::path dir = fs::current_path(ec);
    for (int up = 0; up <= 3 && !ec; ++up) {
        if (fs::exists(dir / rel, ec)) return (dir / rel).lexically_normal().string();
        if (!dir.has_parent_path() || dir.parent_path() == dir) break;
        dir = dir.parent_path();
    }

    const fs::path rooted = fs::path(PROJECT_ROOT) / rel;
    if (fs::exists(rooted, ec)) return rooted.lexically_normal().string();
    return relative;
}
