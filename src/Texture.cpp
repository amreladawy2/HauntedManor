#include "Texture.h"
#include "Paths.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <stb/stb_image.h>

namespace {

// Images are authored in sRGB; sampling through an sRGB format gives linear values to the shader,
// which re-encodes with gamma at the end.
GLuint upload(int w, int h, const unsigned char* rgba, bool smooth) {
    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    const int levels = smooth ? 1 + static_cast<int>(std::log2(std::max(w, h))) : 1;
    glTextureStorage2D(tex, levels, GL_SRGB8_ALPHA8, w, h);
    glTextureSubImage2D(tex, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_REPEAT);
    if (smooth) {
        glGenerateTextureMipmap(tex);
        glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (GLEW_VERSION_4_6 || GLEW_ARB_texture_filter_anisotropic || GLEW_EXT_texture_filter_anisotropic) {
            GLfloat maxAniso = 1.0f;
            glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAniso);
            glTextureParameterf(tex, GL_TEXTURE_MAX_ANISOTROPY, std::min(maxAniso, 16.0f));
        }
    } else {
        glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    return tex;
}

GLuint checkerTexture() {
    static GLuint tex = 0;
    if (!tex) {
        constexpr int N = 8;
        std::vector<unsigned char> px(N * N * 4);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                unsigned char* p = &px[(y * N + x) * 4];
                const bool on = (x + y) % 2 == 0;
                p[0] = on ? 255 : 0; p[1] = 0; p[2] = on ? 255 : 0; p[3] = 255;
            }
        tex = upload(N, N, px.data(), false);
    }
    return tex;
}

} // namespace

GLuint loadTexture(const std::string& path) {
    static std::unordered_map<std::string, GLuint> cache;

    std::error_code ec;
    const std::string resolved = resolvePath(path);
    std::string key = std::filesystem::weakly_canonical(resolved, ec).string();
    if (ec) key = resolved;
    if (auto it = cache.find(key); it != cache.end()) return it->second;

    stbi_set_flip_vertically_on_load(true);   // GL's v=0 is the bottom row
    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(resolved.c_str(), &w, &h, &n, 4);
    GLuint tex;
    if (data) {
        tex = upload(w, h, data, true);
        stbi_image_free(data);
    } else {
        std::cerr << "[Texture] failed to load " << path << ": " << stbi_failure_reason() << "\n";
        tex = checkerTexture();
    }
    cache.emplace(key, tex);   // failures are cached too so the warning prints once
    return tex;
}

GLuint whiteTexture() {
    static GLuint tex = 0;
    if (!tex) {
        const unsigned char white[4] = {255, 255, 255, 255};
        tex = upload(1, 1, white, false);
    }
    return tex;
}
