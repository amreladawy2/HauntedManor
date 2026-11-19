#include "Primitives.h"

Mesh makeQuad(float w, float h, float uRepeat, float vRepeat, GLuint texture) {
    const float x = w * 0.5f, y = h * 0.5f;
    const glm::vec3 n(0.0f, 0.0f, 1.0f);
    const std::vector<Vertex> v = {
        {{-x, -y, 0.0f}, n, {0.0f, 0.0f}},
        {{ x, -y, 0.0f}, n, {uRepeat, 0.0f}},
        {{ x,  y, 0.0f}, n, {uRepeat, vRepeat}},
        {{-x,  y, 0.0f}, n, {0.0f, vRepeat}},
    };
    return uploadMesh(v, {0, 1, 2, 0, 2, 3}, texture);
}

Mesh makeBox(const glm::vec3& size, float uvPerMeter, GLuint texture) {
    // Per face: outward normal n and in-plane axes u, v with cross(u, v) == n, so quads wind CCW.
    struct Face { glm::vec3 n, u, v; };
    const Face faces[6] = {
        {{ 1, 0, 0}, { 0, 0, -1}, {0, 1,  0}},
        {{-1, 0, 0}, { 0, 0,  1}, {0, 1,  0}},
        {{ 0, 1, 0}, { 1, 0,  0}, {0, 0, -1}},
        {{ 0, -1, 0}, { 1, 0,  0}, {0, 0,  1}},
        {{ 0, 0, 1}, { 1, 0,  0}, {0, 1,  0}},
        {{ 0, 0, -1}, {-1, 0,  0}, {0, 1,  0}},
    };

    std::vector<Vertex> verts;
    std::vector<unsigned> idx;
    for (const Face& f : faces) {
        const float du = glm::dot(glm::abs(f.u), size), dv = glm::dot(glm::abs(f.v), size);
        const glm::vec3 c = f.n * (size * 0.5f), hu = f.u * (du * 0.5f), hv = f.v * (dv * 0.5f);
        const float U = du * uvPerMeter, V = dv * uvPerMeter;
        const unsigned base = static_cast<unsigned>(verts.size());
        verts.push_back({c - hu - hv, f.n, {0, 0}});
        verts.push_back({c + hu - hv, f.n, {U, 0}});
        verts.push_back({c + hu + hv, f.n, {U, V}});
        verts.push_back({c - hu + hv, f.n, {0, V}});
        idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    return uploadMesh(verts, idx, texture);
}
