// Layout checker for src/Scene.cpp: builds both scenes with stub models whose AABBs come from the
// docs/CONTRACT.md size table (no GL, no assets), validates placement/interactions/lights and prints a
// top-down ASCII map of each room. Exit code 0 = no failures.
// Build + run from HauntedManor/:
//   mkdir -p build && g++ -std=c++20 -I src tools/check_layout.cpp src/Scene.cpp -o build/check_layout.exe && ./build/check_layout.exe
#include "Model.h"
#include "Scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

namespace {

enum class Origin { Bottom, Wall, Hang };
struct Spec {
    float w, h, d;
    Origin origin;
};

// Copied from docs/CONTRACT.md (W x H x D, metres).
const std::map<std::string, Spec> kSpecs = {
    {"player", {0.55f, 1.80f, 0.35f, Origin::Bottom}},
    {"lib_bookshelf", {2.0f, 2.6f, 0.45f, Origin::Bottom}},
    {"lib_desk", {1.6f, 0.78f, 0.8f, Origin::Bottom}},
    {"lib_armchair", {0.9f, 1.15f, 0.9f, Origin::Bottom}},
    {"lib_fireplace", {2.2f, 1.6f, 0.6f, Origin::Wall}},
    {"lib_clock", {0.6f, 2.3f, 0.4f, Origin::Bottom}},
    {"lib_globe", {0.7f, 1.15f, 0.7f, Origin::Bottom}},
    {"lib_candelabra", {0.45f, 0.6f, 0.15f, Origin::Bottom}},
    {"lib_side_table", {0.6f, 0.7f, 0.6f, Origin::Bottom}},
    {"lib_rug", {3.0f, 0.02f, 2.0f, Origin::Bottom}},
    {"lib_portrait", {1.0f, 1.3f, 0.08f, Origin::Wall}},
    {"lib_bust", {0.45f, 1.65f, 0.45f, Origin::Bottom}},
    {"lib_chandelier", {1.0f, 0.9f, 1.0f, Origin::Hang}},
    {"bed_bed", {1.8f, 2.1f, 2.2f, Origin::Bottom}},
    {"bed_wardrobe", {1.4f, 2.2f, 0.6f, Origin::Bottom}},
    {"bed_rocking_chair", {0.65f, 1.1f, 1.0f, Origin::Bottom}},
    {"bed_nightstand", {0.5f, 0.62f, 0.42f, Origin::Bottom}},
    {"bed_lamp", {0.25f, 0.5f, 0.25f, Origin::Bottom}},
    {"bed_dresser", {1.2f, 1.95f, 0.5f, Origin::Bottom}},
    {"bed_music_box", {0.26f, 0.16f, 0.18f, Origin::Bottom}},
    {"bed_toy_chest", {0.9f, 0.55f, 0.5f, Origin::Bottom}},
    {"bed_doll", {0.22f, 0.42f, 0.25f, Origin::Bottom}},
    {"bed_rug", {2.4f, 0.02f, 2.4f, Origin::Bottom}},
    {"bed_window", {1.6f, 2.2f, 0.2f, Origin::Wall}},
    {"lib_map", {0.45f, 0.06f, 0.32f, Origin::Bottom}},    // quest extension (v2)
    {"bed_key", {0.16f, 0.02f, 0.05f, Origin::Bottom}},
};

// Quest items: one per scene on top of the four regular interactions.
const std::set<std::string> kQuestItems = {"lib_map", "bed_key"};

// Heights of the surfaces small items rest on, measured in the generated OBJs (desk: wood top under the
// inkwell; toy chest: lid between the iron bands; dresser: drawer top below the mirror).
const std::map<std::string, float> kSurfaceTop = {
    {"lib_side_table", 0.70f}, {"lib_desk", 0.725f},  {"bed_nightstand", 0.62f},
    {"bed_toy_chest", 0.541f}, {"bed_dresser", 0.90f},
};

const std::set<std::string> kSounds = {"footstep", "interact",        "bump",            "click",  "switch",
                                       "horror_on", "heartbeat",      "whisper",         "ambient_library",
                                       "ambient_bedroom", "chime",    "music_box",       "light_on", "creak",
                                       "flashlight", "pickup",        "objective",       "riddle", "win",
                                       "scream",     "scare_hit"};

constexpr float kPlayerRadius = 0.25f;   // Player::radius
constexpr float kComfortRadius = 0.35f;  // wider body used to flag cramped paths
constexpr float kFovY = 60.0f, kAspect = 16.0f / 9.0f;

} // namespace

Model* ModelCache::get(const std::string& id) {
    auto found = models_.find(id);
    if (found != models_.end()) return found->second.get();
    auto spec = kSpecs.find(id);
    if (spec == kSpecs.end()) {
        std::printf("FAIL: unknown model id '%s'\n", id.c_str());
        std::exit(1);
    }
    const Spec& s = spec->second;
    auto m = std::make_unique<Model>();
    switch (s.origin) {
    case Origin::Bottom:
        m->boundsMin = {-s.w / 2, 0.0f, -s.d / 2};
        m->boundsMax = {s.w / 2, s.h, s.d / 2};
        break;
    case Origin::Wall:
        m->boundsMin = {-s.w / 2, 0.0f, 0.0f};
        m->boundsMax = {s.w / 2, s.h, s.d};
        break;
    case Origin::Hang:
        m->boundsMin = {-s.w / 2, -s.h, -s.d / 2};
        m->boundsMax = {s.w / 2, 0.0f, s.d / 2};
        break;
    }
    Model* raw = m.get();
    models_[id] = std::move(m);
    return raw;
}

namespace {

int gFailures = 0, gWarnings = 0;

template <typename... A> void fail(const char* fmt, A... a) {
    ++gFailures;
    std::printf("  FAIL: ");
    std::printf(fmt, a...);
    std::printf("\n");
}
template <typename... A> void warn(const char* fmt, A... a) {
    ++gWarnings;
    std::printf("  WARN: ");
    std::printf(fmt, a...);
    std::printf("\n");
}

struct Box3 {
    glm::vec3 min{1e9f}, max{-1e9f};
    void add(const glm::vec3& p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }
    void add(const Box3& b) {
        add(b.min);
        add(b.max);
    }
};

Box3 worldBox(const SceneObject& o) {
    Box3 b;
    const glm::mat4 m = o.modelMatrix();
    for (int i = 0; i < 8; ++i) {
        glm::vec3 c((i & 1) ? o.model->boundsMax.x : o.model->boundsMin.x,
                    (i & 2) ? o.model->boundsMax.y : o.model->boundsMin.y,
                    (i & 4) ? o.model->boundsMax.z : o.model->boundsMin.z);
        b.add(glm::vec3(m * glm::vec4(c, 1.0f)));
    }
    return b;
}

Box3 worldBoxOf(const SceneObject& o, const glm::vec3& lmin, const glm::vec3& lmax) {
    Box3 b;
    const glm::mat4 m = o.modelMatrix();
    for (int i = 0; i < 8; ++i) {
        glm::vec3 c((i & 1) ? lmax.x : lmin.x, (i & 2) ? lmax.y : lmin.y, (i & 4) ? lmax.z : lmin.z);
        b.add(glm::vec3(m * glm::vec4(c, 1.0f)));
    }
    return b;
}

// Volume an object really occupies above the floor: a surface with something standing on it (the
// dresser's mirror) is split into the body below the surface plus a 0.12 m slab at the back above it.
std::vector<Box3> occupiedParts(const SceneObject& o) {
    const glm::vec3 lo = o.model->boundsMin, hi = o.model->boundsMax;
    if (o.name == "bed_bed") {
        // Four corner legs (0.085 m square, measured in bed_bed.obj); frame and mattress start at y = 0.276.
        std::vector<Box3> p{worldBoxOf(o, {lo.x, 0.276f, lo.z}, hi)};
        for (float sx : {-1.0f, 1.0f})
            for (float sz : {-1.0f, 1.0f})
                p.push_back(worldBoxOf(o, {sx > 0 ? hi.x - 0.085f : lo.x, 0.0f, sz > 0 ? hi.z - 0.085f : lo.z},
                                       {sx > 0 ? hi.x : lo.x + 0.085f, 0.276f, sz > 0 ? hi.z : lo.z + 0.085f}));
        return p;
    }
    auto top = kSurfaceTop.find(o.name);
    if (top == kSurfaceTop.end() || top->second >= hi.y - 1e-3f) return {worldBox(o)};
    return {worldBoxOf(o, lo, {hi.x, top->second, hi.z}),
            worldBoxOf(o, {lo.x, top->second, lo.z}, {hi.x, hi.y, lo.z + 0.12f})};
}

bool overlapXZ(const Box3& a, const Box3& b, float eps = 1e-3f) {
    return a.min.x < b.max.x - eps && b.min.x < a.max.x - eps && a.min.z < b.max.z - eps && b.min.z < a.max.z - eps;
}
bool overlap3(const Box3& a, const Box3& b, float eps = 1e-3f) {
    return overlapXZ(a, b, eps) && a.min.y < b.max.y - eps && b.min.y < a.max.y - eps;
}
bool inside3(const glm::vec3& p, const Box3& b) {
    return p.x > b.min.x && p.x < b.max.x && p.y > b.min.y && p.y < b.max.y && p.z > b.min.z && p.z < b.max.z;
}
float distToBox(const glm::vec3& p, const Box3& b) { return glm::length(glm::max(glm::max(b.min - p, p - b.max), glm::vec3(0.0f))); }

// Slab test: does segment p0->p1 pass through box b (shrunk by eps)?
bool segmentHitsBox(const glm::vec3& p0, const glm::vec3& p1, Box3 b, float eps = 0.02f) {
    b.min += eps;
    b.max -= eps;
    float t0 = 0.0f, t1 = 1.0f;
    const glm::vec3 d = p1 - p0;
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(d[k]) < 1e-9f) {
            if (p0[k] < b.min[k] || p0[k] > b.max[k]) return false;
            continue;
        }
        float a = (b.min[k] - p0[k]) / d[k], c = (b.max[k] - p0[k]) / d[k];
        if (a > c) std::swap(a, c);
        t0 = std::max(t0, a);
        t1 = std::min(t1, c);
        if (t0 > t1) return false;
    }
    return true;
}

bool isRug(const SceneObject& o) { return o.name.find("rug") != std::string::npos; }
Origin originOf(const SceneObject& o) { return kSpecs.at(o.name).origin; }

std::string animNames(unsigned a) {
    std::string s;
    const char* names[] = {"ROTATE", "SWING", "TRANSLATE", "SCALE", "COLOR"};
    for (int i = 0; i < 5; ++i)
        if (a & (1u << i)) s += (s.empty() ? "" : "+") + std::string(names[i]);
    return s;
}

// Box of the object in its held end pose (translated, scaled up about the pivot).
Box3 finalBox(const SceneObject& o) {
    SceneObject t = o;
    if (o.interaction.anims & ANIM_TRANSLATE) t.animTranslate = o.interaction.translateBy;
    if (o.interaction.anims & ANIM_SCALE) t.animScale = std::max(1.0f, o.interaction.scaleTo);
    return worldBox(t);
}

// Box of the object over its whole interaction (static + translated + scaled about the pivot).
Box3 animatedBox(const SceneObject& o) {
    Box3 b = worldBox(o);
    SceneObject t = o;
    if (o.interaction.anims & ANIM_TRANSLATE) {
        t.animTranslate = o.interaction.translateBy;
        b.add(worldBox(t));
    }
    if (o.interaction.anims & ANIM_SCALE) {
        t.animScale = std::max(1.0f, o.interaction.scaleTo);
        b.add(worldBox(t));
    }
    return b;
}

struct Grid {
    float x0, z0, step;
    int nx, nz;
    std::vector<char> freeCell, reach;
    glm::vec2 at(int i, int j) const { return {x0 + (i + 0.5f) * step, z0 + (j + 0.5f) * step}; }
};

bool blockedAt(const glm::vec2& p, const std::vector<Box3>& solids, float r) {
    for (const Box3& b : solids)
        if (p.x > b.min.x - r && p.x < b.max.x + r && p.y > b.min.z - r && p.y < b.max.z + r) return true;
    return false;
}

Grid flood(const Scene& s, const std::vector<Box3>& solids, float r, float step = 0.05f) {
    Grid g{s.boundsMin.x, s.boundsMin.y, step, 0, 0, {}, {}};
    g.nx = int(std::ceil((s.boundsMax.x - s.boundsMin.x) / step));
    g.nz = int(std::ceil((s.boundsMax.y - s.boundsMin.y) / step));
    g.freeCell.assign(size_t(g.nx) * g.nz, 0);
    g.reach.assign(size_t(g.nx) * g.nz, 0);
    for (int j = 0; j < g.nz; ++j)
        for (int i = 0; i < g.nx; ++i) {
            glm::vec2 p = g.at(i, j);
            g.freeCell[j * g.nx + i] = p.x <= s.boundsMax.x && p.y <= s.boundsMax.y && !blockedAt(p, solids, r);
        }
    int si = std::clamp(int((s.spawnPos.x - g.x0) / step), 0, g.nx - 1);
    int sj = std::clamp(int((s.spawnPos.z - g.z0) / step), 0, g.nz - 1);
    if (!g.freeCell[sj * g.nx + si]) return g;
    std::queue<std::pair<int, int>> q;
    q.push({si, sj});
    g.reach[sj * g.nx + si] = 1;
    while (!q.empty()) {
        auto [i, j] = q.front();
        q.pop();
        const int di[] = {1, -1, 0, 0}, dj[] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            int a = i + di[k], b = j + dj[k];
            if (a < 0 || b < 0 || a >= g.nx || b >= g.nz) continue;
            size_t idx = size_t(b) * g.nx + a;
            if (g.freeCell[idx] && !g.reach[idx]) {
                g.reach[idx] = 1;
                q.push({a, b});
            }
        }
    }
    return g;
}

// Mirrors Game::findEligible: nearest enabled object (XZ distance to its position) within its range whose
// centre lies inside the player's horizontal view cone. Quest gating is ignored (worst case: all enabled).
int pickAt(const Scene& s, const glm::vec2& p, const glm::vec2& fwd) {
    int best = -1;
    float bestDist = 1e9f;
    for (size_t i = 0; i < s.objects.size(); ++i) {
        const SceneObject& o = s.objects[i];
        const Interaction& it = o.interaction;
        if (!it.enabled) continue;
        const float dist = glm::length(glm::vec2(o.position.x, o.position.z) - p);
        if (dist > it.range) continue;
        const glm::vec3 c = o.worldCenter();
        const glm::vec2 to(c.x - p.x, c.z - p.y);
        if (glm::length(to) > 1e-3f && glm::dot(glm::normalize(to), fwd) < std::cos(glm::radians(it.viewAngleDeg))) continue;
        if (dist < bestDist) {
            bestDist = dist;
            best = int(i);
        }
    }
    return best;
}

// Floor area (m^2) from which the player, facing object k, would get k's prompt.
float pickableArea(const Scene& s, const Grid& g, size_t k) {
    const glm::vec3 c = s.objects[k].worldCenter();
    int n = 0;
    for (int j = 0; j < g.nz; ++j)
        for (int i = 0; i < g.nx; ++i) {
            if (!g.reach[j * g.nx + i]) continue;
            const glm::vec2 p = g.at(i, j), to(c.x - p.x, c.z - p.y);
            if (glm::length(to) > 1e-3f && pickAt(s, p, glm::normalize(to)) == int(k)) ++n;
        }
    return n * g.step * g.step;
}

float nearestReach(const Grid& g, const glm::vec2& target) {
    float best = 1e9f;
    for (int j = 0; j < g.nz; ++j)
        for (int i = 0; i < g.nx; ++i)
            if (g.reach[j * g.nx + i]) best = std::min(best, glm::length(g.at(i, j) - target));
    return best;
}

// Fraction of the view height/width the box covers from the interaction camera.
glm::vec2 viewFill(const Box3& b, const Interaction& it, bool& clipped) {
    const glm::mat4 vp = glm::perspective(glm::radians(kFovY), kAspect, 0.05f, 50.0f) *
                         glm::lookAt(it.camPos, it.camTarget, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec2 lo(1e9f), hi(-1e9f);
    clipped = false;
    for (int i = 0; i < 8; ++i) {
        glm::vec3 c((i & 1) ? b.max.x : b.min.x, (i & 2) ? b.max.y : b.min.y, (i & 4) ? b.max.z : b.min.z);
        glm::vec4 h = vp * glm::vec4(c, 1.0f);
        if (h.w <= 0.0f) {
            clipped = true;
            continue;
        }
        glm::vec2 n(h.x / h.w, h.y / h.w);
        lo = glm::min(lo, n);
        hi = glm::max(hi, n);
    }
    if (lo.x < -1.0f || lo.y < -1.0f || hi.x > 1.0f || hi.y > 1.0f) clipped = true;
    return (glm::min(hi, glm::vec2(1.0f)) - glm::max(lo, glm::vec2(-1.0f))) * 0.5f;
}

void printMap(const Scene& s, const std::vector<Box3>& boxes, const Grid& g) {
    const float cell = 0.25f;
    const float hx = s.room.size.x / 2, hz = s.room.size.z / 2;
    const int cols = int(std::lround(s.room.size.x / cell)), rows = int(std::lround(s.room.size.z / cell));
    std::printf("\n  Top-down map (north = -Z at top, 1 char = %.2f m). Upper = solid, lower = non-solid,\n"
                "  ~ rug, @ spawn, * interaction camera, : free but unreachable, ' ' reachable floor.\n",
                cell);
    std::vector<std::string> lines(rows, std::string(cols, ' '));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            glm::vec2 p(-hx + (c + 0.5f) * cell, -hz + (r + 0.5f) * cell);
            char ch = ' ';
            int gi = int((p.x - g.x0) / g.step), gj = int((p.y - g.z0) / g.step);
            bool inGrid = p.x >= s.boundsMin.x && p.x <= s.boundsMax.x && p.y >= s.boundsMin.y && p.y <= s.boundsMax.y &&
                          gi >= 0 && gj >= 0 && gi < g.nx && gj < g.nz;
            if (!inGrid || !g.reach[gj * g.nx + gi]) ch = ':';
            int rank = 0;   // rug 1 < non-solid 2 < solid 3
            for (size_t k = 0; k < s.objects.size(); ++k) {
                const Box3& b = boxes[k];
                if (p.x < b.min.x || p.x > b.max.x || p.y < b.min.z || p.y > b.max.z) continue;
                const SceneObject& o = s.objects[k];
                int rk = isRug(o) ? 1 : (o.solid ? 3 : 2);
                if (rk < rank) continue;
                // a later object of the same rank (e.g. item on a table) wins only if it is non-solid
                rank = rk;
                ch = isRug(o) ? '~' : char((o.solid ? 'A' : 'a') + int(k));
            }
            lines[r][c] = ch;
        }
    auto mark = [&](const glm::vec3& p, char ch) {
        int c = int((p.x + hx) / cell), r = int((p.z + hz) / cell);
        if (c >= 0 && r >= 0 && c < cols && r < rows) lines[r][c] = ch;
    };
    for (const SceneObject& o : s.objects)
        if (o.interaction.enabled) mark(o.interaction.camPos, '*');
    mark(s.spawnPos, '@');
    std::printf("  +%s+\n", std::string(cols, '-').c_str());
    for (const std::string& l : lines) std::printf("  |%s|\n", l.c_str());
    std::printf("  +%s+\n  Legend:", std::string(cols, '-').c_str());
    for (size_t k = 0; k < s.objects.size(); ++k) {
        const SceneObject& o = s.objects[k];
        if (isRug(o)) continue;
        std::printf("%s %c=%s%s", k % 4 == 0 ? "\n   " : "", char((o.solid ? 'A' : 'a') + int(k)), o.name.c_str(),
                    o.interaction.enabled ? "(!)" : "");
    }
    std::printf("\n");
}

void checkScene(const Scene& s, const char* prefix, std::set<std::string>& usedIds) {
    std::printf("\n==== %s ====\n", s.name.c_str());
    const glm::vec3 half(s.room.size.x / 2, s.room.size.y, s.room.size.z / 2);
    const float eps = 0.011f;

    // --- model usage
    for (const SceneObject& o : s.objects) {
        if (!o.model) fail("%s has no model", o.name.c_str());
        if (o.name.rfind(prefix, 0) != 0) fail("%s is not a %s* model", o.name.c_str(), prefix);
        if (!usedIds.insert(o.name).second) fail("model %s used twice", o.name.c_str());
    }
    for (const auto& [id, spec] : kSpecs)
        if (id.rfind(prefix, 0) == 0 &&
            std::none_of(s.objects.begin(), s.objects.end(), [&](const SceneObject& o) { return o.name == id; }))
            fail("model %s not used", id.c_str());
    if (s.objects.size() < 8) fail("only %zu models", s.objects.size());

    std::vector<Box3> boxes, solids;
    std::vector<std::vector<Box3>> parts;
    for (const SceneObject& o : s.objects) {
        boxes.push_back(worldBox(o));
        parts.push_back(occupiedParts(o));
    }
    auto hitsParts = [&](size_t m, const Box3& b) {
        return std::any_of(parts[m].begin(), parts[m].end(), [&](const Box3& p) { return overlap3(b, p); });
    };

    // --- placement
    for (size_t k = 0; k < s.objects.size(); ++k) {
        const SceneObject& o = s.objects[k];
        const Box3& b = boxes[k];
        if (b.min.x < -half.x - eps || b.max.x > half.x + eps || b.min.z < -half.z - eps || b.max.z > half.z + eps ||
            b.min.y < -eps || b.max.y > half.y + eps)
            fail("%s pokes out of the room", o.name.c_str());
        const Origin org = originOf(o);
        const float yaw = glm::radians(o.rotationDeg.y);
        const glm::vec2 fwd(std::sin(yaw), std::cos(yaw));
        if (org == Origin::Wall || o.solid) {
            // Which wall is the object's back closest to? Its front must face away from it.
            const float gaps[4] = {b.min.x + half.x, half.x - b.max.x, b.min.z + half.z, half.z - b.max.z};
            const glm::vec2 inward[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            int w = int(std::min_element(gaps, gaps + 4) - gaps);
            if (org == Origin::Wall && gaps[w] > 0.02f) fail("[wall] %s is %.2f m off the wall", o.name.c_str(), gaps[w]);
            if (gaps[w] < 0.15f && glm::dot(fwd, inward[w]) < 0.7f)
                fail("%s is against a wall but faces it (or sideways)", o.name.c_str());
        }
        if (org == Origin::Hang && std::fabs(b.max.y - half.y) > eps) fail("%s does not hang from the ceiling", o.name.c_str());
        if (org == Origin::Bottom && !isRug(o) && o.position.y > 0.01f) {
            bool supported = false;
            for (size_t m = 0; m < s.objects.size(); ++m) {
                auto top = kSurfaceTop.find(s.objects[m].name);
                if (m == k || top == kSurfaceTop.end() || std::fabs(top->second - o.position.y) > 0.005f) continue;
                const Box3& sb = boxes[m];
                if (b.min.x >= sb.min.x - eps && b.max.x <= sb.max.x + eps && b.min.z >= sb.min.z - eps && b.max.z <= sb.max.z + eps)
                    supported = true;
            }
            if (!supported) fail("%s floats at y=%.2f with no surface under it", o.name.c_str(), o.position.y);
        }
        if (org == Origin::Bottom && !isRug(o) && o.solid && std::fabs(o.position.y) > 1e-4f)
            fail("solid %s not on the floor", o.name.c_str());
        if (o.solid) solids.push_back(b);
    }
    for (size_t a = 0; a < s.objects.size(); ++a)
        for (size_t c = a + 1; c < s.objects.size(); ++c)
            if (s.objects[a].solid && s.objects[c].solid && overlapXZ(boxes[a], boxes[c]))
                fail("solid footprints overlap: %s / %s", s.objects[a].name.c_str(), s.objects[c].name.c_str());
    // Non-solid, non-rug floor-standing items must not intersect anything except what they stand on.
    for (size_t a = 0; a < s.objects.size(); ++a)
        for (size_t c = 0; c < s.objects.size(); ++c) {
            if (a == c || isRug(s.objects[a]) || isRug(s.objects[c]) || s.objects[a].solid) continue;
            Box3 lifted = boxes[a];
            lifted.min.y += 0.01f;   // resting contact is fine
            if (hitsParts(c, lifted)) fail("%s intersects %s", s.objects[a].name.c_str(), s.objects[c].name.c_str());
        }

    // --- spawn + reachability
    const glm::vec2 sp(s.spawnPos.x, s.spawnPos.z);
    if (glm::any(glm::lessThan(sp, s.boundsMin)) || glm::any(glm::greaterThan(sp, s.boundsMax))) fail("spawn out of bounds");
    if (blockedAt(sp, solids, kPlayerRadius)) fail("spawn inside a solid");
    if (s.boundsMin.x < -half.x || s.boundsMax.x > half.x || s.boundsMin.y < -half.z || s.boundsMax.y > half.z)
        fail("walkable bounds exceed the room");
    const Grid g = flood(s, solids, kPlayerRadius), gWide = flood(s, solids, kComfortRadius);
    size_t freeN = 0, reachN = 0;
    for (size_t i = 0; i < g.freeCell.size(); ++i) {
        freeN += g.freeCell[i];
        reachN += g.reach[i];
    }
    std::printf("  walkable cells: %zu free, %zu reachable from spawn (%.1f m^2)\n", freeN, reachN,
                reachN * g.step * g.step);
    if (reachN < freeN) warn("%.2f m^2 of free floor is unreachable (pockets)", (freeN - reachN) * g.step * g.step);
    std::printf("  spawn (%.2f, %.2f) yaw %.0f\n", s.spawnPos.x, s.spawnPos.z, s.spawnYawDeg);

    // --- interactions
    int count = 0, quest = 0;
    std::set<unsigned> animSets;
    std::printf("  %-18s %-23s %5s %4s %5s %5s %5s %5s %5s  prompt\n", "object", "anims", "range", "ang", "reach",
                "wide", "pick", "fillH", "fillW");
    for (size_t k = 0; k < s.objects.size(); ++k) {
        const SceneObject& o = s.objects[k];
        const Interaction& it = o.interaction;
        if (!it.enabled) continue;
        ++(kQuestItems.count(o.name) ? quest : count);
        if (!animSets.insert(it.anims).second) fail("%s repeats another object's animation set", o.name.c_str());
        if (it.range < 1.5f || it.range > 3.0f) fail("%s range %.2f outside 1.5..3.0", o.name.c_str(), it.range);
        if (it.viewAngleDeg < 30.0f || it.viewAngleDeg > 50.0f) fail("%s angle outside 30..50", o.name.c_str());
        if (it.prompt.empty() || it.prompt.size() > 26) fail("%s prompt empty/too long", o.name.c_str());
        if (!kSounds.count(it.sound)) fail("%s unknown sound '%s'", o.name.c_str(), it.sound.c_str());
        if (it.anims & (ANIM_ROTATE | ANIM_SWING)) {
            const glm::vec3 a = glm::abs(it.rotAxis);
            if (std::fabs(a.x + a.y + a.z - 1.0f) > 1e-5f || std::max({a.x, a.y, a.z}) < 0.999f)
                fail("%s rotAxis is not a unit X/Y/Z axis", o.name.c_str());
        }
        if (it.linkedPointLight >= int(s.lights.points.size())) fail("%s linked light out of range", o.name.c_str());
        if (it.linkedPointLight >= 0) {
            const PointLight& pl = s.lights.points[it.linkedPointLight];
            if (distToBox(pl.position, boxes[k]) > 0.05f) fail("%s linked light is not at the object", o.name.c_str());
            if (it.lightBoost < 1.5f) warn("%s light boost only %.1f", o.name.c_str(), it.lightBoost);
        }

        const glm::vec2 objXZ(o.position.x, o.position.z);
        const float dReach = nearestReach(g, objXZ), dWide = nearestReach(gWide, objXZ);
        if (dReach > it.range - 0.3f) fail("%s: nearest standing spot %.2f m vs range %.2f", o.name.c_str(), dReach, it.range);
        if (dWide > it.range - 0.3f) warn("%s hard to reach with a wider body (%.2f m)", o.name.c_str(), dWide);
        const float pick = pickableArea(s, g, k);
        if (pick < 0.25f) fail("%s is selectable from only %.2f m^2 of floor", o.name.c_str(), pick);

        // Camera: inside the room, not inside furniture, clear line of sight, object fills the view.
        const glm::vec3 c = it.camPos;
        if (std::fabs(c.x) > half.x - 0.1f || std::fabs(c.z) > half.z - 0.1f || c.y < 0.1f || c.y > half.y - 0.1f)
            fail("%s camPos outside the room", o.name.c_str());
        const Box3 animBox = animatedBox(o);
        if (distToBox(it.camTarget, animBox) > 0.01f) fail("%s camTarget not on the object", o.name.c_str());
        for (size_t m = 0; m < s.objects.size(); ++m) {
            if (m == k || isRug(s.objects[m])) continue;
            for (const Box3& p : parts[m]) {
                if (inside3(c, p)) fail("%s camPos inside %s", o.name.c_str(), s.objects[m].name.c_str());
                else if (segmentHitsBox(c, it.camTarget, p))
                    fail("%s camera view blocked by %s", o.name.c_str(), s.objects[m].name.c_str());
            }
        }
        // Fill is measured on the held end pose; an object that rises/slides may enter the frame mid-animation.
        bool clipped = false, pathClipped = false;
        const glm::vec2 fill = viewFill(finalBox(o), it, clipped);
        viewFill(animBox, it, pathClipped);
        if (clipped) warn("%s does not fit the interaction view", o.name.c_str());
        if (std::max(fill.x, fill.y) < 0.45f) fail("%s too small in its interaction view (%.2f)", o.name.c_str(), std::max(fill.x, fill.y));

        // Animated extent must stay in the room and clear of other furniture.
        if (animBox.min.x < -half.x - eps || animBox.max.x > half.x + eps || animBox.min.z < -half.z - eps ||
            animBox.max.z > half.z + eps || animBox.max.y > half.y + eps)
            fail("%s animates out of the room", o.name.c_str());
        if (it.anims & ANIM_TRANSLATE)
            for (size_t m = 0; m < s.objects.size(); ++m) {
                if (m == k || isRug(s.objects[m])) continue;
                for (float t : {0.25f, 0.5f, 0.75f, 1.0f}) {   // the path is a straight line whatever the easing
                    SceneObject moved = o;
                    moved.animTranslate = it.translateBy * t;
                    Box3 mb = worldBox(moved);
                    mb.min.y += 0.01f;
                    if (hitsParts(m, mb)) {
                        fail("%s slides/floats into %s", o.name.c_str(), s.objects[m].name.c_str());
                        break;
                    }
                }
            }

        std::printf("  %-18s %-23s %5.2f %4.0f %5.2f %5.2f %5.1f %5.2f %5.2f  \"%s\" (%s)%s\n", o.name.c_str(),
                    animNames(it.anims).c_str(), it.range, it.viewAngleDeg, dReach, dWide, pick, fill.y, fill.x,
                    it.prompt.c_str(), it.sound.c_str(), pathClipped && !clipped ? " [enters frame]" : "");
    }
    if (count != 4) fail("%d regular interactive objects (need exactly 4)", count);
    if (quest != 1) fail("%d quest items (need exactly 1)", quest);

    // --- lights
    if (s.lights.points.size() > size_t(MAX_POINT_LIGHTS)) fail("too many point lights");
    for (size_t i = 0; i < s.lights.points.size(); ++i) {
        const PointLight& p = s.lights.points[i];
        std::string near;
        float best = 1e9f;
        for (size_t k = 0; k < s.objects.size(); ++k) {
            float d = distToBox(p.position, boxes[k]);
            if (d < best) {
                best = d;
                near = s.objects[k].name;
            }
        }
        std::printf("  light %zu at (%.2f, %.2f, %.2f) I=%.2f range %.1f flicker %.2f -> %s (%.2f m)\n", i, p.position.x,
                    p.position.y, p.position.z, p.intensity, p.range, p.flicker, near.c_str(), best);
        if (std::fabs(p.position.x) > half.x || std::fabs(p.position.z) > half.z || p.position.y < 0 || p.position.y > half.y)
            fail("light %zu outside the room", i);
        if (best > 0.35f) warn("light %zu is not at any model", i);
    }

    printMap(s, boxes, g);
}

} // namespace

int main() {
    ModelCache cache;
    std::set<std::string> used;
    checkScene(buildLibrary(cache), "lib_", used);
    checkScene(buildBedroom(cache), "bed_", used);
    std::printf("\n%d failure(s), %d warning(s)\n", gFailures, gWarnings);
    return gFailures ? 1 : 0;
}
