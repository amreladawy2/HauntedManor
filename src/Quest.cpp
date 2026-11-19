#include "Quest.h"

#include "Hud.h"
#include "Model.h"
#include "Scene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

const char* const Quest::kClueObjects[kClueCount] = {"lib_globe", "lib_bookshelf", "lib_candelabra", "lib_clock"};

const char* const Quest::kRiddle[5] = {
    "I have hands, but nothing I hold.",
    "I have a face that never grows old.",
    "I stopped the night the child went still;",
    "downstairs I wait, and always will.",
    "What am I?",
};

const char* const Quest::kChoices[kChoiceCount] = {"The portrait", "The grandfather clock", "The music box",
                                                   "The rocking chair"};

// =============================================================================================
// Quest state
// =============================================================================================

void Quest::update(float dt) {
    if (stage != QuestStage::Won) playTime += dt;
    if (stage == QuestStage::TakeMap) revealTime += dt;
}

bool Quest::recordClue(const std::string& objectName) {
    if (stage != QuestStage::Investigate) return false;
    for (int i = 0; i < kClueCount; ++i)
        if (objectName == kClueObjects[i]) clues_[size_t(i)] = true;
    if (cluesFound() < kClueCount) return false;
    stage = QuestStage::TakeMap;
    revealTime = 0.0f;
    return true;
}

int Quest::cluesFound() const { return int(std::count(clues_.begin(), clues_.end(), true)); }

std::string Quest::objective() const {
    switch (stage) {
    case QuestStage::Investigate:
        return "Investigate the Library (" + std::to_string(cluesFound()) + "/" + std::to_string(kClueCount) + ")";
    case QuestStage::TakeMap: return "Take the map from the desk";
    case QuestStage::SearchKey: return "Read the map [M] and search the Nursery";
    case QuestStage::Riddle: return triedWrong ? "Try the key again..." : "Solve the riddle";
    case QuestStage::Won: return "The manor is quiet... for now.";
    }
    return {};
}

void Quest::shuffleChoices(std::mt19937& rng) {
    const auto previous = order;
    do {
        std::shuffle(order.begin(), order.end(), rng);
    } while (order == previous);
}

// =============================================================================================
// HUD panels
// =============================================================================================

namespace {

const glm::vec4 kInk(0.22f, 0.13f, 0.07f, 1.0f);
const glm::vec4 kInkSoft(0.22f, 0.13f, 0.07f, 0.6f);
const glm::vec4 kBlood(0.70f, 0.04f, 0.03f, 1.0f);

float textScale(float base, float ui) { return std::max(1.0f, std::floor(base * ui + 0.25f)); }

float smooth01(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

glm::vec4 fadeA(glm::vec4 c, float a) {
    c.a *= a;
    return c;
}

void frame(Hud& hud, float x, float y, float w, float h, float t, const glm::vec4& c) {
    hud.rect(x, y, w, t, c);
    hud.rect(x, y + h - t, w, t, c);
    hud.rect(x, y + t, t, h - 2.0f * t, c);
    hud.rect(x + w - t, y + t, t, h - 2.0f * t, c);
}

// Aged parchment sheet: warm paper, scorched edges darkening toward the rim, a double ink rule.
void parchment(Hud& hud, float x, float y, float w, float h, float ui, float a) {
    hud.rect(x, y, w, h, {0.85f, 0.77f, 0.58f, 0.97f * a});
    const float band = std::max(2.0f, std::round(5.0f * ui));
    for (int i = 0; i < 7; ++i) {
        const float inset = float(i) * band;
        frame(hud, x + inset, y + inset, w - 2.0f * inset, h - 2.0f * inset, band,
              {0.42f, 0.27f, 0.10f, (0.26f - 0.035f * float(i)) * a});
    }
    const float m = std::round(12.0f * ui), t = std::max(1.0f, std::round(2.0f * ui));
    frame(hud, x + m, y + m, w - 2.0f * m, h - 2.0f * m, t, fadeA(kInk, 0.75f * a));
    frame(hud, x + m + 4.0f * t, y + m + 4.0f * t, w - 2.0f * (m + 4.0f * t), h - 2.0f * (m + 4.0f * t), 1.0f,
          fadeA(kInk, 0.35f * a));
}

void centeredText(Hud& hud, float cx, float y, const std::string& s, float scale, const glm::vec4& c) {
    hud.text(cx - (hud.textWidth(s, scale) - scale) * 0.5f, y, s, scale, c);
}

// "bed_rocking_chair" -> "rocking chair"
std::string shortLabel(const std::string& name) {
    std::string s = name;
    const size_t us = s.find('_');
    if (us != std::string::npos && us <= 4) s = s.substr(us + 1);
    std::replace(s.begin(), s.end(), '_', ' ');
    return s;
}

}  // namespace

void drawNurseryMap(Hud& hud, const Scene& sc, bool keyStillThere, bool showPlayer, const glm::vec3& playerPos,
                    float playerYawDeg, float time, float ui, const glm::vec4& region) {
    const float ts = textScale(2.0f, ui), tsTitle = textScale(3.0f, ui);
    const float roomW = sc.room.size.x, roomD = sc.room.size.z;

    // Layout: the plan keeps the room's proportions and is as large as the free region allows.
    const float header = std::round(64.0f * ui), footer = std::round(84.0f * ui), side = std::round(76.0f * ui);
    const float k = std::max(1.0f, std::min((region.w - header - footer) / roomD, (region.z - 2.0f * side) / roomW));
    const float mapW = roomW * k, mapH = roomD * k;
    const float panelW = mapW + 2.0f * side, panelH = mapH + header + footer;
    const float px = std::round(region.x + (region.z - panelW) * 0.5f);
    const float py = std::round(region.y + (region.w - panelH) * 0.5f);
    const float x0 = px + side, y0 = py + header;
    auto toMap = [&](float x, float z) { return glm::vec2(x0 + (x + roomW * 0.5f) * k, y0 + (z + roomD * 0.5f) * k); };

    parchment(hud, px, py, panelW, panelH, ui, 1.0f);
    centeredText(hud, px + panelW * 0.5f, py + std::round(24.0f * ui), "MAP OF THE NURSERY", tsTitle, kInk);

    // Floor wash and a faint 1 m survey grid.
    hud.rect(x0, y0, mapW, mapH, {0.76f, 0.66f, 0.47f, 0.55f});
    for (float gx = 1.0f; gx < roomW - 0.01f; gx += 1.0f) hud.rect(x0 + gx * k, y0, 1.0f, mapH, fadeA(kInk, 0.08f));
    for (float gz = 1.0f; gz < roomD - 0.01f; gz += 1.0f) hud.rect(x0, y0 + gz * k, mapW, 1.0f, fadeA(kInk, 0.08f));

    // Footprints of every object (transformed model AABB), largest first so small things stay on top.
    struct Item {
        std::string label;
        glm::vec2 lo, hi;
        bool window;
    };
    std::vector<Item> items;
    glm::vec2 keyAt(0.0f);
    bool haveKey = false;
    for (const SceneObject& o : sc.objects) {
        if (!o.model) continue;
        if (o.name == Quest::kKeyObject) {
            const glm::vec3 c = o.worldCenter();
            keyAt = {c.x, c.z};
            haveKey = true;
            continue;
        }
        const glm::mat4 m = o.modelMatrix();
        glm::vec2 lo(FLT_MAX), hi(-FLT_MAX);
        for (int i = 0; i < 8; ++i) {
            const glm::vec3 c((i & 1) ? o.model->boundsMax.x : o.model->boundsMin.x,
                              (i & 2) ? o.model->boundsMax.y : o.model->boundsMin.y,
                              (i & 4) ? o.model->boundsMax.z : o.model->boundsMin.z);
            const glm::vec4 w = m * glm::vec4(c, 1.0f);
            lo = glm::min(lo, glm::vec2(w.x, w.z));
            hi = glm::max(hi, glm::vec2(w.x, w.z));
        }
        items.push_back({shortLabel(o.name), lo, hi, o.name.find("window") != std::string::npos});
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return (a.hi.x - a.lo.x) * (a.hi.y - a.lo.y) > (b.hi.x - b.lo.x) * (b.hi.y - b.lo.y);
    });
    for (const Item& it : items) {
        const glm::vec2 a = toMap(it.lo.x, it.lo.y), b = toMap(it.hi.x, it.hi.y);
        hud.rect(a.x, a.y, b.x - a.x, b.y - a.y,
                 it.window ? glm::vec4(0.30f, 0.45f, 0.72f, 0.65f) : glm::vec4(0.45f, 0.29f, 0.13f, 0.22f));
        frame(hud, a.x, a.y, b.x - a.x, b.y - a.y, 1.0f, kInkSoft);
    }

    // Walls, with the open doorway in the middle of the +Z (south) wall.
    const float wt = std::max(2.0f, std::round(3.0f * ui));
    const float cx = x0 + mapW * 0.5f, doorHalf = 0.6f * k;
    hud.rect(x0 - wt, y0 - wt, mapW + 2.0f * wt, wt, kInk);
    hud.rect(x0 - wt, y0, wt, mapH, kInk);
    hud.rect(x0 + mapW, y0, wt, mapH, kInk);
    hud.rect(x0 - wt, y0 + mapH, cx - doorHalf - (x0 - wt), wt, kInk);
    hud.rect(cx + doorHalf, y0 + mapH, x0 + mapW + wt - (cx + doorHalf), wt, kInk);
    hud.rect(cx - doorHalf - 1.0f, y0 + mapH, 2.0f, wt + 8.0f * ui, kInk);
    hud.rect(cx + doorHalf - 1.0f, y0 + mapH, 2.0f, wt + 8.0f * ui, kInk);
    centeredText(hud, cx, y0 + mapH + wt + 10.0f * ui, "door", ts, kInk);

    // Labels, nudged up/down so neighbours never print over each other.
    std::vector<glm::vec4> placed;
    const float lh = 7.0f * ts, gap = 3.0f * ui;
    for (const Item& it : items) {
        const glm::vec2 c = toMap((it.lo.x + it.hi.x) * 0.5f, (it.lo.y + it.hi.y) * 0.5f);
        const float lw = hud.textWidth(it.label, ts);
        const float lx = std::clamp(c.x - lw * 0.5f, px + 20.0f * ui, px + panelW - 20.0f * ui - lw);
        for (int attempt = 0; attempt < 7; ++attempt) {
            const float off = float((attempt + 1) / 2) * ((attempt % 2) ? 1.0f : -1.0f) * (lh + gap);
            const glm::vec4 r(lx, c.y - lh * 0.5f + off, lw, lh);
            bool clash = false;
            for (const glm::vec4& p : placed)
                clash |= r.x < p.x + p.z + gap && p.x < r.x + r.z + gap && r.y < p.y + p.w + gap && p.y < r.y + r.w + gap;
            if (!clash || attempt == 6) {
                placed.push_back(r);
                hud.text(r.x, r.y, it.label, ts, it.window ? glm::vec4(0.10f, 0.22f, 0.45f, 1.0f) : kInk);
                break;
            }
        }
    }

    // Pulsing red X where the key lies.
    if (keyStillThere && haveKey) {
        const glm::vec2 p = toMap(keyAt.x, keyAt.y);
        const float ring = std::fmod(time, 1.3f) / 1.3f;
        const float r = (8.0f + 26.0f * ring) * ui;
        frame(hud, p.x - r, p.y - r, 2.0f * r, 2.0f * r, std::max(1.0f, std::round(2.0f * ui)),
              fadeA(kBlood, 0.85f * (1.0f - ring)));
        const float xs = textScale(5.0f, ui);
        const float pulse = 0.5f + 0.5f * std::sin(time * 5.0f);
        centeredText(hud, p.x, p.y - 3.5f * xs, "X", xs, fadeA(kBlood, 0.6f + 0.4f * pulse));
    }

    // The player: a marker with a dotted facing line.
    if (showPlayer) {
        const glm::vec2 p = toMap(playerPos.x, playerPos.z);
        const float yaw = glm::radians(playerYawDeg);
        const glm::vec2 fwd(-std::sin(yaw), -std::cos(yaw));
        const glm::vec4 blue(0.08f, 0.22f, 0.55f, 1.0f);
        const float dot = std::max(2.0f, std::round(4.0f * ui));
        for (int i = 1; i <= 6; ++i) {
            const glm::vec2 q = p + fwd * (float(i) * 0.14f * k);
            hud.rect(q.x - dot * 0.5f, q.y - dot * 0.5f, dot, dot, fadeA(blue, 1.0f - 0.12f * float(i)));
        }
        const float d = std::round(12.0f * ui);
        hud.rect(p.x - d * 0.5f, p.y - d * 0.5f, d, d, blue);
        frame(hud, p.x - d * 0.5f - 2.0f, p.y - d * 0.5f - 2.0f, d + 4.0f, d + 4.0f, 1.0f, kInk);
        hud.text(p.x + d, p.y - d - lh * 0.5f, "you", ts, blue);
    }

    // Compass rose (north = -Z, top of the plan).
    {
        const float ccx = px + panelW - side * 0.55f, top = y0 + 6.0f * ui, len = 46.0f * ui;
        const float t = std::max(2.0f, std::round(2.0f * ui));
        centeredText(hud, ccx, top, "N", ts, kInk);
        hud.rect(ccx - t * 0.5f, top + lh + 6.0f * ui, t, len, kInk);
        for (int i = 0; i < 4; ++i) {
            const float w = float(2 * i + 1) * t;
            hud.rect(ccx - w * 0.5f, top + lh + 6.0f * ui + float(i) * t, w, t, kInk);
        }
    }

    centeredText(hud, px + panelW * 0.5f, py + panelH - std::round(30.0f * ui) - lh,
                 "X marks what the child hid.   [M] close map", ts, kInkSoft);
}

int drawRiddle(Hud& hud, const Quest& q, float openTime, float ui, int fbW, int fbH, double mx, double my,
               bool clicked) {
    const float W = float(fbW), H = float(fbH);
    const float a = smooth01(openTime / 0.35f);
    const float ts = textScale(3.0f, ui), tsSmall = textScale(2.0f, ui);
    hud.rect(0, 0, W, H, {0.0f, 0.0f, 0.0f, 0.6f * a});

    float poemW = 0.0f;
    for (const char* line : Quest::kRiddle) poemW = std::max(poemW, hud.textWidth(line, ts));
    const float lineH = 12.0f * ts, bh = std::round(40.0f * ui), gap = std::round(12.0f * ui);
    const float margin = std::round(46.0f * ui);
    const float panelW = std::round(std::max(poemW + 2.0f * margin + 40.0f * ui, 760.0f * ui));
    const float panelH = std::round(margin + 7.0f * ts + 28.0f * ui + 5.0f * lineH + 22.0f * ui + 2.0f * bh + gap +
                                    26.0f * ui + 7.0f * tsSmall + margin);
    const float px = std::round((W - panelW) * 0.5f);
    const float py = std::round((H - panelH) * 0.5f + (1.0f - a) * 24.0f * ui);
    const float cx = px + panelW * 0.5f;
    parchment(hud, px, py, panelW, panelH, ui, 1.0f);

    float y = py + margin;
    centeredText(hud, cx, y, "WORDS ARE SCRATCHED INTO THE KEY", ts, {0.45f, 0.06f, 0.04f, 1.0f});
    y += 7.0f * ts + 28.0f * ui;
    for (int i = 0; i < 5; ++i) {
        centeredText(hud, cx, y, Quest::kRiddle[i], ts, i == 4 ? kBlood : kInk);
        y += lineH;
    }
    hud.rect(cx - panelW * 0.3f, y + 4.0f * ui, panelW * 0.6f, 1.0f, kInkSoft);
    y += 22.0f * ui;

    int picked = -1;
    const float bw = std::round((panelW - 2.0f * margin - gap) * 0.5f);
    for (int slot = 0; slot < Quest::kChoiceCount; ++slot) {
        const float bx = px + margin + float(slot % 2) * (bw + gap);
        const float by = y + float(slot / 2) * (bh + gap);
        char label[64];
        std::snprintf(label, sizeof label, "%d   %s", slot + 1, Quest::kChoices[q.order[size_t(slot)]]);
        if (hud.button(bx, by, bw, bh, label, false, mx, my, clicked)) picked = slot;
    }
    y += 2.0f * bh + gap + 26.0f * ui;
    centeredText(hud, cx, y, "Press 1-4 or click an answer.     Esc: put the key down", tsSmall, kInkSoft);
    return picked;
}

void drawWinScreen(Hud& hud, const Quest& q, float shown, float ui, int fbW, int fbH) {
    const float W = float(fbW), H = float(fbH);
    const float a = smooth01(shown / 1.2f);
    hud.rect(0, 0, W, H, {0.0f, 0.0f, 0.0f, 0.82f * a});
    for (int i = 0; i < 6; ++i) {  // soft dawn glow behind the title
        const float h = H * (0.08f + 0.06f * float(i));
        hud.rect(0, H * 0.42f - h * 0.5f, W, h, {0.45f, 0.28f, 0.08f, 0.05f * a});
    }

    const float tsTitle = textScale(7.0f, ui), ts = textScale(3.0f, ui), tsSmall = textScale(2.0f, ui);
    const glm::vec4 gold(0.98f, 0.82f, 0.45f, a), parchmentC(0.93f, 0.87f, 0.75f, a), dim(0.78f, 0.72f, 0.62f, 0.9f * a);
    const float cx = W * 0.5f;
    float y = H * 0.33f;
    centeredText(hud, cx, y, "YOU ESCAPED THE MANOR", tsTitle, gold);
    y += 7.0f * tsTitle + 22.0f * ui;
    const float ruleW = hud.textWidth("YOU ESCAPED THE MANOR", tsTitle) * 0.7f;
    hud.rect(cx - ruleW * 0.5f, y, ruleW, std::max(1.0f, std::round(2.0f * ui)), {0.96f, 0.63f, 0.22f, 0.8f * a});
    y += 30.0f * ui;

    const int secs = int(q.playTime);
    char line[64];
    std::snprintf(line, sizeof line, "Time taken   %02d:%02d", secs / 60, secs % 60);
    centeredText(hud, cx, y, line, ts, parchmentC);
    y += 7.0f * ts + 22.0f * ui;
    centeredText(hud, cx, y, "The grandfather clock ticks once more. Somewhere, a child laughs.", tsSmall, dim);
    y += 7.0f * tsSmall + 46.0f * ui;
    if (shown > 1.0f) {
        const float blink = 0.6f + 0.4f * std::sin(shown * 3.0f);
        centeredText(hud, cx, y, "Press ENTER to keep exploring", ts, {0.96f, 0.63f, 0.22f, blink * a});
    }
}
