#include "Hud.h"

#include <stb/stb_easy_font.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

const char* kVertSrc = R"(#version 460 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
uniform vec2 uScreen;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = vec4(aPos.x / uScreen.x * 2.0 - 1.0, 1.0 - aPos.y / uScreen.y * 2.0, 0.0, 1.0);
}
)";

const char* kFragSrc = R"(#version 460 core
in vec4 vColor;
out vec4 FragColor;
void main() { FragColor = vColor; }
)";

GLuint compileStage(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "[Hud] shader compile failed: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

// Vertex layout written by stb_easy_font_print: x, y, z floats + 4 colour bytes (16 bytes).
struct EasyVert { float x, y, z; unsigned char rgba[4]; };

// stb_easy_font indexes a table with (c - 32): anything outside printable ASCII would read out of bounds.
std::string sanitize(const std::string& s) {
    std::string out = s;
    for (char& c : out)
        if (c != '\n' && (static_cast<unsigned char>(c) < 32 || static_cast<unsigned char>(c) > 126)) c = '?';
    return out;
}

void frame(Hud& hud, float x, float y, float w, float h, float t, const glm::vec4& c) {
    hud.rect(x, y, w, t, c);
    hud.rect(x, y + h - t, w, t, c);
    hud.rect(x, y + t, t, h - 2 * t, c);
    hud.rect(x + w - t, y + t, t, h - 2 * t, c);
}

} // namespace

bool Hud::init() {
    GLuint vs = compileStage(GL_VERTEX_SHADER, kVertSrc);
    GLuint fs = compileStage(GL_FRAGMENT_SHADER, kFragSrc);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return false;
    }
    prog_ = glCreateProgram();
    glAttachShader(prog_, vs);
    glAttachShader(prog_, fs);
    glLinkProgram(prog_);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(prog_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog_, sizeof log, nullptr, log);
        std::fprintf(stderr, "[Hud] shader link failed: %s\n", log);
        glDeleteProgram(prog_);
        prog_ = 0;
        return false;
    }
    uScreen_ = glGetUniformLocation(prog_, "uScreen");

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(V), reinterpret_cast<void*>(offsetof(V, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(V), reinterpret_cast<void*>(offsetof(V, r)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return true;
}

void Hud::begin(int fbWidth, int fbHeight) {
    w_ = std::max(1, fbWidth);
    h_ = std::max(1, fbHeight);
    verts_.clear();
}

void Hud::rect(float x, float y, float w, float h, const glm::vec4& c) {
    if (w <= 0.0f || h <= 0.0f) return;
    const float x1 = x + w, y1 = y + h;
    const V v[6] = {
        {x, y, c.r, c.g, c.b, c.a},  {x1, y, c.r, c.g, c.b, c.a}, {x1, y1, c.r, c.g, c.b, c.a},
        {x, y, c.r, c.g, c.b, c.a},  {x1, y1, c.r, c.g, c.b, c.a}, {x, y1, c.r, c.g, c.b, c.a},
    };
    verts_.insert(verts_.end(), v, v + 6);
}

float Hud::text(float x, float y, const std::string& s, float scale, const glm::vec4& color) {
    if (s.empty()) return 0.0f;
    std::string str = sanitize(s);
    std::vector<EasyVert> buf(str.size() * 24 + 16);  // stb averages ~17 verts per glyph
    const int quads = stb_easy_font_print(0.0f, 0.0f, str.data(), nullptr, buf.data(),
                                          static_cast<int>(buf.size() * sizeof(EasyVert)));
    // Snap to whole pixels so the 1-px font strokes stay crisp.
    x = std::floor(x + 0.5f);
    y = std::floor(y + 0.5f);

    auto emit = [&](float ox, float oy, const glm::vec4& c) {
        for (int q = 0; q < quads; ++q) {
            const EasyVert* e = &buf[q * 4];
            V v[4];
            for (int k = 0; k < 4; ++k) v[k] = {ox + e[k].x * scale, oy + e[k].y * scale, c.r, c.g, c.b, c.a};
            verts_.insert(verts_.end(), {v[0], v[1], v[2], v[0], v[2], v[3]});
        }
    };
    // Drop shadow keeps text readable over bright flames and lamps.
    const float sh = std::max(1.0f, std::floor(scale * 0.5f + 0.5f));
    emit(x + sh, y + sh, {0.0f, 0.0f, 0.0f, color.a * 0.75f});
    emit(x, y, color);
    return textWidth(str, scale);
}

float Hud::textWidth(const std::string& s, float scale) const {
    if (s.empty()) return 0.0f;
    std::string str = sanitize(s);
    return static_cast<float>(stb_easy_font_width(str.data())) * scale;
}

bool Hud::button(float x, float y, float w, float h, const std::string& label, bool active,
                 double mouseX, double mouseY, bool clicked) {
    const bool hover = mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + h;

    // Palette: soot-black panel, tarnished-brass rules; active = blood red with amber trim.
    glm::vec4 fill = active ? glm::vec4(0.30f, 0.025f, 0.03f, 0.84f) : glm::vec4(0.045f, 0.03f, 0.028f, 0.74f);
    glm::vec4 border = active ? glm::vec4(0.96f, 0.63f, 0.22f, 1.0f) : glm::vec4(0.52f, 0.42f, 0.28f, 0.85f);
    glm::vec4 label_c = active ? glm::vec4(1.0f, 0.86f, 0.62f, 1.0f) : glm::vec4(0.80f, 0.74f, 0.63f, 1.0f);
    if (hover) {
        fill += glm::vec4(0.07f, 0.04f, 0.03f, 0.08f);
        border = glm::min(border * 1.3f, glm::vec4(1.0f));
        label_c = glm::min(label_c * 1.2f, glm::vec4(1.0f));
    }

    if (hover || active) {  // faint candle-glow halo
        glm::vec4 glow(border.r, border.g, border.b, hover ? 0.22f : 0.14f);
        frame(*this, x - 3, y - 3, w + 6, h + 6, 3, glow);
    }
    rect(x, y, w, h, fill);
    rect(x, y + h * 0.55f, w, h * 0.45f, {0.0f, 0.0f, 0.0f, 0.22f});  // shade lower half for depth
    frame(*this, x, y, w, h, 1, border);
    glm::vec4 inner(border.r, border.g, border.b, border.a * 0.45f);  // Victorian double rule
    frame(*this, x + 3, y + 3, w - 6, h - 6, 1, inner);
    for (float cx : {x + 2.0f, x + w - 5.0f})  // corner studs
        for (float cy : {y + 2.0f, y + h - 5.0f}) rect(cx, cy, 3, 3, border);
    if (active) rect(x + 8, y + h - 6, w - 16, 1, {0.75f, 0.06f, 0.05f, 0.9f});  // blood-red underline

    float scale = std::clamp(std::floor(h / 18.0f), 1.0f, 4.0f);
    const float room = w - 12.0f;
    while (scale > 1.0f && textWidth(label, scale) > room) scale -= 1.0f;
    std::string shown = label;
    if (textWidth(shown, scale) > room) {  // still too wide at the smallest scale: truncate with ".."
        while (!shown.empty() && textWidth(shown + "..", scale) > room) shown.pop_back();
        shown += "..";
    }
    const float tw = textWidth(shown, scale);
    // stb_easy_font capitals are 7 units tall; trailing advance adds ~1 unit to the measured width.
    text(x + (w - tw + scale) * 0.5f, y + (h - 7.0f * scale) * 0.5f, shown, scale, label_c);

    return clicked && hover;
}

void Hud::end() {
    if (verts_.empty() || !prog_) {
        verts_.clear();
        return;
    }
    const GLboolean depth = glIsEnabled(GL_DEPTH_TEST);
    const GLboolean blend = glIsEnabled(GL_BLEND);
    const GLboolean cull = glIsEnabled(GL_CULL_FACE);
    GLint srcRGB, dstRGB, srcA, dstA, prevProg, prevVao, prevBuf, vp[4];
    glGetIntegerv(GL_BLEND_SRC_RGB, &srcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB, &dstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcA);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &dstA);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);
    glGetIntegerv(GL_VIEWPORT, vp);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glViewport(0, 0, w_, h_);

    glUseProgram(prog_);
    glUniform2f(uScreen_, static_cast<float>(w_), static_cast<float>(h_));
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts_.size() * sizeof(V)), verts_.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts_.size()));
    verts_.clear();

    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(prevBuf));
    glBindVertexArray(static_cast<GLuint>(prevVao));
    glUseProgram(static_cast<GLuint>(prevProg));
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glBlendFuncSeparate(srcRGB, dstRGB, srcA, dstA);
    if (depth) glEnable(GL_DEPTH_TEST);
    if (!blend) glDisable(GL_BLEND);
    if (cull) glEnable(GL_CULL_FACE);
}
