#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

// Immediate-mode 2D overlay. All coordinates are framebuffer pixels, origin top-left.
// Usage per frame: begin(w,h); rect/text/button...; end();
class Hud {
public:
    bool init();                        // compiles its own embedded shader, creates buffers
    void begin(int fbWidth, int fbHeight);
    void end();                         // flushes; saves/restores depth-test and blend state

    void rect(float x, float y, float w, float h, const glm::vec4& color);
    // Bitmap text (stb_easy_font). scale 1 = ~7px tall glyphs; 2-3 is readable. Returns drawn width.
    float text(float x, float y, const std::string& s, float scale, const glm::vec4& color);
    float textWidth(const std::string& s, float scale) const;

    // Draws a labelled button. 'active' highlights it, hover highlight when the mouse is inside.
    // Returns true when 'clicked' is true and (mouseX, mouseY) is inside the button.
    bool button(float x, float y, float w, float h, const std::string& label, bool active,
                double mouseX, double mouseY, bool clicked);

private:
    struct V { float x, y, r, g, b, a; };
    std::vector<V> verts_;
    GLuint vao_ = 0, vbo_ = 0, prog_ = 0;
    GLint uScreen_ = -1;
    int w_ = 1, h_ = 1;
};
