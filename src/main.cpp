#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdio>
#include <string>

#include "Audio.h"
#include "Game.h"

namespace {

void glfwErrorCallback(int code, const char* message) {
    std::fprintf(stderr, "[GLFW] error %d: %s\n", code, message);
}

// Tries OpenGL 4.6 core first, then 4.3 and 3.3 (the shaders need 4.6, so older contexts are only logged).
GLFWwindow* createWindow() {
    const int versions[][2] = {{4, 6}, {4, 3}, {3, 3}};
    for (const auto& v : versions) {
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, v[0]);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, v[1]);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        if (GLFWwindow* w = glfwCreateWindow(1600, 900, "Haunted Manor", nullptr, nullptr)) {
            if (v[0] != 4 || v[1] != 6)
                std::fprintf(stderr, "[main] warning: running on an OpenGL %d.%d context; "
                                     "the #version 460 shaders may fail to compile\n", v[0], v[1]);
            return w;
        }
        std::fprintf(stderr, "[main] OpenGL %d.%d core context unavailable\n", v[0], v[1]);
    }
    return nullptr;
}

Game* gameOf(GLFWwindow* w) { return static_cast<Game*>(glfwGetWindowUserPointer(w)); }

}  // namespace

int main(int argc, char** argv) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        std::fprintf(stderr, "[main] glfwInit failed\n");
        return 1;
    }

    GLFWwindow* window = createWindow();
    if (!window) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync

    glewExperimental = GL_TRUE;
    if (const GLenum err = glewInit(); err != GLEW_OK) {
        std::fprintf(stderr, "[main] glewInit failed: %s\n", reinterpret_cast<const char*>(glewGetErrorString(err)));
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    glGetError();  // glewInit can leave GL_INVALID_ENUM behind on core profiles
    std::printf("OpenGL %s | %s\n", reinterpret_cast<const char*>(glGetString(GL_VERSION)),
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);  // the room shell faces inward, so outside walls vanish in the cut-away views
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glEnable(GL_MULTISAMPLE);

    Audio::init();
    Game game;
    if (!game.init(window)) {
        Audio::shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glfwSetWindowUserPointer(window, &game);
    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int mods) {
        gameOf(w)->onKey(key, action, mods);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods) {
        gameOf(w)->onMouseButton(button, action, mods);
    });
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); });

    const bool tour = argc >= 3 && std::string(argv[1]) == "--shots";
    if (tour) game.devTour(argv[2]);

    double last = glfwGetTime();
    while (!tour && !glfwWindowShouldClose(window)) {
        glfwPollEvents();
        const double now = glfwGetTime();
        const float dt = std::min(static_cast<float>(now - last), 0.1f);
        last = now;

        game.update(dt);
        game.render();
        glfwSwapBuffers(window);
    }

    game.shutdown();
    Audio::shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
