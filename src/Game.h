#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <array>
#include <random>
#include <string>
#include <vector>

#include "Camera.h"
#include "Hud.h"
#include "Model.h"
#include "Player.h"
#include "Quest.h"
#include "Scene.h"
#include "Shader.h"

class Game {
public:
    bool init(GLFWwindow* window);       // needs a current GL context; Audio::init() must already have run
    void update(float dt);
    void render();
    void shutdown();                     // stops the looping sounds
    void devTour(const std::string& outDir);  // --shots: scripted screenshot tour (src/DevTour.cpp)

    void onKey(int key, int action, int mods);
    void onMouseButton(int button, int action, int mods);

private:
    enum ShellSide { SIDE_FLOOR, SIDE_CEILING, SIDE_NEG_Z, SIDE_POS_Z, SIDE_NEG_X, SIDE_POS_X };
    struct ShellPiece {
        Mesh mesh;
        glm::mat4 model{1.0f};
        ShellSide side = SIDE_FLOOR;
    };
    struct SceneSlot {
        Scene scene;
        std::vector<ShellPiece> shell;
        bool visited = false;
        glm::vec3 savedPos{0.0f};
        float savedYaw = 0.0f, savedPitch = 0.0f;
    };
    enum class Phase { None, CameraIn, Animate, CameraOut };

    // --- setup / scene management
    void buildShell(SceneSlot& slot);
    void enterScene(int index);
    void requestScene(int index);       // fade to black, then enterScene()
    SceneSlot& slot() { return slots_[current_]; }
    Scene& scene() { return slots_[current_].scene; }

    // --- per-frame logic
    void updatePlayer(float dt, bool acceptInput);
    void collectObstacles();
    int findEligible() const;
    void startInteraction(int index);
    void cancelInteraction();
    CameraPose updateInteraction(float dt, const CameraPose& navPose);
    void applyInteractionAnim(SceneObject& o, float t) const;
    void setHorror(bool on);
    void updateHorror(float dt);
    void updateHorrorObjects(float dt, float prevBlend);
    void updateCameraShake();
    void buildLights();
    float interactionLightWeight() const;
    bool cutAway(ShellSide side) const;
    bool onCutAwayWall(const SceneObject& o) const;

    // --- quest (v2): map -> key -> riddle -> win, plus the jump scare
    bool isHidden(const SceneObject& o) const;        // not drawn, not solid (map before/after, key after the win)
    bool isLocked(const SceneObject& o) const;        // drawn but not interactable yet
    void onInteractionAnimated(int index);            // Animate phase finished
    void onInteractionFinished(int index);            // camera back, interaction complete
    void showToast(const std::string& text, float seconds = 3.5f);
    void toggleMap();
    void openRiddle();
    void answerRiddle(int slot);
    void startScare();
    bool scareActive() const { return scareT_ >= 0.0f; }
    bool modalOpen() const { return riddleOpen_ || winScreen_ || scareActive(); }
    int nurseryIndex() const;
    void renderScare();
    void drawScareOverlay();

    // --- rendering
    void setObjectUniforms(const glm::mat4& model, const glm::vec3& tint, const glm::vec3& emissive);
    void drawMesh(const Mesh& m, const glm::mat4& model, const glm::vec3& tint, const glm::vec3& emissive);
    glm::vec3 backgroundColor() const;  // fog colour after the shader's horror grade (display space)
    void drawHud();
    void cursorInFramebuffer(double& x, double& y) const;
    void pressViewButton(int index);    // 0 Top, 1 Front, 2 Side, 3 Perspective
    float aspect() const { return fbH_ > 0 ? float(fbW_) / float(fbH_) : 16.0f / 9.0f; }

    GLFWwindow* window_ = nullptr;
    Shader shader_;
    Hud hud_;
    ModelCache cache_;
    Camera camera_;
    Player player_;
    std::array<SceneSlot, 2> slots_;
    int current_ = 0;
    int pendingScene_ = -1;              // scene entered once the fade-out completes

    std::vector<Aabb> obstacles3d_;      // solid objects' world AABBs (camera spring arm)
    std::vector<Box2> obstacles2d_;      // same, XZ footprints (player collision)
    LightRig rig_;                       // effective lights for this frame
    Mesh wispMesh_;                      // glowing orb that follows the horror light
    glm::vec3 wispPos_{0.0f};

    // Interaction state machine.
    Phase phase_ = Phase::None;
    int active_ = -1;                    // object being interacted with
    float phaseT_ = 0.0f;
    CameraPose fromPose_;
    SceneObject snapshot_;               // anim state at the end of Animate (eased back to identity)
    int eligible_ = -1;                  // object the [E] prompt refers to
    std::vector<float> hauntAngle_;      // per-object yaw state for HorrorAnim::Spin / StareAtPlayer
    std::vector<float> hauntWeight_;     // per-object 0..1, eases horror motion back in after an interaction

    // Toggles and levels.
    bool flashlightOn_ = true;
    bool horror_ = false;
    bool helpOpen_ = false;
    float moonMul_ = 1.0f, lampMul_ = 1.0f, flashMul_ = 1.0f;

    // Timers.
    float time_ = 0.0f;
    float horrorBlend_ = 0.0f;           // eases 0..1, drives uHorror and every horror effect
    float whisperTimer_ = 8.0f;
    float titleTimer_ = 0.0f;
    float fade_ = 0.0f;                  // black overlay alpha after a scene switch
    float hintTimer_ = 0.0f;             // boundary hint flash
    float bumpCooldown_ = 0.0f;

    // Quest / jump scare.
    Quest quest_;
    bool mapOpen_ = false, riddleOpen_ = false, winScreen_ = false;
    float riddleT_ = 0.0f, winT_ = 0.0f;  // seconds since the riddle / win screen opened
    float scareT_ = -1.0f;                // seconds into the jump scare, < 0 when idle
    bool scareSounded_ = false;
    std::string toast_;
    float toastTimer_ = 0.0f, toastLength_ = 1.0f;
    std::string lastObjective_;
    float objectiveFlash_ = 0.0f;         // highlights the objective line after it changes

    int fbW_ = 1600, fbH_ = 900;
    bool clickPending_ = false;
    std::mt19937 rng_{20261212u};
};
