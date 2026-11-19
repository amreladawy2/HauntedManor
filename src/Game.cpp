#include "Game.h"

#include "Audio.h"
#include "Lighting.h"
#include "Paths.h"
#include "Primitives.h"
#include "Texture.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>

namespace {

constexpr float kPi = 3.14159265359f;
constexpr float kCamEase = 0.9f;        // seconds for the interaction camera to fly in / out
constexpr float kFadeOut = 0.3f;        // scene switch: fade to black...
constexpr float kFadeIn = 0.55f;        // ...and back

// Jump scare timeline (seconds): darkness, lunge, twitching hold, hard cut to black, fade back.
constexpr float kScareDark = 0.18f;
constexpr float kScareLunge = 0.25f;
constexpr float kScareCut = 1.42f;
constexpr float kScareLength = 1.8f;

// Deterministic pseudo-random value in [0, 1).
float hash1(float n) {
    const double s = std::sin(double(n) * 127.1 + 311.7) * 43758.5453;
    return float(s - std::floor(s));
}

// Smooth 1D value noise in [-1, 1].
float noise1(float x) {
    const float i = std::floor(x), f = x - i;
    const float u = f * f * (3.0f - 2.0f * f);
    return glm::mix(hash1(i), hash1(i + 1.0f), u) * 2.0f - 1.0f;
}

float smooth01(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float wrap180(float a) {
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0.0f) a += 360.0f;
    return a - 180.0f;
}

glm::vec3 wrap180(const glm::vec3& v) { return {wrap180(v.x), wrap180(v.y), wrap180(v.z)}; }

std::string upper(std::string s) {
    for (char& c : s) c = char(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// Integer text scale for stb_easy_font (fractional scales make the 1-px strokes uneven).
float textScale(float base, float uiScale) { return std::max(1.0f, std::floor(base * uiScale + 0.25f)); }

void hudFrame(Hud& hud, float x, float y, float w, float h, float t, const glm::vec4& c) {
    hud.rect(x, y, w, t, c);
    hud.rect(x, y + h - t, w, t, c);
    hud.rect(x, y + t, t, h - 2.0f * t, c);
    hud.rect(x + w - t, y + t, t, h - 2.0f * t, c);
}

}  // namespace

// =============================================================================================
// Setup
// =============================================================================================

bool Game::init(GLFWwindow* window) {
    window_ = window;
    glfwGetFramebufferSize(window_, &fbW_, &fbH_);

    if (!shader_.load("shaders/scene.vert", "shaders/scene.frag")) {
        std::fprintf(stderr, "[Game] failed to load the scene shader\n");
        return false;
    }
    shader_.use();
    shader_.setInt("uDiffuse", 0);

    if (!hud_.init()) std::fprintf(stderr, "[Game] HUD init failed; continuing without the overlay\n");

    slots_[0].scene = buildLibrary(cache_);
    slots_[1].scene = buildBedroom(cache_);
    for (SceneSlot& s : slots_) buildShell(s);
    cache_.get("player");      // load up front so the first third-person frame does not hitch
    cache_.get("scare_doll");  // ...and so the jump scare never stalls on a disk read

    wispMesh_ = makeBox(glm::vec3(0.08f), 1.0f, whiteTexture());

    enterScene(0);
    return true;
}

void Game::shutdown() {
    Audio::stop(scene().ambientSound);
    Audio::stop("heartbeat");
}

// Floor, ceiling and four walls facing inward, plus a dark wood skirting board along each wall.
void Game::buildShell(SceneSlot& slot) {
    const RoomShell& r = slot.scene.room;
    const float W = r.size.x, H = r.size.y, D = r.size.z;
    auto tex = [](const std::string& id) { return loadTexture(resolvePath("assets/textures/" + id + ".png")); };
    const GLuint floorT = tex(r.floorTex), wallT = tex(r.wallTex), ceilT = tex(r.ceilingTex), trimT = tex("wood_dark");

    const glm::mat4 I(1.0f);
    const glm::vec3 X(1, 0, 0), Y(0, 1, 0);
    auto add = [&](const Mesh& m, const glm::mat4& model, ShellSide side) { slot.shell.push_back({m, model, side}); };

    // makeQuad faces +Z; rotations turn that normal into the room.
    add(makeQuad(W, D, W * r.floorTile, D * r.floorTile, floorT), glm::rotate(I, glm::radians(-90.0f), X), SIDE_FLOOR);
    add(makeQuad(W, D, W * r.wallTile, D * r.wallTile, ceilT),
        glm::translate(I, {0, H, 0}) * glm::rotate(I, glm::radians(90.0f), X), SIDE_CEILING);
    add(makeQuad(W, H, W * r.wallTile, H * r.wallTile, wallT), glm::translate(I, {0, H * 0.5f, -D * 0.5f}), SIDE_NEG_Z);
    add(makeQuad(W, H, W * r.wallTile, H * r.wallTile, wallT),
        glm::translate(I, {0, H * 0.5f, D * 0.5f}) * glm::rotate(I, glm::radians(180.0f), Y), SIDE_POS_Z);
    add(makeQuad(D, H, D * r.wallTile, H * r.wallTile, wallT),
        glm::translate(I, {-W * 0.5f, H * 0.5f, 0}) * glm::rotate(I, glm::radians(90.0f), Y), SIDE_NEG_X);
    add(makeQuad(D, H, D * r.wallTile, H * r.wallTile, wallT),
        glm::translate(I, {W * 0.5f, H * 0.5f, 0}) * glm::rotate(I, glm::radians(-90.0f), Y), SIDE_POS_X);

    const float sh = 0.16f, st = 0.035f;  // skirting height / thickness
    add(makeBox({W, sh, st}, 1.5f, trimT), glm::translate(I, {0, sh * 0.5f, -D * 0.5f + st * 0.5f}), SIDE_NEG_Z);
    add(makeBox({W, sh, st}, 1.5f, trimT), glm::translate(I, {0, sh * 0.5f, D * 0.5f - st * 0.5f}), SIDE_POS_Z);
    add(makeBox({st, sh, D - 2 * st}, 1.5f, trimT), glm::translate(I, {-W * 0.5f + st * 0.5f, sh * 0.5f, 0}), SIDE_NEG_X);
    add(makeBox({st, sh, D - 2 * st}, 1.5f, trimT), glm::translate(I, {W * 0.5f - st * 0.5f, sh * 0.5f, 0}), SIDE_POS_X);
}

// =============================================================================================
// Scene switching
// =============================================================================================

void Game::enterScene(int index) {
    current_ = index;
    SceneSlot& s = slot();
    if (!s.visited) {
        player_.reset(s.scene.spawnPos, s.scene.spawnYawDeg);
        s.visited = true;
    } else {
        player_.reset(s.savedPos, s.savedYaw, s.savedPitch);
    }
    for (SceneObject& o : s.scene.objects) o.resetAnim();
    hauntAngle_.assign(s.scene.objects.size(), 0.0f);
    hauntWeight_.assign(s.scene.objects.size(), 1.0f);
    Audio::loop(s.scene.ambientSound);

    collectObstacles();
    camera_.snap(camera_.desiredPose(player_, s.scene.room, aspect(), obstacles3d_));
    eligible_ = -1;
    buildLights();
}

void Game::requestScene(int index) {
    const int target = pendingScene_ >= 0 ? pendingScene_ : current_;
    if (index == target) return;
    if (index == current_) {  // changed mind during the fade-out: just fade back in
        pendingScene_ = -1;
        return;
    }
    cancelInteraction();
    Audio::play("switch");
    pendingScene_ = index;
}

// =============================================================================================
// Update
// =============================================================================================

void Game::update(float dt) {
    glfwGetFramebufferSize(window_, &fbW_, &fbH_);
    time_ += dt;
    titleTimer_ += dt;
    hintTimer_ = std::max(0.0f, hintTimer_ - dt);
    bumpCooldown_ = std::max(0.0f, bumpCooldown_ - dt);
    toastTimer_ = std::max(0.0f, toastTimer_ - dt);
    objectiveFlash_ = std::max(0.0f, objectiveFlash_ - dt);
    quest_.update(dt);
    if (riddleOpen_) riddleT_ += dt;
    if (winScreen_) winT_ += dt;
    if (scareActive()) {
        scareT_ += dt;
        if (!scareSounded_ && scareT_ >= kScareDark) {
            Audio::play("scare_hit");
            Audio::play("scream");
            scareSounded_ = true;
        }
        if (scareT_ >= kScareLength) scareT_ = -1.0f;
    }
    if (const std::string objective = quest_.objective(); objective != lastObjective_) {
        if (!lastObjective_.empty()) objectiveFlash_ = 2.5f;
        lastObjective_ = objective;
    }

    if (pendingScene_ >= 0) {
        fade_ += dt / kFadeOut;
        if (fade_ >= 1.0f) {
            fade_ = 1.0f;
            SceneSlot& old = slot();
            old.savedPos = player_.position;
            old.savedYaw = player_.yawDeg;
            old.savedPitch = player_.pitchDeg;
            for (SceneObject& o : old.scene.objects) o.resetAnim();
            Audio::stop(old.scene.ambientSound);
            enterScene(pendingScene_);
            pendingScene_ = -1;
        }
    } else {
        fade_ = std::max(0.0f, fade_ - dt / kFadeIn);
    }

    collectObstacles();
    // Locked during interactions, the riddle, the win screen and the scare (the walk cycle still settles).
    updatePlayer(dt, phase_ == Phase::None && !modalOpen());

    const CameraPose nav = camera_.desiredPose(player_, scene().room, aspect(), obstacles3d_);
    if (phase_ != Phase::None)
        camera_.snap(updateInteraction(dt, nav));
    else
        camera_.follow(nav, dt);

    updateHorror(dt);
    eligible_ = findEligible();
    updateCameraShake();
    buildLights();
}

void Game::updatePlayer(float dt, bool acceptInput) {
    auto down = [&](int k) { return acceptInput && glfwGetKey(window_, k) == GLFW_PRESS; };
    auto axis = [&](int pos, int neg) { return (down(pos) ? 1.0f : 0.0f) - (down(neg) ? 1.0f : 0.0f); };

    Player::Input in;
    in.forward = axis(GLFW_KEY_W, GLFW_KEY_S);
    in.strafe = axis(GLFW_KEY_D, GLFW_KEY_A);
    in.turn = axis(GLFW_KEY_LEFT, GLFW_KEY_RIGHT);
    in.look = axis(GLFW_KEY_UP, GLFW_KEY_DOWN);
    in.run = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT);

    const Player::Events ev = player_.update(in, dt, obstacles2d_, scene().boundsMin, scene().boundsMax);
    if (ev.footstep) Audio::play("footstep");
    if (ev.hitBoundary) {
        hintTimer_ = 1.6f;
        if (bumpCooldown_ <= 0.0f) {
            Audio::play("bump");
            bumpCooldown_ = 0.6f;
        }
    }
}

// World AABBs of solid objects. Only boxes between ankle and head height block walking
// (rugs and hanging chandeliers never do).
void Game::collectObstacles() {
    obstacles3d_.clear();
    obstacles2d_.clear();
    for (const SceneObject& o : scene().objects) {
        if (!o.solid || !o.model || isHidden(o)) continue;
        const glm::mat4 m = o.modelMatrix();
        const glm::vec3 bmin = o.model->boundsMin, bmax = o.model->boundsMax;
        glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
        for (int i = 0; i < 8; ++i) {
            const glm::vec3 c((i & 1) ? bmax.x : bmin.x, (i & 2) ? bmax.y : bmin.y, (i & 4) ? bmax.z : bmin.z);
            const glm::vec3 w(m * glm::vec4(c, 1.0f));
            lo = glm::min(lo, w);
            hi = glm::max(hi, w);
        }
        obstacles3d_.push_back({lo, hi});
        if (hi.y > 0.1f && lo.y < 1.9f) obstacles2d_.push_back({{lo.x, lo.z}, {hi.x, hi.z}});
    }
}

// =============================================================================================
// Interactions
// =============================================================================================

// Nearest enabled object within its range whose centre lies inside the player's view cone.
int Game::findEligible() const {
    if (phase_ != Phase::None || pendingScene_ >= 0 || modalOpen()) return -1;
    const glm::vec2 pp(player_.position.x, player_.position.z);
    const glm::vec3 f = player_.forward();
    const glm::vec2 fwd(f.x, f.z);
    const std::vector<SceneObject>& objs = slots_[current_].scene.objects;

    int best = -1;
    float bestDist = FLT_MAX;
    for (size_t i = 0; i < objs.size(); ++i) {
        const SceneObject& o = objs[i];
        const Interaction& it = o.interaction;
        if (!it.enabled || isHidden(o) || isLocked(o)) continue;
        const float dist = glm::length(glm::vec2(o.position.x, o.position.z) - pp);
        if (dist > it.range) continue;
        const glm::vec3 c = o.worldCenter();
        const glm::vec2 to(c.x - pp.x, c.z - pp.y);
        if (glm::length(to) > 1e-3f) {
            const float cosAngle = glm::dot(glm::normalize(to), fwd);
            if (cosAngle < std::cos(glm::radians(it.viewAngleDeg))) continue;
        }
        if (dist < bestDist) {
            bestDist = dist;
            best = int(i);
        }
    }
    return best;
}

void Game::startInteraction(int index) {
    SceneObject& o = scene().objects[size_t(index)];
    o.resetAnim();  // drop any horror pose; the interaction drives the object from its initial state
    if (size_t(index) < hauntAngle_.size()) {
        hauntAngle_[size_t(index)] = 0.0f;   // horror motion restarts from rest afterwards...
        hauntWeight_[size_t(index)] = 0.0f;  // ...and eases back in
    }
    active_ = index;
    phase_ = Phase::CameraIn;
    phaseT_ = 0.0f;
    fromPose_ = camera_.pose;
    Audio::play(o.interaction.sound.empty() ? "interact" : o.interaction.sound);
}

void Game::cancelInteraction() {
    if (phase_ == Phase::None) return;
    std::vector<SceneObject>& objs = scene().objects;
    if (active_ >= 0 && size_t(active_) < objs.size()) objs[size_t(active_)].resetAnim();
    phase_ = Phase::None;
    active_ = -1;
    phaseT_ = 0.0f;
}

// Implements the AnimFlags exactly as documented in Scene.h; t = seconds since Animate began.
void Game::applyInteractionAnim(SceneObject& o, float t) const {
    const Interaction& it = o.interaction;
    const float dur = std::max(it.duration, 0.1f);
    o.resetAnim();
    if (it.anims & ANIM_ROTATE) o.animRotDeg += it.rotAxis * std::fmod(it.rotSpeedDeg * t, 360.0f);
    if (it.anims & ANIM_SWING)
        o.animRotDeg += it.rotAxis * (it.swingAmpDeg * std::sin(2.0f * kPi * it.swingFreq * t));
    if (it.anims & ANIM_TRANSLATE) o.animTranslate = it.translateBy * smooth01(t / (0.4f * dur));
    if (it.anims & ANIM_SCALE)
        o.animScale = 1.0f + (it.scaleTo - 1.0f) * (0.5f - 0.5f * std::cos(2.0f * kPi * 1.0f * t));
    if (it.anims & ANIM_COLOR) {
        const float c = smooth01(t / (0.3f * dur));
        o.animTint = glm::mix(glm::vec3(1.0f), it.colorTo, c);
        o.animEmissive = it.emissiveTo * c;
    }
}

// Advances CameraIn -> Animate -> CameraOut and returns the camera pose for this frame.
CameraPose Game::updateInteraction(float dt, const CameraPose& navPose) {
    std::vector<SceneObject>& objs = scene().objects;
    if (active_ < 0 || size_t(active_) >= objs.size()) {
        cancelInteraction();
        return navPose;
    }
    SceneObject& o = objs[size_t(active_)];
    const Interaction& it = o.interaction;
    const float dur = std::max(it.duration, 0.1f);
    CameraPose focus;
    focus.eye = it.camPos;
    focus.target = it.camTarget;
    focus.up = glm::vec3(0.0f, 1.0f, 0.0f);

    phaseT_ += dt;
    switch (phase_) {
    case Phase::CameraIn: {
        const float e = smooth01(phaseT_ / kCamEase);
        if (phaseT_ >= kCamEase) {
            phase_ = Phase::Animate;
            phaseT_ = 0.0f;
        }
        return lerpPose(fromPose_, focus, e);
    }
    case Phase::Animate:
        applyInteractionAnim(o, std::min(phaseT_, dur));
        if (phaseT_ >= dur) {
            snapshot_ = o;
            snapshot_.animRotDeg = wrap180(o.animRotDeg);  // ease back the short way round
            phase_ = Phase::CameraOut;
            phaseT_ = 0.0f;
            onInteractionAnimated(active_);
        }
        return focus;
    case Phase::CameraOut: {
        const float e = smooth01(phaseT_ / kCamEase);
        o.animTranslate = glm::mix(snapshot_.animTranslate, glm::vec3(0.0f), e);
        o.animRotDeg = glm::mix(snapshot_.animRotDeg, glm::vec3(0.0f), e);
        o.animScale = glm::mix(snapshot_.animScale, 1.0f, e);
        o.animTint = glm::mix(snapshot_.animTint, glm::vec3(1.0f), e);
        o.animEmissive = glm::mix(snapshot_.animEmissive, glm::vec3(0.0f), e);
        if (phaseT_ >= kCamEase) {
            o.resetAnim();
            const int finished = active_;
            phase_ = Phase::None;
            active_ = -1;
            phaseT_ = 0.0f;
            onInteractionFinished(finished);
            return navPose;
        }
        return lerpPose(focus, navPose, e);
    }
    case Phase::None:
        break;
    }
    return navPose;
}

// 0..1 envelope for the linked light's boost: ramps with the colour change, fades on the way out.
float Game::interactionLightWeight() const {
    if (active_ < 0) return 0.0f;
    const float dur = std::max(slots_[current_].scene.objects[size_t(active_)].interaction.duration, 0.1f);
    switch (phase_) {
    case Phase::Animate: return smooth01(phaseT_ / (0.3f * dur));
    case Phase::CameraOut: return 1.0f - smooth01(phaseT_ / kCamEase);
    default: return 0.0f;
    }
}

// =============================================================================================
// Horror mode
// =============================================================================================

void Game::setHorror(bool on) {
    horror_ = on;
    if (on) {
        if (horrorBlend_ <= 0.0f) std::fill(hauntAngle_.begin(), hauntAngle_.end(), 0.0f);
        Audio::play("horror_on");
        Audio::loop("heartbeat");
        whisperTimer_ = std::uniform_real_distribution<float>(6.0f, 12.0f)(rng_);
    } else {
        Audio::stop("heartbeat");
    }
}

void Game::updateHorror(float dt) {
    const float prev = horrorBlend_;
    horrorBlend_ = horror_ ? std::min(1.0f, horrorBlend_ + dt / 1.5f) : std::max(0.0f, horrorBlend_ - dt / 0.9f);

    if (horror_) {
        whisperTimer_ -= dt;
        if (whisperTimer_ <= 0.0f) {
            Audio::play("whisper");
            whisperTimer_ = std::uniform_real_distribution<float>(6.0f, 12.0f)(rng_);
        }
    }

    if (horrorBlend_ > 0.0f) {
        updateHorrorObjects(dt, prev);
    } else if (prev > 0.0f) {
        std::vector<SceneObject>& objs = scene().objects;
        for (size_t i = 0; i < objs.size(); ++i)
            if (int(i) != active_) objs[i].resetAnim();
    }
}

// Drives each object's HorrorAnim; every effect is scaled by horrorBlend_ so it eases in and out.
void Game::updateHorrorObjects(float dt, float prevBlend) {
    const float t = time_;
    std::vector<SceneObject>& objs = scene().objects;
    if (hauntAngle_.size() != objs.size()) {
        hauntAngle_.assign(objs.size(), 0.0f);
        hauntWeight_.assign(objs.size(), 1.0f);
    }

    for (size_t i = 0; i < objs.size(); ++i) {
        if (int(i) == active_) continue;
        SceneObject& o = objs[i];
        hauntWeight_[i] = std::min(1.0f, hauntWeight_[i] + dt / 0.8f);
        const float h = horrorBlend_ * smooth01(hauntWeight_[i]);
        const float ph = float(i) * 1.37f;  // per-object phase so things don't move in lockstep

        switch (o.horror) {
        case HorrorAnim::None:
            break;
        case HorrorAnim::Sway:  // hanging things swing slowly from their pivot
            o.animRotDeg = glm::vec3(1.5f * std::sin(t * 0.9f + ph), 0.0f, 5.0f * std::sin(t * 1.3f + ph)) * h;
            break;
        case HorrorAnim::Float:
            o.animTranslate = glm::vec3(0.0f, 0.18f + 0.08f * std::sin(t * 1.6f + ph), 0.0f) * h;
            o.animRotDeg = glm::vec3(4.0f * std::sin(t * 0.8f + ph), 12.0f * std::sin(t * 0.5f + ph),
                                     3.0f * std::sin(t * 1.1f + ph)) * h;
            break;
        case HorrorAnim::Rattle: {  // violent bursts every few seconds, still in between
            const float burst = smooth01((noise1(t * 0.6f + ph * 3.0f) - 0.25f) * 3.0f) * h;
            o.animRotDeg = glm::vec3(noise1(t * 31.0f + ph) * 1.2f, noise1(t * 27.0f + ph + 5.0f) * 1.5f,
                                     noise1(t * 35.0f + ph + 9.0f) * 1.2f) * burst;
            o.animTranslate = glm::vec3(noise1(t * 29.0f + ph + 2.0f) * 0.012f,
                                        std::fabs(noise1(t * 33.0f + ph + 4.0f)) * 0.01f,
                                        noise1(t * 25.0f + ph + 7.0f) * 0.012f) * burst;
            break;
        }
        case HorrorAnim::Rock:  // rocks on its rockers (about model X)
            o.animRotDeg = glm::vec3(9.0f * std::sin(t * 2.1f + ph), 0.0f, 0.0f) * h;
            break;
        case HorrorAnim::Spin: {
            // Speeds up with the blend; when horror turns off it unwinds the short way round,
            // shrinking in step with the blend so it is exactly at rest when the blend reaches 0.
            float& a = hauntAngle_[i];
            if (horror_)
                a = std::fmod(a + 90.0f * h * dt, 360.0f);
            else
                a = prevBlend > 0.0f ? wrap180(a) * (horrorBlend_ / prevBlend) : 0.0f;
            o.animRotDeg = glm::vec3(0.0f, a, 0.0f);
            break;
        }
        case HorrorAnim::StareAtPlayer: {
            // Slowly turn about Y so the model's front (+Z) faces the player.
            const glm::vec3 c = o.worldCenter();
            const float want = glm::degrees(std::atan2(player_.position.x - c.x, player_.position.z - c.z));
            const float rel = wrap180(want - o.rotationDeg.y);
            float& s = hauntAngle_[i];
            s = wrap180(s + wrap180(rel - s) * (1.0f - std::exp(-2.5f * dt)));
            o.animRotDeg = glm::vec3(0.0f, s * h, 0.0f);
            break;
        }
        case HorrorAnim::Pulse: {  // heartbeat-like swell with a blood-red glow
            const float beat = std::pow(0.5f + 0.5f * std::sin(t * 3.2f + ph), 3.0f);
            o.animScale = 1.0f + 0.05f * beat * h;
            o.animTint = glm::mix(glm::vec3(1.0f), glm::vec3(1.5f, 0.55f, 0.5f), h * (0.4f + 0.6f * beat));
            o.animEmissive = glm::vec3(0.22f, 0.01f, 0.0f) * beat * h;
            break;
        }
        }
    }
}

// Horror camera: layered-noise shake with occasional jolts, slow breathing sway and a slight roll.
void Game::updateCameraShake() {
    camera_.shakeEye = camera_.shakeTarget = glm::vec3(0.0f);
    camera_.rollDeg = 0.0f;

    // Jump scare: violent shake while the doll is in your face.
    if (scareActive() && scareT_ >= kScareDark && scareT_ < kScareCut) {
        const float t = time_, landed = smooth01((scareT_ - kScareDark) / kScareLunge);
        const float amp = 0.012f + 0.03f * landed;
        camera_.shakeEye = glm::vec3(noise1(t * 47.0f), noise1(t * 53.0f + 3.0f), noise1(t * 59.0f + 7.0f)) * amp;
        camera_.shakeTarget =
            glm::vec3(noise1(t * 43.0f + 11.0f), noise1(t * 39.0f + 13.0f), noise1(t * 51.0f + 17.0f)) * (amp * 3.0f);
        camera_.rollDeg = noise1(t * 37.0f + 19.0f) * 6.0f * landed;
    }

    const float h = horrorBlend_;
    if (h <= 0.0f) return;
    const float t = time_;
    auto layered = [t](float off) {
        return noise1(t * 2.1f + off) * 0.55f + noise1(t * 7.3f + off * 1.7f) * 0.3f +
               noise1(t * 19.0f + off * 2.3f) * 0.15f;
    };
    const glm::vec3 f = camera_.pose.target - camera_.pose.eye;
    const float dist = glm::length(f);
    const glm::vec3 fwd = dist > 1e-4f ? f / dist : glm::vec3(0, 0, -1);
    glm::vec3 right = glm::cross(fwd, camera_.pose.up);
    right = glm::length(right) > 1e-4f ? glm::normalize(right) : glm::vec3(1, 0, 0);
    const glm::vec3 up = glm::normalize(glm::cross(right, fwd));
    const float k = std::max(1.0f, dist / 3.0f);  // stay visible from the distant fixed views

    const float jolt = smooth01((noise1(t * 0.45f + 50.0f) - 0.5f) * 4.0f);
    const float amp = (0.022f + 0.05f * jolt) * h * k;
    const float breath = std::sin(t * 1.35f);  // ~13 breaths per minute

    camera_.shakeEye += glm::vec3(layered(0.0f), layered(13.0f), layered(29.0f)) * amp + up * (0.035f * breath * h * k);
    camera_.shakeTarget += glm::vec3(layered(41.0f), layered(57.0f), layered(73.0f)) * (amp * 2.0f) +
                           (right * (0.07f * std::sin(t * 0.62f)) + up * (0.03f * std::sin(t * 1.35f + 0.6f))) * (h * k);
    camera_.rollDeg += (1.4f * std::sin(t * 0.5f) + layered(91.0f) * 1.2f * (1.0f + 2.0f * jolt)) * h;
}

// =============================================================================================
// Quest (v2): Library clues -> map -> key -> riddle -> win
// =============================================================================================

bool Game::isHidden(const SceneObject& o) const {
    if (o.name == Quest::kMapObject) return !quest_.mapOnDesk();
    if (o.name == Quest::kKeyObject) return quest_.won();
    return false;
}

bool Game::isLocked(const SceneObject& o) const {
    return o.name == Quest::kKeyObject && !quest_.keyAvailable();
}

// The map is "taken" the moment its animation ends, so it is gone while the camera pulls back.
void Game::onInteractionAnimated(int index) {
    const SceneObject& o = scene().objects[size_t(index)];
    if (o.name == Quest::kMapObject && quest_.mapOnDesk()) {
        quest_.stage = QuestStage::SearchKey;
        Audio::play("pickup");
        showToast("You took a map of the Nursery. Press M to read it.", 5.0f);
    }
}

void Game::onInteractionFinished(int index) {
    const std::string& name = scene().objects[size_t(index)].name;
    if (quest_.recordClue(name)) {
        Audio::play("objective");
        showToast("Something rustles on the desk...", 4.5f);
    }
    if (name == Quest::kKeyObject && quest_.keyAvailable()) openRiddle();
}

void Game::showToast(const std::string& text, float seconds) {
    toast_ = text;
    toastTimer_ = toastLength_ = seconds;
}

void Game::toggleMap() {
    if (!quest_.hasMap()) {
        mapOpen_ = false;
        showToast("You have no map.", 2.2f);
        return;
    }
    mapOpen_ = !mapOpen_;
    Audio::play(mapOpen_ ? "pickup" : "click");
}

void Game::openRiddle() {
    if (quest_.stage == QuestStage::SearchKey) quest_.stage = QuestStage::Riddle;
    quest_.shuffleChoices(rng_);
    riddleOpen_ = true;
    riddleT_ = 0.0f;
    mapOpen_ = false;
    Audio::play("riddle");
}

void Game::answerRiddle(int slot) {
    if (!riddleOpen_ || slot < 0 || slot >= Quest::kChoiceCount) return;
    riddleOpen_ = false;
    if (quest_.isCorrect(slot)) {
        quest_.stage = QuestStage::Won;
        winScreen_ = true;
        winT_ = 0.0f;
        mapOpen_ = false;
        Audio::play("win");
    } else {
        quest_.triedWrong = true;
        startScare();
    }
}

void Game::startScare() {
    if (scareActive() || winScreen_) return;
    scareT_ = 0.0f;
    scareSounded_ = false;
}

int Game::nurseryIndex() const {
    for (size_t i = 0; i < slots_.size(); ++i)
        for (const SceneObject& o : slots_[i].scene.objects)
            if (o.name == Quest::kKeyObject) return int(i);
    return 1;
}

// =============================================================================================
// Lights
// =============================================================================================

void Game::buildLights() {
    const Scene& sc = scene();
    rig_ = sc.lights;
    const float t = time_, h = horrorBlend_;

    // Moonlight: user dimmer; under horror it weakens and turns a sickly green.
    rig_.dir.intensity *= moonMul_ * glm::mix(1.0f, 0.6f + 0.1f * std::sin(t * 0.7f), h);
    rig_.dir.color = glm::mix(rig_.dir.color, glm::vec3(0.55f, 0.85f, 0.4f), 0.75f * h);

    // Lamps, candles and fire: natural flicker, user dimmer, erratic dimming under horror.
    for (size_t i = 0; i < rig_.points.size(); ++i) {
        PointLight& p = rig_.points[i];
        const float fi = float(i) * 7.31f;
        float k = 1.0f;
        if (p.flicker > 0.0f) {
            const float wave = 0.5f * std::sin(t * 7.3f + fi) + 0.3f * std::sin(t * 13.1f + fi * 1.3f) +
                               0.2f * std::sin(t * 23.7f + fi * 0.7f);
            k *= 1.0f + p.flicker * (0.22f * wave + 0.18f * noise1(t * 11.0f + fi));
        }
        if (h > 0.0f) {
            float erratic = 0.55f + 0.45f * noise1(t * 9.0f + fi * 3.0f);
            if (hash1(std::floor(t * 8.0f) + fi) > 0.85f) erratic *= 0.12f;  // brief dropouts
            k *= glm::mix(1.0f, 0.9f * erratic, h);
        }
        p.intensity *= std::max(k, 0.0f) * lampMul_;
    }

    if (active_ >= 0) {
        const Interaction& it = sc.objects[size_t(active_)].interaction;
        if (it.linkedPointLight >= 0 && size_t(it.linkedPointLight) < rig_.points.size())
            rig_.points[size_t(it.linkedPointLight)].intensity += it.lightBoost * interactionLightWeight();
    }

    // Horror: a ghostly light wanders the room on a Lissajous path.
    if (h > 0.0f) {
        const glm::vec3 size = sc.room.size;
        const float a = t * 0.38f;
        wispPos_ = glm::vec3(std::cos(a) * size.x * 0.32f + noise1(t * 0.7f) * 0.4f,
                             1.35f + 0.5f * std::sin(t * 0.9f) + 0.2f * noise1(t * 1.3f + 5.0f),
                             std::sin(a * 1.3f) * size.z * 0.32f + noise1(t * 0.6f + 9.0f) * 0.4f);
        PointLight wisp;
        wisp.position = wispPos_;
        wisp.color = glm::vec3(0.45f, 1.0f, 0.75f);
        wisp.intensity = 1.8f * h * (0.75f + 0.25f * std::sin(t * 5.0f));
        wisp.range = 5.0f;
        if (rig_.points.size() < size_t(MAX_POINT_LIGHTS))
            rig_.points.push_back(wisp);
        else
            rig_.points.back() = wisp;
    }

    // Flashlight: from the eye in first person, otherwise from the chest; always along the look direction.
    SpotLight& s = rig_.spot;
    s.on = flashlightOn_;
    if (camera_.mode == ViewMode::FirstPerson)
        s.position = player_.eyePosition() + player_.right() * 0.08f - glm::vec3(0.0f, 0.08f, 0.0f);
    else
        s.position = player_.position + glm::vec3(0.0f, 1.3f, 0.0f) + player_.forward() * 0.3f;
    s.direction = player_.lookDir();
    float flash = flashMul_;
    if (h > 0.0f) {
        float stutter = 1.0f - 0.25f * h * (0.5f + 0.5f * std::sin(t * 37.0f));
        if (hash1(std::floor(t * 14.0f) + 3.0f) < 0.12f * h) stutter = 0.05f;  // rapid stutter
        if (h > 0.5f && noise1(t * 0.5f + 77.0f) > 0.72f) stutter = 0.03f;     // dies for a moment
        flash *= stutter;
    }
    s.intensity *= flash;
}

// =============================================================================================
// Rendering
// =============================================================================================

bool Game::cutAway(ShellSide side) const {
    switch (camera_.mode) {
    case ViewMode::Top: return side == SIDE_CEILING;
    case ViewMode::Front: return side == SIDE_POS_Z;
    case ViewMode::Side: return side == SIDE_POS_X;
    default: return false;
    }
}

// Thin wall-mounted pieces (portrait, window) vanish together with the cut-away wall they hang on.
bool Game::onCutAwayWall(const SceneObject& o) const {
    const glm::vec3 c = o.worldCenter(), half = slots_[current_].scene.room.size * 0.5f;
    switch (camera_.mode) {
    case ViewMode::Front: return c.z > half.z - 0.15f;
    case ViewMode::Side: return c.x > half.x - 0.15f;
    default: return false;
    }
}

void Game::setObjectUniforms(const glm::mat4& model, const glm::vec3& tint, const glm::vec3& emissive) {
    shader_.setMat4("uModel", model);
    shader_.setMat3("uNormalMat", glm::transpose(glm::inverse(glm::mat3(model))));
    shader_.setVec3("uTint", tint);
    shader_.setVec3("uEmissive", emissive);
}

void Game::drawMesh(const Mesh& m, const glm::mat4& model, const glm::vec3& tint, const glm::vec3& emissive) {
    setObjectUniforms(model, tint, emissive);
    m.draw(shader_);
}

// Mirrors the fog + horror grade in scene.frag so the background matches fully fogged geometry.
glm::vec3 Game::backgroundColor() const {
    const float h = horrorBlend_;
    const glm::vec3 fog = slots_[current_].scene.fogColor;
    glm::vec3 c = glm::mix(glm::pow(glm::max(fog, glm::vec3(0.0f)), glm::vec3(2.2f)),
                           glm::vec3(0.025f, 0.0f, 0.002f), h);
    c = glm::mix(c, glm::vec3(glm::dot(c, glm::vec3(0.2126f, 0.7152f, 0.0722f))), 0.4f * h);
    c *= glm::mix(glm::vec3(1.0f), glm::vec3(1.35f, 0.72f, 0.68f), h) * (1.0f - 0.08f * h);
    return glm::pow(glm::clamp(c, 0.0f, 1.0f), glm::vec3(1.0f / 2.2f));
}

void Game::render() {
    if (fbW_ <= 0 || fbH_ <= 0) {  // minimised
        clickPending_ = false;
        return;
    }
    glViewport(0, 0, fbW_, fbH_);
    const Scene& sc = scene();
    const float h = horrorBlend_;

    const glm::vec3 bg = backgroundColor();
    glClearColor(bg.r, bg.g, bg.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader_.use();
    shader_.setMat4("uView", camera_.view());
    shader_.setMat4("uProj", camera_.projection(aspect()));
    shader_.setVec3("uViewPos", camera_.eye());
    shader_.setVec3("uAmbient", sc.ambient * glm::mix(1.0f, 0.9f, h));
    shader_.setFloat("uTime", time_);
    shader_.setFloat("uHorror", h);
    shader_.setVec3("uFogColor", sc.fogColor);
    shader_.setFloat("uFogDensity", sc.fogDensity);  // the shader thickens it under horror itself
    shader_.setVec2("uScreenSize", glm::vec2(float(fbW_), float(fbH_)));
    uploadLights(shader_, rig_);

    for (const ShellPiece& p : slot().shell)
        if (!cutAway(p.side)) drawMesh(p.mesh, p.model, glm::vec3(1.0f), glm::vec3(0.0f));

    for (const SceneObject& o : sc.objects) {
        if (!o.model || onCutAwayWall(o) || isHidden(o)) continue;
        glm::mat4 m = o.modelMatrix();
        glm::vec3 emissive = o.emissive + o.animEmissive;
        if (o.name == Quest::kMapObject) {
            // Materialises on the desk: grows out of a warm glow, then keeps a faint beckoning shimmer.
            const float r = quest_.revealTime;
            const glm::vec3 c = o.worldCenter();
            const float grow = glm::mix(0.05f, 1.0f, smooth01(r / 1.6f));
            m = glm::translate(glm::mat4(1.0f), c) * glm::scale(glm::mat4(1.0f), glm::vec3(grow)) *
                glm::translate(glm::mat4(1.0f), -c) * m;
            const float glow = 0.8f * (1.0f - smooth01((r - 1.0f) / 2.0f)) + 0.1f * (0.5f + 0.5f * std::sin(time_ * 3.0f));
            emissive += glm::vec3(1.0f, 0.72f, 0.38f) * glow;
        } else if (o.name == Quest::kKeyObject && quest_.keyAvailable() && phase_ == Phase::None) {
            emissive += glm::vec3(0.35f, 0.25f, 0.08f) * std::pow(0.5f + 0.5f * std::sin(time_ * 2.3f), 6.0f);  // glint
        }
        setObjectUniforms(m, o.tint * o.animTint, emissive);
        o.model->draw(shader_);
    }

    // Player model: hidden in first person, and whenever the camera would sit inside the body.
    const glm::vec3 eye = camera_.eye();
    const bool eyeInBody = glm::length(glm::vec2(eye.x - player_.position.x, eye.z - player_.position.z)) < 0.45f &&
                           eye.y < player_.position.y + 2.0f;
    // During an interaction the object must fill the view, so the player steps aside if in the way.
    bool blocksFocus = false;
    if (phase_ != Phase::None) {
        const glm::vec2 a(eye.x, eye.z), b(camera_.pose.target.x, camera_.pose.target.z);
        const glm::vec2 pp(player_.position.x, player_.position.z), ab = b - a;
        const float t = glm::dot(ab, ab) > 1e-6f ? std::clamp(glm::dot(pp - a, ab) / glm::dot(ab, ab), 0.0f, 1.0f) : 0.0f;
        blocksFocus = glm::length(pp - (a + ab * t)) < 0.55f;
    }
    if (camera_.mode != ViewMode::FirstPerson && !eyeInBody && !blocksFocus) {
        setObjectUniforms(player_.modelMatrix(), glm::vec3(1.0f), glm::vec3(0.0f));
        cache_.get("player")->draw(shader_);
    }

    // The wandering horror light is visible as a small glowing shard.
    if (h > 0.01f) {
        const glm::vec3 glow = glm::vec3(0.7f, 1.5f, 1.1f) * h * (0.8f + 0.2f * std::sin(time_ * 5.0f));
        glm::mat4 m = glm::translate(glm::mat4(1.0f), wispPos_);
        m = glm::rotate(m, time_ * 1.7f, glm::normalize(glm::vec3(1.0f, 1.0f, 0.3f)));
        drawMesh(wispMesh_, m, glm::vec3(0.0f), glow);
        drawMesh(wispMesh_, glm::rotate(m, glm::radians(45.0f), glm::vec3(0, 1, 0)) *
                                glm::scale(glm::mat4(1.0f), glm::vec3(0.7f, 1.6f, 0.7f)),
                 glm::vec3(0.0f), glow);
    }

    if (scareActive()) renderScare();
    drawHud();
}

// The doll is drawn after the scene with the depth buffer cleared, positioned from the camera's
// unshaken pose (so the scare shake makes it jolt on screen) and lit only by its own lights.
void Game::renderScare() {
    const float t = scareT_;
    if (t < kScareDark || t >= kScareCut) return;  // darkness and hard cut are HUD overlays
    const Model* doll = cache_.get("scare_doll");

    // The room sinks into darkness behind it.
    hud_.begin(fbW_, fbH_);
    hud_.rect(0, 0, float(fbW_), float(fbH_), {0.0f, 0.0f, 0.0f, 0.74f});
    hud_.end();

    const glm::vec3 eye = camera_.pose.eye;
    glm::vec3 fwd = camera_.pose.target - eye;
    fwd = glm::length(fwd) > 1e-4f ? glm::normalize(fwd) : glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 right = glm::cross(fwd, camera_.pose.up);
    right = glm::length(right) > 1e-4f ? glm::normalize(right) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 up = glm::normalize(glm::cross(right, fwd));

    // Lunge from 3 m, accelerating, then twitch: fast jitter plus sudden head jerks.
    const float e = std::clamp((t - kScareDark) / kScareLunge, 0.0f, 1.0f);
    const float dist = glm::mix(3.0f, 0.4f, e * e);
    const float strength = glm::mix(0.35f, 1.0f, e);
    const float jerk = hash1(std::floor(t * 9.0f) + 5.0f) > 0.55f ? 1.0f : 0.0f;
    const glm::vec3 twitch(noise1(t * 38.0f) * 6.0f + jerk * 10.0f * noise1(std::floor(t * 9.0f) * 3.1f),
                           noise1(t * 31.0f + 4.0f) * 8.0f, noise1(t * 45.0f + 8.0f) * 9.0f + jerk * 16.0f);
    const glm::vec3 jitter = (right * noise1(t * 41.0f + 2.0f) + up * noise1(t * 37.0f + 6.0f)) * (0.015f * strength);

    // Anchor near the face (upper part of the model, centred), so the face lands mid-screen.
    const glm::vec3 bmin = doll->boundsMin, bmax = doll->boundsMax;
    const glm::vec3 anchor((bmin.x + bmax.x) * 0.5f, glm::mix(bmin.y, bmax.y, 0.8f), (bmin.z + bmax.z) * 0.5f);
    glm::mat4 basis(1.0f);  // model +X right, +Y up, +Z (its face) toward the viewer
    basis[0] = glm::vec4(right, 0.0f);
    basis[1] = glm::vec4(up, 0.0f);
    basis[2] = glm::vec4(-fwd, 0.0f);
    glm::mat4 m = glm::translate(glm::mat4(1.0f), eye + fwd * dist + jitter) * basis;
    m = glm::rotate(m, glm::radians(twitch.y * strength), glm::vec3(0, 1, 0));
    m = glm::rotate(m, glm::radians(twitch.x * strength), glm::vec3(1, 0, 0));
    m = glm::rotate(m, glm::radians(twitch.z * strength), glm::vec3(0, 0, 1));
    m = glm::translate(m, -anchor);

    // Its own lighting: a hot under-light just below the face and a cold rim light behind it.
    LightRig rig;
    rig.dir.direction = fwd - up * 0.5f;
    rig.dir.color = glm::vec3(0.6f, 0.6f, 0.9f);
    rig.dir.intensity = 0.08f;
    PointLight under;
    under.position = eye + fwd * (dist * 0.55f) - up * 0.42f;
    under.color = glm::vec3(1.0f, 0.42f, 0.22f);
    under.intensity = 3.2f * (0.85f + 0.15f * noise1(t * 30.0f));
    under.range = 3.0f;
    PointLight rim;
    rim.position = eye + fwd * (dist + 0.6f) + up * 0.7f - right * 0.5f;
    rim.color = glm::vec3(0.55f, 0.65f, 1.0f);
    rim.intensity = 2.2f;
    rim.range = 3.0f;
    rig.points = {under, rim};
    rig.spot.on = false;

    glClear(GL_DEPTH_BUFFER_BIT);
    shader_.use();
    uploadLights(shader_, rig);
    shader_.setVec3("uAmbient", glm::vec3(0.06f, 0.02f, 0.02f));
    shader_.setFloat("uFogDensity", 0.0f);
    shader_.setFloat("uHorror", std::max(horrorBlend_, 0.55f));
    setObjectUniforms(m, glm::vec3(1.1f, 1.0f, 1.0f), glm::vec3(0.03f, 0.0f, 0.0f));
    doll->draw(shader_);
}

// Darkness before the lunge, red/white flash frames, heavy red vignette, hard cut and fade back.
void Game::drawScareOverlay() {
    const float W = float(fbW_), H = float(fbH_), t = scareT_;
    if (t < kScareDark) {
        hud_.rect(0, 0, W, H, {0.0f, 0.0f, 0.0f, 0.94f * smooth01(t / (kScareDark * 0.8f))});
        return;
    }
    if (t >= kScareCut) {
        const float hold = 0.1f;
        const float a = t < kScareCut + hold ? 1.0f : 1.0f - smooth01((t - kScareCut - hold) / (kScareLength - kScareCut - hold));
        hud_.rect(0, 0, W, H, {0.0f, 0.0f, 0.0f, a});
        return;
    }
    // Stacked frames all starting at the screen edge build a thick red vignette.
    const float step = std::min(W, H) * 0.024f;
    const float pulse = 0.85f + 0.15f * std::sin(t * 40.0f);
    for (int i = 0; i < 14; ++i)
        hudFrame(hud_, 0, 0, W, H, step * float(i + 1), {0.5f, 0.0f, 0.0f, 0.085f * pulse});

    const float since = t - kScareDark;
    if (since < 0.5f) {  // strobe while it lunges
        const int f = int(since * 30.0f);
        const float fall = 1.0f - since / 0.5f;
        if (f % 3 == 0) hud_.rect(0, 0, W, H, {1.0f, 0.95f, 0.9f, 0.65f * fall});
        else if (f % 3 == 1) hud_.rect(0, 0, W, H, {0.85f, 0.0f, 0.0f, 0.5f * fall});
    } else if (hash1(std::floor(t * 18.0f) + 7.0f) > 0.82f) {
        hud_.rect(0, 0, W, H, {0.9f, 0.04f, 0.02f, 0.3f});
    }
}

// =============================================================================================
// HUD
// =============================================================================================

void Game::cursorInFramebuffer(double& x, double& y) const {
    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(window_, &cx, &cy);
    int ww = 0, wh = 0;
    glfwGetWindowSize(window_, &ww, &wh);
    // Window coordinates differ from framebuffer pixels on HiDPI displays.
    x = ww > 0 ? cx * double(fbW_) / double(ww) : cx;
    y = wh > 0 ? cy * double(fbH_) / double(wh) : cy;
}

void Game::pressViewButton(int index) {
    switch (index) {
    case 0: camera_.setMode(ViewMode::Top); break;
    case 1: camera_.setMode(ViewMode::Front); break;
    case 2: camera_.setMode(ViewMode::Side); break;
    default: camera_.togglePerspective(); break;
    }
    Audio::play("click");
}

void Game::drawHud() {
    const float W = float(fbW_), H = float(fbH_);
    const float s = std::max(0.75f, H / 900.0f);  // UI scale for resolution / HiDPI
    const float pad = std::round(18.0f * s);
    const float tsSmall = textScale(2.0f, s), tsMed = textScale(3.0f, s), tsBig = textScale(4.0f, s);
    double mx = 0.0, my = 0.0;
    cursorInFramebuffer(mx, my);

    const glm::vec4 parchment(0.93f, 0.87f, 0.75f, 1.0f), dim(0.78f, 0.72f, 0.62f, 0.92f);
    const glm::vec4 accent(0.96f, 0.63f, 0.22f, 1.0f), panel(0.02f, 0.015f, 0.015f, 0.62f);
    auto withAlpha = [](glm::vec4 c, float a) { c.a *= a; return c; };

    hud_.begin(fbW_, fbH_);

    // Nothing but the scare while it plays.
    if (scareActive()) {
        drawScareOverlay();
        hud_.end();
        clickPending_ = false;
        return;
    }
    float leftHudRight = 0.0f, topHudBottom = 0.0f;  // free region for the map sheet

    // Cinematic letterbox and title while an interaction plays.
    if (phase_ != Phase::None && active_ >= 0) {
        float a = 1.0f;
        if (phase_ == Phase::CameraIn) a = smooth01(phaseT_ / 0.5f);
        if (phase_ == Phase::CameraOut) a = 1.0f - smooth01(phaseT_ / 0.6f);
        const float bar = H * 0.085f * a;
        hud_.rect(0, 0, W, bar, {0, 0, 0, 0.9f});
        hud_.rect(0, H - bar, W, bar, {0, 0, 0, 0.9f});
        const std::string title = upper(scene().objects[size_t(active_)].interaction.prompt);
        const float tw = hud_.textWidth(title, tsBig);
        const float ty = H - bar * 0.5f - 3.5f * tsBig;
        hud_.text((W - tw) * 0.5f, ty, title, tsBig, withAlpha(parchment, a));
    }

    // Scene name and light levels (top-left).
    {
        struct Row { const char* label; std::string value; const char* keys; glm::vec4 color; };
        char moon[16], lamps[16], flash[24];
        std::snprintf(moon, sizeof moon, "%.1f", moonMul_);
        std::snprintf(lamps, sizeof lamps, "%.1f", lampMul_);
        std::snprintf(flash, sizeof flash, "%.1f  %s", flashMul_, flashlightOn_ ? "ON" : "OFF");
        const Row rows[4] = {
            {"Moonlight", moon, "[Z / X]", parchment},
            {"Lamps & fire", lamps, "[C / V]", parchment},
            {"Flashlight", flash, "[B / N]  [F]", flashlightOn_ ? parchment : dim},
            {"Horror", horror_ ? "ON" : "OFF", "[H]", horror_ ? glm::vec4(0.95f, 0.15f, 0.1f, 1.0f) : parchment},
        };
        const std::string name = upper(scene().name), sub = "TAB / F1 / F2  change room";
        const float lineH = 12.0f * tsSmall, colGap = 14.0f * s;
        float labelW = 0.0f, valueW = 0.0f, keysW = 0.0f;
        for (const Row& r : rows) {
            labelW = std::max(labelW, hud_.textWidth(r.label, tsSmall));
            valueW = std::max(valueW, hud_.textWidth(r.value, tsSmall));
            keysW = std::max(keysW, hud_.textWidth(r.keys, tsSmall));
        }
        const float col1 = pad + labelW + colGap, col2 = col1 + valueW + colGap;
        const float contentW = std::max({hud_.textWidth(name, tsBig), hud_.textWidth(sub, tsSmall), col2 + keysW - pad});
        const float panelW = contentW + pad, panelH = 9.0f * tsBig + 0.25f * pad + 5.3f * lineH + 7.0f * tsSmall + pad;
        hud_.rect(pad * 0.5f, pad * 0.5f, panelW, panelH, panel);
        hud_.rect(pad * 0.5f, pad * 0.5f + panelH - 2.0f, panelW, 2.0f, withAlpha(accent, 0.6f));

        float y = pad;
        hud_.text(pad, y, name, tsBig, parchment);
        y += 9.0f * tsBig + 0.25f * pad;
        hud_.text(pad, y, sub, tsSmall, dim);
        y += lineH * 1.3f;
        for (const Row& r : rows) {
            hud_.text(pad, y, r.label, tsSmall, dim);
            hud_.text(col1, y, r.value, tsSmall, r.color);
            hud_.text(col2, y, r.keys, tsSmall, dim);
            y += lineH;
        }

        // Objective line, always visible; glows for a moment whenever it changes.
        const std::string objective = quest_.objective();
        const float glowAmt = objectiveFlash_ > 0.0f ? (objectiveFlash_ / 2.5f) * (0.6f + 0.4f * std::sin(time_ * 9.0f)) : 0.0f;
        const float oy = pad * 0.5f + panelH + std::round(8.0f * s);
        const float ow = std::max(panelW, hud_.textWidth(objective, tsSmall) + pad);
        const float oh = 14.0f * tsSmall + 0.5f * lineH + pad * 0.5f;
        hud_.rect(pad * 0.5f, oy, ow, oh, panel);
        if (glowAmt > 0.0f) hudFrame(hud_, pad * 0.5f, oy, ow, oh, std::max(1.0f, std::round(2.0f * s)), withAlpha(accent, glowAmt));
        hud_.rect(pad * 0.5f, oy, std::max(2.0f, std::round(3.0f * s)), oh, withAlpha(accent, 0.8f));
        hud_.text(pad, oy + pad * 0.3f, "OBJECTIVE", tsSmall, withAlpha(accent, 0.85f));
        hud_.text(pad, oy + pad * 0.3f + 7.0f * tsSmall + 0.5f * lineH, objective, tsSmall,
                  glm::mix(parchment, accent, glowAmt));
        leftHudRight = pad * 0.5f + std::max(panelW, ow);
    }

    // View buttons (top-right), always visible and clickable.
    {
        static const char* kLabels[4] = {"Top [1]", "Front [2]", "Side [3]", "Perspective [4]"};
        const float bh = std::round(36.0f * s), gap = std::round(8.0f * s);
        const float labelScale = std::clamp(std::floor(bh / 18.0f), 1.0f, 4.0f);  // what Hud::button uses
        float widths[4], total = gap * 3.0f;
        for (int i = 0; i < 4; ++i) {
            widths[i] = std::round(hud_.textWidth(kLabels[i], labelScale) + 30.0f * s);
            total += widths[i];
        }
        const bool active[4] = {camera_.mode == ViewMode::Top, camera_.mode == ViewMode::Front,
                                camera_.mode == ViewMode::Side, !camera_.isFixed()};
        float x = W - pad - total;
        int pressed = -1;
        const bool canClick = clickPending_ && !modalOpen();  // view changes are locked under modals
        for (int i = 0; i < 4; ++i) {
            if (hud_.button(x, pad, widths[i], bh, kLabels[i], active[i], mx, my, canClick)) pressed = i;
            x += widths[i] + gap;
        }
        if (pressed >= 0) pressViewButton(pressed);

        const std::string nav = camera_.navMode == ViewMode::FirstPerson ? "1st person" : "3rd person";
        const std::string sub = camera_.isFixed() ? "back to " + nav : nav;
        const float sw = hud_.textWidth(sub, tsSmall);
        hud_.text(W - pad - widths[3] * 0.5f - sw * 0.5f, pad + bh + 7.0f * s, sub, tsSmall,
                  camera_.isFixed() ? dim : accent);
        topHudBottom = pad + bh + 7.0f * s + 7.0f * tsSmall;
    }

    // Interaction prompt (bottom-centre).
    if (eligible_ >= 0 && phase_ == Phase::None && !modalOpen() && !mapOpen_) {
        const std::string key = "[E]  ";
        const std::string prompt = scene().objects[size_t(eligible_)].interaction.prompt;
        const float tw = hud_.textWidth(key + prompt, tsMed);
        const float x = (W - tw) * 0.5f, y = H - std::round(120.0f * s);
        hud_.rect(x - 18.0f * s, y - 12.0f * s, tw + 36.0f * s, 7.0f * tsMed + 24.0f * s, panel);
        hud_.rect(x - 18.0f * s, y + 7.0f * tsMed + 12.0f * s - 2.0f, tw + 36.0f * s, 2.0f, withAlpha(accent, 0.7f));
        const float kw = hud_.text(x, y, key, tsMed, accent);
        hud_.text(x + kw, y, prompt, tsMed, parchment);
    }

    // Small aiming dot in first person helps line up interactions.
    if (camera_.mode == ViewMode::FirstPerson && phase_ == Phase::None && !modalOpen() && !mapOpen_) {
        const float d = std::max(2.0f, std::round(3.0f * s));
        hud_.rect(W * 0.5f - d * 0.5f, H * 0.5f - d * 0.5f, d, d, {0.9f, 0.85f, 0.75f, 0.45f});
    }

    // Controls help (bottom-left), collapsible with I.
    {
        const float lineH = 12.0f * tsSmall;
        if (helpOpen_) {
            static const char* kHelp[][2] = {
                {"W A S D", "Move / strafe"},       {"Shift", "Run"},
                {"Arrow keys", "Turn, look up/down"}, {"E", "Interact (face the object)"},
                {"1  2  3", "Top / Front / Side view"}, {"4", "1st / 3rd person"},
                {"Mouse", "Click the view buttons"}, {"TAB  F1  F2", "Switch room"},
                {"F", "Flashlight on / off"},     {"Z / X", "Moonlight - / +"},
                {"C / V", "Lamps & fire - / +"},  {"B / N", "Flashlight - / +"},
                {"H", "Horror mode"},             {"M", "Map (once you find it)"},
                {"J", "Jump scare"},              {"I", "Hide controls"},
                {"Esc", "Close map / riddle, or quit"},
            };
            const int n = int(sizeof kHelp / sizeof kHelp[0]);
            float keyW = 0.0f, descW = 0.0f;
            for (const auto& line : kHelp) {
                keyW = std::max(keyW, hud_.textWidth(line[0], tsSmall));
                descW = std::max(descW, hud_.textWidth(line[1], tsSmall));
            }
            const float colKey = pad, colDesc = pad + keyW + 18.0f * s;
            const float panelW = colDesc - colKey + descW + 1.5f * pad;
            const float panelH = lineH * float(n + 1) + 1.5f * pad;
            const float y0 = H - pad * 0.5f - panelH;
            hud_.rect(pad * 0.5f, y0, panelW, panelH, panel);
            hud_.rect(pad * 0.5f, y0, panelW, 2.0f, withAlpha(accent, 0.6f));
            float y = y0 + pad * 0.75f;
            hud_.text(colKey, y, "CONTROLS", tsSmall, accent);
            y += lineH;
            for (const auto& line : kHelp) {
                hud_.text(colKey, y, line[0], tsSmall, parchment);
                hud_.text(colDesc, y, line[1], tsSmall, dim);
                y += lineH;
            }
        } else {
            hud_.text(pad, H - pad - 7.0f * tsSmall, "Press I for controls", tsSmall, dim);
        }
    }

    // Boundary hint.
    if (hintTimer_ > 0.0f) {
        const float a = std::min(1.0f, hintTimer_ / 0.5f);
        const std::string msg = "You can't go any further.";
        const float tw = hud_.textWidth(msg, tsMed);
        const float x = (W - tw) * 0.5f, y = H * 0.66f;
        hud_.rect(x - 14.0f * s, y - 10.0f * s, tw + 28.0f * s, 7.0f * tsMed + 20.0f * s, withAlpha(panel, a));
        hud_.text(x, y, msg, tsMed, withAlpha(glm::vec4(0.95f, 0.75f, 0.65f, 1.0f), a));
    }

    // Horror: flickering, jittering red message.
    if (horrorBlend_ > 0.02f) {
        static const char* kMessages[] = {"IT KNOWS YOU ARE HERE", "DON'T LOOK BEHIND YOU", "SHE NEVER LEFT"};
        const std::string msg = kMessages[int(time_ / 7.0f) % 3];
        float a = horrorBlend_ * (0.55f + 0.45f * noise1(time_ * 13.0f));
        if (hash1(std::floor(time_ * 10.0f)) < 0.18f) a *= 0.15f;
        const float ts = textScale(5.0f, s);
        const float tw = hud_.textWidth(msg, ts);
        const float x = (W - tw) * 0.5f + noise1(time_ * 23.0f) * 3.0f * s;
        const float y = H * 0.2f + noise1(time_ * 19.0f + 7.0f) * 2.0f * s;
        hud_.text(x + 3.0f * s, y + 1.0f * s, msg, ts, {0.35f, 0.0f, 0.0f, a * 0.5f});  // smeared ghost copy
        hud_.text(x, y, msg, ts, {0.88f, 0.05f, 0.03f, a});
    }

    // Quest panels.
    if (mapOpen_) {
        const int n = nurseryIndex();
        const float rx = leftHudRight + pad, ry = topHudBottom + pad * 0.5f;
        drawNurseryMap(hud_, slots_[size_t(n)].scene, !quest_.won(), current_ == n, player_.position, player_.yawDeg,
                       time_, s, {rx, ry, W - pad - rx, H - pad - ry});
    }
    if (toastTimer_ > 0.0f && !toast_.empty()) {
        const float a = std::min({1.0f, (toastLength_ - toastTimer_) / 0.25f, toastTimer_ / 0.6f});
        const float tw = hud_.textWidth(toast_, tsMed);
        const float x = (W - tw) * 0.5f, y = std::round(H * 0.3f);
        hud_.rect(x - 20.0f * s, y - 13.0f * s, tw + 40.0f * s, 7.0f * tsMed + 26.0f * s, withAlpha(panel, 1.2f * a));
        hud_.rect(x - 20.0f * s, y - 13.0f * s, tw + 40.0f * s, 2.0f, withAlpha(accent, 0.7f * a));
        hud_.text(x, y, toast_, tsMed, withAlpha(parchment, a));
    }
    if (riddleOpen_ && !scareActive()) {
        const int slotPicked = drawRiddle(hud_, quest_, riddleT_, s, fbW_, fbH_, mx, my, clickPending_);
        if (slotPicked >= 0) answerRiddle(slotPicked);
    }
    if (winScreen_) drawWinScreen(hud_, quest_, winT_, s, fbW_, fbH_);
    clickPending_ = false;

    // Title card on start.
    if (titleTimer_ < 4.2f) {
        const float a = titleTimer_ < 2.6f ? 1.0f : 1.0f - smooth01((titleTimer_ - 2.6f) / 1.6f);
        hud_.rect(0, 0, W, H, {0.0f, 0.0f, 0.0f, 0.88f * a});
        const std::string title = "HAUNTED MANOR";
        const float ts = textScale(9.0f, s);
        const float tw = hud_.textWidth(title, ts);
        const float y = H * 0.36f;
        hud_.text((W - tw) * 0.5f, y, title, ts, {0.80f, 0.07f, 0.05f, a});
        const float ruleW = tw * 0.8f;
        hud_.rect((W - ruleW) * 0.5f, y + 9.0f * ts, ruleW, std::max(1.0f, std::round(2.0f * s)), withAlpha(accent, 0.7f * a));
        const std::string sub = "An abandoned Victorian manor. Something never left.";
        const float sw = hud_.textWidth(sub, tsSmall + 1.0f);
        hud_.text((W - sw) * 0.5f, y + 9.0f * ts + 18.0f * s, sub, tsSmall + 1.0f, withAlpha(parchment, a));
        const std::string hint = "Press I for controls";
        const float hw = hud_.textWidth(hint, tsSmall);
        hud_.text((W - hw) * 0.5f, y + 9.0f * ts + 60.0f * s, hint, tsSmall, withAlpha(dim, a));
    }

    // Scene switch fade.
    if (fade_ > 0.0f) hud_.rect(0, 0, W, H, {0.0f, 0.0f, 0.0f, fade_});

    hud_.end();
}

// =============================================================================================
// Input
// =============================================================================================

void Game::onKey(int key, int action, int) {
    if (action == GLFW_RELEASE) return;
    if (scareActive()) return;  // the jump scare locks all input

    if (winScreen_) {  // modal: only ENTER (or Esc) dismisses it
        if (action == GLFW_PRESS &&
            (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER || key == GLFW_KEY_ESCAPE)) {
            winScreen_ = false;
            Audio::play("click");
        }
        return;
    }

    // Dimmers step by 0.1 and auto-repeat while held.
    auto step = [](float& v, float d) { v = std::clamp(std::round((v + d) * 10.0f) / 10.0f, 0.0f, 3.0f); };
    switch (key) {
    case GLFW_KEY_Z: step(moonMul_, -0.1f); return;
    case GLFW_KEY_X: step(moonMul_, +0.1f); return;
    case GLFW_KEY_C: step(lampMul_, -0.1f); return;
    case GLFW_KEY_V: step(lampMul_, +0.1f); return;
    case GLFW_KEY_B: step(flashMul_, -0.1f); return;
    case GLFW_KEY_N: step(flashMul_, +0.1f); return;
    default: break;
    }
    if (action != GLFW_PRESS) return;

    if (riddleOpen_) {  // modal: 1-4 answer, Esc puts the key down; movement, views, E, M and TAB are locked
        switch (key) {
        case GLFW_KEY_1: case GLFW_KEY_KP_1: answerRiddle(0); return;
        case GLFW_KEY_2: case GLFW_KEY_KP_2: answerRiddle(1); return;
        case GLFW_KEY_3: case GLFW_KEY_KP_3: answerRiddle(2); return;
        case GLFW_KEY_4: case GLFW_KEY_KP_4: answerRiddle(3); return;
        case GLFW_KEY_ESCAPE:
            riddleOpen_ = false;
            Audio::play("click");
            return;
        case GLFW_KEY_F: case GLFW_KEY_H: case GLFW_KEY_I: case GLFW_KEY_J: break;
        default: return;
        }
    }

    switch (key) {
    case GLFW_KEY_ESCAPE:
        if (mapOpen_)
            mapOpen_ = false;
        else
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        break;
    case GLFW_KEY_M: toggleMap(); break;
    case GLFW_KEY_J: startScare(); break;
    case GLFW_KEY_1: case GLFW_KEY_KP_1: pressViewButton(0); break;
    case GLFW_KEY_2: case GLFW_KEY_KP_2: pressViewButton(1); break;
    case GLFW_KEY_3: case GLFW_KEY_KP_3: pressViewButton(2); break;
    case GLFW_KEY_4: case GLFW_KEY_KP_4: pressViewButton(3); break;
    case GLFW_KEY_E:
        if (phase_ == Phase::None) {
            const int idx = findEligible();
            if (idx >= 0) startInteraction(idx);  // nothing in range and in view: do nothing
        }
        break;
    case GLFW_KEY_TAB: requestScene(1 - (pendingScene_ >= 0 ? pendingScene_ : current_)); break;
    case GLFW_KEY_F1: requestScene(0); break;
    case GLFW_KEY_F2: requestScene(1); break;
    case GLFW_KEY_F:
        flashlightOn_ = !flashlightOn_;
        Audio::play("flashlight");
        break;
    case GLFW_KEY_H: setHorror(!horror_); break;
    case GLFW_KEY_I: helpOpen_ = !helpOpen_; break;
    default: break;
    }
}

void Game::onMouseButton(int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) clickPending_ = true;
}
