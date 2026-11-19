#include "Scene.h"
#include "Model.h"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

// ---------------------------------------------------------------------------------------------
// SceneObject
// ---------------------------------------------------------------------------------------------

namespace {

// Euler rotation applied Y, then X, then Z (degrees).
glm::mat4 rotationYXZ(const glm::vec3& deg) {
    glm::mat4 m = glm::rotate(glm::mat4(1.0f), glm::radians(deg.y), glm::vec3(0.0f, 1.0f, 0.0f));
    m = glm::rotate(m, glm::radians(deg.x), glm::vec3(1.0f, 0.0f, 0.0f));
    return glm::rotate(m, glm::radians(deg.z), glm::vec3(0.0f, 0.0f, 1.0f));
}

} // namespace

glm::mat4 SceneObject::modelMatrix() const {
    glm::mat4 m = glm::translate(glm::mat4(1.0f), position + animTranslate);
    m *= rotationYXZ(rotationDeg);
    m = glm::scale(m, scale);
    m = glm::translate(m, pivot);
    m *= rotationYXZ(animRotDeg);
    m = glm::scale(m, glm::vec3(animScale));
    return glm::translate(m, -pivot);
}

glm::vec3 SceneObject::worldCenter() const {
    if (!model) return position;
    // An affine map sends the box centre to the centre of the transformed box's AABB.
    glm::mat4 m = glm::translate(glm::mat4(1.0f), position) * rotationYXZ(rotationDeg) *
                  glm::scale(glm::mat4(1.0f), scale);
    return glm::vec3(m * glm::vec4(0.5f * (model->boundsMin + model->boundsMax), 1.0f));
}

void SceneObject::resetAnim() {
    animTranslate = glm::vec3(0.0f);
    animRotDeg = glm::vec3(0.0f);
    animScale = 1.0f;
    animTint = glm::vec3(1.0f);
    animEmissive = glm::vec3(0.0f);
}

// ---------------------------------------------------------------------------------------------
// Scene building helpers
// ---------------------------------------------------------------------------------------------

namespace {

// Model conventions (docs/CONTRACT.md): front faces +Z, so yaw 0 = faces +Z, 90 = faces +X,
// -90 = faces -X, 180 = faces -Z. [wall] models have their back at local z = 0.
// The returned reference is only valid until the next place() call.
SceneObject& place(Scene& s, ModelCache& cache, const char* id, glm::vec3 pos, float yawDeg,
                   bool solid, HorrorAnim horror = HorrorAnim::None) {
    SceneObject o;
    o.name = id;
    o.model = cache.get(id);
    o.position = pos;
    o.rotationDeg = glm::vec3(0.0f, yawDeg, 0.0f);
    o.solid = solid;
    o.horror = horror;
    s.objects.push_back(o);
    return s.objects.back();
}

Interaction& interact(SceneObject& o, const char* prompt, float range, float viewAngleDeg,
                      unsigned anims, const char* sound, float duration) {
    Interaction& it = o.interaction;
    it.enabled = true;
    it.prompt = prompt;
    it.range = range;
    it.viewAngleDeg = viewAngleDeg;
    it.anims = anims;
    it.sound = sound;
    it.duration = duration;
    return it;
}

// Interaction camera: `dist` metres in front of the object's front face (its +Z side after yaw),
// `lift` metres above the look-at point, which is the object's centre + `aim`.
// With a 60 deg vertical FOV an object of height h fills ~87% of the view at
// dist = h * 1.15 / (2 * tan 30deg) ~= h.
void frameCamera(SceneObject& o, float dist, float lift, glm::vec3 aim = glm::vec3(0.0f)) {
    float yaw = glm::radians(o.rotationDeg.y);
    glm::vec3 forward(std::sin(yaw), 0.0f, std::cos(yaw));
    float halfDepth = 0.5f * (o.model->boundsMax.z - o.model->boundsMin.z) * o.scale.z;
    o.interaction.camTarget = o.worldCenter() + aim;
    o.interaction.camPos = o.interaction.camTarget + forward * (halfDepth + dist) + glm::vec3(0.0f, lift, 0.0f);
}

PointLight pointLight(glm::vec3 pos, glm::vec3 color, float intensity, float range, float flicker) {
    PointLight p;
    p.position = pos;
    p.color = color;
    p.intensity = intensity;
    p.range = range;
    p.flicker = flicker;
    return p;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// The Library — 12 x 4 x 10 m. Fireplace centred on the north (-Z) wall flanked by a marble bust
// and the grandfather clock; reading nook (rug, wingback armchair angled at the fire, side table
// with candelabra) in front of it; bookcase on the west wall; writing desk under the portrait on
// the east wall with the globe beside it (quest map on the desk); chandelier at the ceiling centre.
// Door side (+Z) is open.
// ---------------------------------------------------------------------------------------------

Scene buildLibrary(ModelCache& cache) {
    Scene s;
    s.name = "The Library";
    s.room.size = glm::vec3(12.0f, 4.0f, 10.0f);
    s.room.wallTex = "wallpaper_damask";
    s.room.floorTex = "wood_floor";
    s.room.ceilingTex = "plaster";
    s.ambient = glm::vec3(0.065f, 0.055f, 0.050f);
    s.fogColor = glm::vec3(0.020f, 0.016f, 0.014f);
    s.fogDensity = 0.045f;
    s.boundsMin = glm::vec2(-5.65f, -4.65f);
    s.boundsMax = glm::vec2(5.65f, 4.65f);
    s.spawnPos = glm::vec3(0.0f, 0.0f, 3.6f);
    s.spawnYawDeg = 0.0f;                       // facing the fireplace
    s.ambientSound = "ambient_library";

    // Lights. No windows here, so the moonlight is only a faint cold fill.
    s.lights.dir.direction = glm::vec3(-0.3f, -1.0f, 0.25f);
    s.lights.dir.color = glm::vec3(0.45f, 0.55f, 0.85f);
    s.lights.dir.intensity = 0.15f;
    enum { L_FIRE, L_CANDELABRA, L_CHANDELIER };
    s.lights.points = {
        pointLight({0.0f, 0.40f, -4.60f}, {1.0f, 0.50f, 0.18f}, 2.4f, 8.0f, 0.6f),    // embers/flames, mouth of the firebox
        pointLight({-1.8f, 1.22f, -2.15f}, {1.0f, 0.68f, 0.32f}, 0.5f, 4.5f, 0.3f),   // candelabra flames
        pointLight({0.0f, 3.54f, 0.0f}, {1.0f, 0.72f, 0.42f}, 0.45f, 7.0f, 0.15f),    // chandelier candle ring
    };

    // North wall: fireplace, bust, clock.
    place(s, cache, "lib_fireplace", {0.0f, 0.0f, -5.0f}, 0.0f, true);
    place(s, cache, "lib_bust", {-2.8f, 0.0f, -4.55f}, 0.0f, true, HorrorAnim::StareAtPlayer);
    {
        auto& clock = place(s, cache, "lib_clock", {2.8f, 0.0f, -4.78f}, 0.0f, true, HorrorAnim::Pulse);
        clock.pivot = glm::vec3(0.0f);             // base centre
        auto& it = interact(clock, "Wind the clock", 2.4f, 30.0f, ANIM_SCALE | ANIM_COLOR, "chime", 5.0f);
        it.scaleTo = 1.06f;
        it.colorTo = glm::vec3(1.0f, 0.35f, 0.30f);
        it.emissiveTo = glm::vec3(0.12f, 0.0f, 0.0f);
        frameCamera(clock, 2.6f, 0.2f);
    }

    // Reading nook in front of the fire.
    place(s, cache, "lib_rug", {0.0f, 0.003f, -2.5f}, 0.0f, false);
    place(s, cache, "lib_armchair", {-0.75f, 0.0f, -2.45f}, 160.0f, true, HorrorAnim::Rock);   // angled at the fire
    place(s, cache, "lib_side_table", {-1.8f, 0.0f, -2.15f}, 0.0f, true, HorrorAnim::Rattle);
    {
        auto& candelabra = place(s, cache, "lib_candelabra", {-1.8f, 0.7f, -2.15f}, 0.0f, false, HorrorAnim::Rattle);
        auto& it = interact(candelabra, "Light the candles", 1.7f, 45.0f, ANIM_COLOR, "light_on", 4.0f);
        it.emissiveTo = glm::vec3(0.45f, 0.25f, 0.07f);   // uEmissive is added flat: keep it a warm glow
        it.linkedPointLight = L_CANDELABRA;
        it.lightBoost = 2.5f;
        frameCamera(candelabra, 0.6f, 0.15f);
    }

    // West wall: bookcase that slides along the wall to reveal a hidden recess.
    {
        auto& shelf = place(s, cache, "lib_bookshelf", {-5.755f, 0.0f, -1.4f}, 90.0f, true, HorrorAnim::Rattle);
        auto& it = interact(shelf, "Pull the odd book", 3.0f, 35.0f, ANIM_TRANSLATE, "creak", 5.0f);
        it.translateBy = glm::vec3(0.0f, 0.0f, 1.6f);
        frameCamera(shelf, 2.6f, 0.1f, it.translateBy * 0.5f);
    }

    // East wall: desk under the portrait, globe beside it.
    place(s, cache, "lib_desk", {5.53f, 0.0f, 0.4f}, -90.0f, true, HorrorAnim::Rattle);
    {
        // Quest map (hidden by the game until the four Library interactions are done). Lies on the loose
        // papers (desk wood top y = 0.725, papers 3 mm) between the pen and the inkwell, slightly askew.
        auto& map = place(s, cache, "lib_map", {5.50f, 0.728f, 0.32f}, -98.0f, false);
        map.pivot = glm::vec3(0.0f, 0.03f, 0.0f);         // centre of the map
        auto& it = interact(map, "Take the map", 1.8f, 40.0f, ANIM_TRANSLATE | ANIM_SCALE | ANIM_COLOR, "interact", 4.0f);
        it.translateBy = glm::vec3(0.0f, 0.25f, 0.0f);
        it.scaleTo = 1.4f;
        it.emissiveTo = glm::vec3(0.35f, 0.22f, 0.08f);
        frameCamera(map, 0.3f, 0.42f, it.translateBy);    // looks down onto the lifted map
    }
    {
        auto& portrait = place(s, cache, "lib_portrait", {6.0f, 1.0f, 0.4f}, -90.0f, false, HorrorAnim::Sway);
        portrait.pivot = glm::vec3(0.0f, 1.3f, 0.04f);   // top centre of the frame (hanging nail)
    }
    {
        auto& globe = place(s, cache, "lib_globe", {4.2f, 0.0f, 2.3f}, -90.0f, true, HorrorAnim::Spin);
        auto& it = interact(globe, "Spin the globe", 2.0f, 40.0f, ANIM_ROTATE, "creak", 4.0f);
        it.rotAxis = glm::vec3(0.0f, 1.0f, 0.0f);
        it.rotSpeedDeg = 240.0f;
        frameCamera(globe, 1.15f, 0.25f, {0.0f, 0.1f, 0.0f});
    }

    // Ceiling centre ([hang]: origin at the top attachment point).
    place(s, cache, "lib_chandelier", {0.0f, 4.0f, 0.0f}, 0.0f, false, HorrorAnim::Sway);

    return s;
}

// ---------------------------------------------------------------------------------------------
// The Nursery Bedroom — 10 x 3.5 x 9 m. Four-poster bed against the north (-Z) wall with the
// nightstand + oil lamp beside it and the toy chest (doll sitting on it) at its foot; window on
// the west wall with the rocking chair on the round rug in front of it; dresser (music box on top)
// and wardrobe on the east wall; the quest key lies half under the bed's window side. Door side (+Z) is open.
// ---------------------------------------------------------------------------------------------

Scene buildBedroom(ModelCache& cache) {
    Scene s;
    s.name = "The Nursery Bedroom";
    s.room.size = glm::vec3(10.0f, 3.5f, 9.0f);
    s.room.wallTex = "wallpaper_faded";
    s.room.floorTex = "wood_floor";
    s.room.ceilingTex = "plaster";
    s.ambient = glm::vec3(0.050f, 0.058f, 0.085f);
    s.fogColor = glm::vec3(0.015f, 0.020f, 0.035f);
    s.fogDensity = 0.05f;
    s.boundsMin = glm::vec2(-4.65f, -4.15f);
    s.boundsMax = glm::vec2(4.65f, 4.15f);
    s.spawnPos = glm::vec3(0.8f, 0.0f, 3.3f);
    s.spawnYawDeg = 5.0f;                       // facing the bed and the doll
    s.ambientSound = "ambient_bedroom";

    // Moonlight falls in through the west window (travels toward +X).
    s.lights.dir.direction = glm::vec3(0.6f, -0.7f, 0.25f);
    s.lights.dir.color = glm::vec3(0.50f, 0.60f, 0.95f);
    s.lights.dir.intensity = 0.32f;
    enum { L_LAMP, L_WINDOW, L_DRESSER };
    s.lights.points = {
        pointLight({1.3f, 0.955f, -4.25f}, {1.0f, 0.66f, 0.32f}, 0.6f, 5.0f, 0.25f),   // oil lamp flame
        pointLight({-4.6f, 1.7f, -1.2f}, {0.45f, 0.55f, 0.95f}, 0.35f, 4.5f, 0.0f),    // moonlit window panes
        pointLight({4.25f, 1.4f, -2.2f}, {1.0f, 0.60f, 0.30f}, 0.2f, 2.5f, 0.5f),      // candle-like glow in the dresser mirror
    };

    // North wall: bed, nightstand with oil lamp, toy chest + doll at the foot of the bed.
    place(s, cache, "bed_bed", {0.0f, 0.0f, -3.38f}, 0.0f, true, HorrorAnim::Rattle);
    {
        // Quest key: on the floor on the dark window side of the bed, half under the side rail (the rail's
        // underside is at y = 0.28, so the key stays visible from a low camera). Only usable once the map is taken.
        auto& key = place(s, cache, "bed_key", {-0.9f, 0.0f, -3.7f}, -70.0f, false);
        key.pivot = glm::vec3(0.0f, 0.01f, 0.0f);         // centre of the key
        auto& it = interact(key, "Pick up the strange key", 1.6f, 45.0f, ANIM_TRANSLATE | ANIM_ROTATE | ANIM_COLOR,
                            "pickup", 4.5f);
        it.translateBy = glm::vec3(-0.2f, 0.3f, 0.0f);   // rises out from under the rail
        it.rotAxis = glm::vec3(0.0f, 1.0f, 0.0f);
        it.rotSpeedDeg = 200.0f;
        it.emissiveTo = glm::vec3(0.3f, 0.21f, 0.06f);  // stays below white so the brass detail reads
        frameCamera(key, 0.13f, 0.07f, it.translateBy);   // very close: the key is only 16 cm long
    }
    place(s, cache, "bed_nightstand", {1.3f, 0.0f, -4.27f}, 0.0f, true);
    {
        auto& lamp = place(s, cache, "bed_lamp", {1.3f, 0.62f, -4.25f}, 0.0f, false);
        auto& it = interact(lamp, "Turn up the lamp", 1.6f, 45.0f, ANIM_COLOR, "light_on", 4.0f);
        it.emissiveTo = glm::vec3(0.40f, 0.22f, 0.06f);
        it.linkedPointLight = L_LAMP;
        it.lightBoost = 2.5f;
        frameCamera(lamp, 0.55f, 0.12f);
    }
    place(s, cache, "bed_toy_chest", {0.0f, 0.0f, -1.95f}, 0.0f, true, HorrorAnim::Rattle);
    {
        auto& doll = place(s, cache, "bed_doll", {0.18f, 0.541f, -1.95f}, 0.0f, false, HorrorAnim::StareAtPlayer);
        doll.pivot = glm::vec3(0.0f, 0.21f, 0.0f);       // centre of the doll
        auto& it = interact(doll, "Pick up the doll", 2.0f, 50.0f, ANIM_TRANSLATE | ANIM_SWING, "whisper", 5.0f);
        it.translateBy = glm::vec3(0.0f, 0.35f, 0.0f);   // floats up off the chest
        it.rotAxis = glm::vec3(0.0f, 0.0f, 1.0f);        // side-to-side head-tilt sway
        it.swingAmpDeg = 15.0f;
        it.swingFreq = 0.5f;
        frameCamera(doll, 0.6f, 0.1f, {0.0f, 0.28f, 0.0f});
    }

    // West wall: window, round rug, rocking chair beside the window angled into the room.
    {
        auto& window = place(s, cache, "bed_window", {-5.0f, 0.6f, -1.2f}, 90.0f, false, HorrorAnim::Sway);
        window.pivot = glm::vec3(0.0f, 2.2f, 0.1f);      // curtains sway from the rail
    }
    place(s, cache, "bed_rug", {-3.4f, 0.003f, -0.4f}, 0.0f, false);
    {
        auto& chair = place(s, cache, "bed_rocking_chair", {-3.7f, 0.0f, -0.6f}, 65.0f, true, HorrorAnim::Rock);
        chair.pivot = glm::vec3(0.0f);                   // rockers' floor contact
        auto& it = interact(chair, "Rock the chair", 2.2f, 40.0f, ANIM_SWING, "creak", 5.0f);
        it.rotAxis = glm::vec3(1.0f, 0.0f, 0.0f);
        it.swingAmpDeg = 12.0f;
        it.swingFreq = 0.8f;
        frameCamera(chair, 1.3f, 0.3f);
    }

    // East wall: dresser with the music box on its top, wardrobe.
    place(s, cache, "bed_dresser", {4.73f, 0.0f, -2.2f}, -90.0f, true, HorrorAnim::Rattle);
    {
        // On the drawer top (y = 0.90 in bed_dresser.obj), front right, clear of the mirror posts.
        auto& box = place(s, cache, "bed_music_box", {4.61f, 0.9f, -1.9f}, -90.0f, false, HorrorAnim::Spin);
        box.pivot = glm::vec3(0.0f, 0.08f, 0.0f);         // centre of the box
        auto& it = interact(box, "Open the music box", 1.8f, 35.0f, ANIM_ROTATE | ANIM_SCALE, "music_box", 6.0f);
        it.rotAxis = glm::vec3(0.0f, 1.0f, 0.0f);
        it.rotSpeedDeg = 90.0f;
        it.scaleTo = 1.25f;
        frameCamera(box, 0.32f, 0.18f);
    }
    place(s, cache, "bed_wardrobe", {4.67f, 0.0f, 1.3f}, -90.0f, true, HorrorAnim::Rattle);

    return s;
}
