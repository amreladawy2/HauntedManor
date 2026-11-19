#include "Lighting.h"
#include "Shader.h"

#include <algorithm>
#include <cmath>
#include <string>

void uploadLights(const Shader& s, const LightRig& rig) {
    s.setVec3("uDirLight.direction", glm::normalize(rig.dir.direction));
    s.setVec3("uDirLight.color", rig.dir.color * rig.dir.intensity);

    const int count = std::min<int>(static_cast<int>(rig.points.size()), MAX_POINT_LIGHTS);
    s.setInt("uPointCount", count);
    for (int i = 0; i < count; ++i) {
        const PointLight& p = rig.points[i];
        const std::string base = "uPoints[" + std::to_string(i) + "].";
        s.setVec3(base + "position", p.position);
        s.setVec3(base + "color", p.color * p.intensity);
        s.setFloat(base + "range", p.range);
    }

    const SpotLight& sp = rig.spot;
    s.setVec3("uSpot.position", sp.position);
    s.setVec3("uSpot.direction", glm::normalize(sp.direction));
    s.setVec3("uSpot.color", sp.on ? sp.color * sp.intensity : glm::vec3(0.0f));
    s.setFloat("uSpot.innerCos", std::cos(glm::radians(sp.innerDeg)));
    s.setFloat("uSpot.outerCos", std::cos(glm::radians(sp.outerDeg)));
    s.setFloat("uSpot.range", sp.range);
}
