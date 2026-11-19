#version 460 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;

out vec4 FragColor;

#define MAX_POINT_LIGHTS 8

struct DirLight   { vec3 direction; vec3 color; };
struct PointLight { vec3 position; vec3 color; float range; };
struct SpotLight  { vec3 position; vec3 direction; vec3 color; float innerCos; float outerCos; float range; };

uniform sampler2D uDiffuse;
uniform vec3 uBaseColor;      // MTL Kd
uniform vec3 uMeshEmissive;   // MTL Ke
uniform vec3 uTint;
uniform vec3 uEmissive;

uniform vec3 uViewPos;
uniform vec3 uAmbient;
uniform DirLight uDirLight;
uniform int uPointCount;
uniform PointLight uPoints[MAX_POINT_LIGHTS];
uniform SpotLight uSpot;

uniform float uTime;
uniform float uHorror;
uniform vec3 uFogColor;       // display (sRGB) colour, so it matches glClearColor
uniform float uFogDensity;
uniform vec2 uScreenSize;

const float SHININESS = 40.0;

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Smooth window that reaches exactly zero at `range`, times inverse-square-ish falloff.
float rangeAttenuation(float d, float range, float falloff) {
    float x = d / max(range, 1e-4);
    float window = clamp(1.0 - x * x * x * x, 0.0, 1.0);
    return window * window / (1.0 + falloff * d * d);
}

// Blinn-Phong for one light. L points from the surface to the light; radiance is already attenuated.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float specStrength) {
    float NdotL = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), SHININESS) * specStrength;
    return radiance * NdotL * (albedo + vec3(spec));
}

// Flashlight reflector pattern: bright hot spot plus faint rings toward the rim.
float flashlightCookie(float cosTheta, float outerCos) {
    float r = sqrt(clamp((1.0 - cosTheta) / max(1.0 - outerCos, 1e-4), 0.0, 1.0));   // 0 axis .. 1 rim
    float hotspot = 1.0 + 0.6 * exp(-r * r * 10.0);
    float rings = 1.0 - 0.08 * smoothstep(0.3, 1.0, r) * (0.5 + 0.5 * sin(r * 38.0));
    return hotspot * rings;
}

// "Lub-dub" at ~66 bpm, peaks at 1.
float heartbeat(float t) {
    float b = fract(t * 1.1);
    return exp(-b * 14.0) + 0.6 * step(0.2, b) * exp(-(b - 0.2) * 14.0);
}

// Linear below t, smooth roll-off toward 1 above it, so flames and hot spots don't clip hard.
vec3 softClip(vec3 c) {
    const float t = 0.6;
    vec3 over = max(c - t, 0.0);
    return min(c, vec3(t)) + (1.0 - t) * (1.0 - exp(-over / (1.0 - t)));
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec4 tex = texture(uDiffuse, vUV);
    if (tex.a < 0.1) discard;
    vec3 albedo = tex.rgb * uBaseColor * uTint;
    // No gloss maps: bright materials (brass, marble, white paint) read shinier than dark wood or velvet.
    float specStrength = mix(0.05, 0.35, smoothstep(0.15, 0.7, luminance(tex.rgb)));

    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;   // thin single-sided geometry is lit from both sides
    vec3 toEye = uViewPos - vWorldPos;
    float viewDist = length(toEye);
    vec3 V = toEye / max(viewDist, 1e-4);

    // Hemisphere ambient: surfaces facing the floor get less bounce light than those facing the ceiling.
    vec3 color = uAmbient * albedo * mix(0.55, 1.0, N.y * 0.5 + 0.5);

    color += blinnPhong(N, V, normalize(-uDirLight.direction), uDirLight.color, albedo, specStrength);

    for (int i = 0; i < min(uPointCount, MAX_POINT_LIGHTS); ++i) {
        vec3 Lv = uPoints[i].position - vWorldPos;
        float d = length(Lv);
        vec3 radiance = uPoints[i].color * rangeAttenuation(d, uPoints[i].range, 1.0);
        color += blinnPhong(N, V, Lv / max(d, 1e-4), radiance, albedo, specStrength);
    }

    if (any(greaterThan(uSpot.color, vec3(0.0)))) {
        vec3 Lv = uSpot.position - vWorldPos;
        float d = length(Lv);
        vec3 L = Lv / max(d, 1e-4);
        float cosTheta = dot(-L, normalize(uSpot.direction));
        float cone = smoothstep(uSpot.outerCos, max(uSpot.innerCos, uSpot.outerCos + 1e-4), cosTheta);
        // Gentler distance falloff than lamps so the beam reaches the far wall.
        float atten = rangeAttenuation(d, uSpot.range, 0.15);
        vec3 radiance = uSpot.color * atten * cone * flashlightCookie(cosTheta, uSpot.outerCos);
        color += blinnPhong(N, V, L, radiance, albedo, specStrength);
    }

    // Emission follows the diffuse texture so flames and sky panes keep their detail.
    color += uMeshEmissive * tex.rgb + uEmissive;

    // Exponential-squared fog; horror makes it thicker and blood-dark.
    vec3 fogColor = mix(pow(max(uFogColor, 0.0), vec3(2.2)), vec3(0.025, 0.0, 0.002), uHorror);
    float fogDensity = uFogDensity * (1.0 + 0.6 * uHorror);
    float fogAmount = 1.0 - exp(-(fogDensity * viewDist) * (fogDensity * viewDist));
    color = mix(color, fogColor, fogAmount);

    // Horror grade: drain colour, push toward red, darken a little.
    color = mix(color, vec3(luminance(color)), 0.4 * uHorror);
    color *= mix(vec3(1.0), vec3(1.35, 0.72, 0.68), uHorror);
    color *= 1.0 - 0.08 * uHorror;

    color = softClip(max(color, 0.0));

    // Vignette: always a little, and a pulsing blood-red tunnel under horror.
    vec2 res = max(uScreenSize, vec2(1.0));
    float r = length((gl_FragCoord.xy - 0.5 * res) / res.y);   // aspect-correct, 0.5 = top/bottom edge
    float pulse = heartbeat(uTime) * uHorror;
    float edge = smoothstep(mix(0.55, 0.22, uHorror) - 0.08 * pulse, mix(1.05, 0.78, uHorror), r);
    color = mix(color, color * vec3(0.55, 0.06, 0.05), edge * uHorror);
    color *= 1.0 - mix(0.4, 0.8, uHorror) * edge;

    color = pow(color, vec3(1.0 / 2.2));

    // Film grain (also dithers away banding in dark gradients); fades out in highlights.
    float n = hash12(gl_FragCoord.xy + fract(uTime * 7.31) * 517.0) - 0.5;
    color += n * mix(0.035, 0.06, uHorror) * (1.0 - smoothstep(0.4, 1.0, luminance(color)));

    FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
