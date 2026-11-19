# Haunted Manor — shared contract

Project root: `C:\Users\LENOVO\Desktop\java codeing\HauntedManor` (path contains a space — always quote it).
Theme: an abandoned, haunted Victorian manor. Two rooms: **The Library** and **The Nursery Bedroom**.

Stack: C++20, OpenGL **4.6 core** (`#version 460 core`), GLFW, GLEW (`glewExperimental = GL_TRUE`), GLM, Assimp, stb (`<stb/stb_image.h>`, `<stb/stb_easy_font.h>`).
Toolchain: MSYS2 UCRT64 `g++` 16, CMake 4 + Ninja (`/c/msys64/ucrt64/bin`). Headers live in `/c/msys64/ucrt64/include`.
Python for asset generators: `py -3.10` (has Pillow; no numpy).

Single-file compile check (works before the full project links):
```
g++ -std=c++20 -fsyntax-only -I"src" src/Foo.cpp
```
Only the integrator builds/links the whole project. Do not edit files owned by another agent; headers in `src/*.h`
written by the integrator are the contract — if one is genuinely unworkable, note it in your final report instead of editing.

## Ownership
| Owner | Files |
|---|---|
| Integrator | `src/*.h` headers listed here, `docs/` |
| Core agent | `CMakeLists.txt`, `src/Paths.cpp`, `src/Shader.cpp`, `src/Texture.cpp`, `src/Model.cpp`, `src/Primitives.cpp`, `src/Lighting.cpp`, `src/stb_impl.cpp`, `shaders/scene.vert`, `shaders/scene.frag` |
| HUD/Audio agent | `src/Hud.cpp`, `src/Audio.cpp`, `tools/gen_sounds.py`, `assets/sounds/*.wav` |
| Texture agent | `tools/gen_textures.py`, `assets/textures/*.png` |
| Model agent | `tools/gen_models.py`, `assets/models/*.obj|*.mtl` |
| Scenes agent | `src/Scene.cpp` (SceneObject methods + buildLibrary + buildBedroom) |
| Gameplay agent | `src/Camera.h/.cpp`, `src/Player.h/.cpp`, `src/Game.h/.cpp`, `src/main.cpp` |

## Scene shader uniform contract (`shaders/scene.vert` / `scene.frag`)
Vertex attributes: `layout(location=0) vec3 aPos; 1 vec3 aNormal; 2 vec2 aUV`.

| Uniform | Type | Set by | Meaning |
|---|---|---|---|
| `uModel`, `uView`, `uProj` | mat4 | game | transforms |
| `uNormalMat` | mat3 | game | `transpose(inverse(mat3(model)))` |
| `uDiffuse` | sampler2D | game (=0 once) | diffuse texture, unit 0 |
| `uBaseColor` | vec3 | Mesh::draw | MTL Kd, multiplies texture |
| `uMeshEmissive` | vec3 | Mesh::draw | MTL Ke |
| `uTint` | vec3 | game | per-object tint (multiplies albedo) |
| `uEmissive` | vec3 | game | per-object extra emissive (added) |
| `uViewPos` | vec3 | game | camera eye |
| `uAmbient` | vec3 | game | ambient light colour |
| `uDirLight.direction`, `uDirLight.color` | vec3 | uploadLights | colour pre-multiplied by intensity |
| `uPointCount` | int | uploadLights | number of point lights (<= 8) |
| `uPoints[i].position`, `.color`, `.range` | vec3/vec3/float | uploadLights | colour pre-multiplied |
| `uSpot.position`, `.direction`, `.color`, `.innerCos`, `.outerCos`, `.range` | | uploadLights | colour = 0 when off |
| `uTime` | float | game | seconds |
| `uHorror` | float | game | 0..1 horror blend |
| `uFogColor`, `uFogDensity` | vec3, float | game | exponential-squared fog |
| `uScreenSize` | vec2 | game | framebuffer size (vignette) |

Lighting: Blinn-Phong per fragment, smooth range attenuation `pow(clamp(1 - (d/range)^4, 0, 1), 2) / (1 + d*d)` style,
spotlight smooth cone between outer/inner cos. Output gamma-corrected (pow 1/2.2). Horror: desaturate ~40%, red shift,
pulsing vignette, fog darker; all scaled by `uHorror`.

## Textures (`assets/textures/<id>.png`, 512x512 RGB, seamless when tiled)
wood_floor, wallpaper_damask, wallpaper_faded, plaster, wood_dark, wood_light, wood_painted_white, stone, brick_soot,
velvet_red, linen, quilt, carpet_persian, carpet_round, brass, iron, book_spines, world_map, portrait, clock_face,
mirror, glass, wax, flame, embers, skin, coat, trousers, leather, hair, doll_dress, curtain, marble, night_sky, paper

## Models (`assets/models/<id>.obj` + `<id>.mtl`, `map_Kd ../textures/<id>.png`)
Units: metres. +Y up. Front faces **+Z**. Origin = bottom centre (y = 0 is where it rests), centred on X and Z,
unless marked **[wall]** (back face at z = 0, body extends toward +Z) or **[hang]** (origin at the top attachment point,
body extends to negative Y). Sizes are W(x) x H(y) x D(z) and must be respected within ~5%.
Every material has a `map_Kd` texture. Glowing parts also set `Ke`.

| id | Description | Size W x H x D | Textures |
|---|---|---|---|
| player | adult in long coat, arms at sides, faces +Z | 0.55 x 1.80 x 0.35 | skin, coat, trousers, leather, hair |
| lib_bookshelf | tall bookcase, 5 shelves full of books, crown moulding | 2.0 x 2.6 x 0.45 | wood_dark, book_spines |
| lib_desk | writing desk, two drawer pedestals, papers + inkwell on top | 1.6 x 0.78 x 0.8 | wood_dark, paper, brass |
| lib_armchair | wingback armchair | 0.9 x 1.15 x 0.9 | velvet_red, wood_dark |
| lib_fireplace [wall] | stone fireplace, wood mantel, recessed sooty firebox with glowing logs + flames | 2.2 x 1.6 x 0.6 | stone, wood_dark, brick_soot, embers(Ke), flame(Ke) |
| lib_clock | grandfather clock, face on front of hood, brass pendulum | 0.6 x 2.3 x 0.4 | wood_dark, clock_face, brass |
| lib_globe | floor globe on round-base stand, sphere d=0.55 at y≈0.8, brass meridian | 0.7 x 1.15 x 0.7 | world_map, wood_light, brass |
| lib_candelabra | 5-arm candelabra with candles and flames | 0.45 x 0.6 x 0.15 | brass, wax, flame(Ke) |
| lib_side_table | round pedestal table | 0.6 x 0.7 x 0.6 | wood_light |
| lib_rug | rectangular Persian rug | 3.0 x 0.02 x 2.0 | carpet_persian |
| lib_portrait [wall] | gilded frame + painted portrait | 1.0 x 1.3 x 0.08 | brass, portrait |
| lib_bust | marble bust on a pedestal | 0.45 x 1.65 x 0.45 | marble |
| lib_chandelier [hang] | iron/brass chandelier with chain and candles | 1.0 x 0.9 x 1.0 | iron, brass, wax, flame(Ke) |
| bed_bed | four-poster bed, headboard at -Z end, pillows, quilt | 1.8 x 2.1 x 2.2 | wood_dark, linen, quilt |
| bed_wardrobe | tall paneled wardrobe, two doors, brass knobs | 1.4 x 2.2 x 0.6 | wood_light, brass |
| bed_rocking_chair | rocking chair, curved rockers along Z | 0.65 x 1.1 x 1.0 | wood_light |
| bed_nightstand | small table with drawer | 0.5 x 0.62 x 0.42 | wood_painted_white, brass |
| bed_lamp | oil lamp: brass base, glass chimney, flame | 0.25 x 0.5 x 0.25 | brass, glass(Ke small), flame(Ke) |
| bed_dresser | chest of drawers with tall mirror on top | 1.2 x 1.95 x 0.5 | wood_painted_white, mirror, brass |
| bed_music_box | small box with lid and tiny ballerina on top | 0.26 x 0.16 x 0.18 | wood_dark, brass, doll_dress |
| bed_toy_chest | toy chest with iron bands | 0.9 x 0.55 x 0.5 | wood_light, iron |
| bed_doll | sitting porcelain doll | 0.22 x 0.42 x 0.25 | skin, doll_dress, hair |
| bed_rug | round rug | 2.4 x 0.02 x 2.4 | carpet_round |
| bed_window [wall] | window with night-sky panes and curtains | 1.6 x 2.2 x 0.2 | wood_painted_white, night_sky(Ke dim blue), curtain |

## Sounds (`assets/sounds/<name>.wav`, 16-bit PCM mono 22050 Hz)
footstep, interact, bump, click, switch, horror_on, heartbeat (loop), whisper, ambient_library (loop), ambient_bedroom (loop),
chime, music_box, light_on, creak, flashlight

## Controls
| Key | Action |
|---|---|
| W / S, A / D | move forward/back, strafe left/right (time-based) |
| Arrow keys | turn view left/right/up/down |
| E | interact with the nearest object in its range + view angle |
| 1 / 2 / 3 | Top / Front / Side view buttons (player still moves) |
| 4 | Perspective button: toggles first-person / third-person; from Top/Front/Side returns to navigation |
| mouse click | the four on-screen buttons |
| TAB | switch scene; F1 = Library, F2 = Bedroom |
| F | flashlight (spotlight) on/off |
| Z / X | moonlight dimmer / brighter |
| C / V | lamps & fire dimmer / brighter |
| B / N | flashlight dimmer / brighter |
| H | toggle horror mode |
| ESC | quit |

---

# Quest extension (v2): map → key → riddle → win, plus jump scare

## Story / flow (state persists across scene switches)
1. **Library:** complete all 4 Library interactions (globe, bookshelf, candelabra, clock — any order, each counts once).
   When the 4th finishes: play `objective`, HUD toast "Something rustles on the desk...", and `lib_map` appears on the desk
   (hidden and non-interactive until then; it fades in with a soft glow).
2. **Map:** `lib_map` is a normal interaction (camera close-up; lifts + unrolls + glows). When it finishes: sound `pickup`,
   the map object disappears (taken), toast "You took a map of the Nursery. Press M to read it."
3. **M key** toggles a parchment-styled map overlay (drawn with Hud rects/text): top-down plan of the **Nursery** — room
   outline, every object's XZ footprint (from its transformed model AABB) with a short label, the window/door side marks,
   a pulsing red **X** at `bed_key`, and the player's position + facing if the player is in the Nursery. Before the map is
   taken, M shows the toast "You have no map." Movement keeps working while the map is open.
4. **Nursery:** `bed_key` lies hidden in a dark spot. It is drawn always but is NOT interactable and shows no prompt until
   the map is taken. Then it is a normal interaction ("Pick up the strange key"; camera close-up; rises, spins, glows).
   When that interaction finishes, the **riddle** opens.
5. **Riddle** (modal; movement, views and E are locked; `riddle` sound on open): parchment panel with the poem and four
   answer buttons, chosen with keys **1–4** or a mouse click. **Esc** closes the riddle without answering (it re-opens
   by interacting with the key again); Esc only quits the game when nothing modal is open.
   - Correct → sound `win`, the key disappears, **win screen**: "YOU ESCAPED THE MANOR", time taken, "Press ENTER to keep
     exploring". After ENTER the game continues; objective shows "The manor is quiet... for now."
   - Wrong → the riddle closes and the **jump scare** fires immediately; the key stays; next time the answer order is reshuffled.
6. **Jump scare** — key **J** at any time (not while the win screen or another scare is showing), and automatically on a
   wrong answer. About 1.8 s, input locked: brief darkness → `scare_doll` lunges from about 3 m to fill the screen in
   about 0.25 s, twitching with violent camera shake, red/white flash frames and a heavy red vignette; sounds `scare_hit` +
   `scream`; hold, then a hard cut to black and a fade back. Render the doll after the scene with the depth buffer cleared,
   lit by its own strong under-light so it reads in any lighting. Works in every view mode and both scenes.
7. **Objective line** in the HUD (under the room panel), always visible:
   "Investigate the Library (n/4)" → "Take the map from the desk" → "Read the map [M] and search the Nursery" →
   "Solve the riddle" (after a wrong answer: "Try the key again...") → "The manor is quiet... for now."
8. Add **J = Jump scare** and **M = Map** to the controls help panel (and the README, integrator).

## Riddle (use exactly)
```
I have hands, but nothing I hold.
I have a face that never grows old.
I stopped the night the child went still;
downstairs I wait, and always will.
What am I?
```
Choices (shuffle order each opening): "The portrait", "The grandfather clock" (correct), "The music box", "The rocking chair".

## New models (model agent)
| id | Description | Size W x H x D | Textures |
|---|---|---|---|
| lib_map | parchment map lying flat with both short ends curled into rolls, tied ribbon; rests on the desk top | 0.45 x 0.06 x 0.32 | map_parchment, velvet_red |
| bed_key | ornate antique skeleton key lying flat: looped bow, shaft, toothed bit | 0.16 x 0.02 x 0.05 | brass (with Ke ≈ 0.25 0.18 0.05 so it glints) |
| scare_doll | creepy porcelain doll, lunging pose: arms reaching toward +Z, head tilted, wild hair, torn dress; face front faces +Z | 0.55 x 0.9 x 0.45 | porcelain_cracked (face, mapped once on the front of the head), doll_dress, hair, skin; eye sockets Ke ≈ 0.6 0 0 |

`scare_doll` is never a scene object; it is only drawn by the jump-scare effect.

## New textures (texture agent)
- **map_parchment:** aged parchment showing a hand-drawn ink floor plan with little furniture sketches, a compass rose,
  scribbled notes and a red X. Mapped once.
- **porcelain_cracked:** a pale porcelain doll face filling the image: hollow black eyes, hairline cracks, rosy cheeks
  gone grey, a thin smeared red smile. Mapped once on the face.

## New sounds (audio agent)
scream (~1.3 s, piercing distorted shriek), scare_hit (~0.9 s, loud dissonant orchestral stab and impact),
win (~3.5 s, eerie music-box phrase resolving to a warm major chord with a bell), riddle (~1.5 s, low whisper swell and
chime), pickup (~0.6 s, paper rustle and a small metallic jingle), objective (~0.9 s, soft low chime).

## Ownership for v2
Gameplay agent: Game.h/.cpp, DevTour.cpp (+ new src/Quest.* files if wanted). Scenes agent: Scene.cpp, tools/check_layout.cpp.
Model agent: tools/gen_models.py + new OBJs. Texture agent: tools/gen_textures.py + new PNGs.
Audio agent: tools/gen_sounds.py + new WAVs. Integrator: README, CONTRACT.
