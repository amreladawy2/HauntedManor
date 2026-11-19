#include "Model.h"
#include "Paths.h"
#include "Primitives.h"
#include "Shader.h"
#include "Texture.h"

#include <assimp/cimport.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>

void Mesh::draw(const Shader& s) const {
    glBindTextureUnit(0, texture);
    s.setVec3("uBaseColor", baseColor);
    s.setVec3("uMeshEmissive", emissive);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, nullptr);
}

Mesh uploadMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned>& indices, GLuint texture) {
    Mesh m;
    m.texture = texture;
    m.indexCount = static_cast<GLsizei>(indices.size());

    glCreateBuffers(1, &m.vbo);
    glNamedBufferStorage(m.vbo, vertices.size() * sizeof(Vertex), vertices.data(), 0);
    glCreateBuffers(1, &m.ebo);
    glNamedBufferStorage(m.ebo, indices.size() * sizeof(unsigned), indices.data(), 0);

    glCreateVertexArrays(1, &m.vao);
    glVertexArrayVertexBuffer(m.vao, 0, m.vbo, 0, sizeof(Vertex));
    glVertexArrayElementBuffer(m.vao, m.ebo);
    const GLuint offsets[3] = {offsetof(Vertex, position), offsetof(Vertex, normal), offsetof(Vertex, uv)};
    const GLint sizes[3] = {3, 3, 2};
    for (GLuint a = 0; a < 3; ++a) {
        glEnableVertexArrayAttrib(m.vao, a);
        glVertexArrayAttribFormat(m.vao, a, sizes[a], GL_FLOAT, GL_FALSE, offsets[a]);
        glVertexArrayAttribBinding(m.vao, a, 0);
    }
    return m;
}

namespace {

glm::mat4 toGlm(const aiMatrix4x4& m) {
    // Assimp is row-major, glm column-major.
    return glm::mat4(m.a1, m.b1, m.c1, m.d1,
                     m.a2, m.b2, m.c2, m.d2,
                     m.a3, m.b3, m.c3, m.d3,
                     m.a4, m.b4, m.c4, m.d4);
}

struct Loader {
    const aiScene* scene;
    std::filesystem::path dir;   // model file's directory; MTL texture paths are relative to it
    Model& model;

    Mesh buildMesh(const aiMesh* am, const glm::mat4& xf) {
        const glm::mat3 nxf = glm::transpose(glm::inverse(glm::mat3(xf)));
        std::vector<Vertex> verts(am->mNumVertices);
        for (unsigned i = 0; i < am->mNumVertices; ++i) {
            const aiVector3D& p = am->mVertices[i];
            verts[i].position = glm::vec3(xf * glm::vec4(p.x, p.y, p.z, 1.0f));
            verts[i].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            if (am->mNormals) {
                const aiVector3D& n = am->mNormals[i];
                verts[i].normal = glm::normalize(nxf * glm::vec3(n.x, n.y, n.z));
            }
            verts[i].uv = glm::vec2(0.0f);
            if (am->mTextureCoords[0]) verts[i].uv = {am->mTextureCoords[0][i].x, am->mTextureCoords[0][i].y};
            model.boundsMin = glm::min(model.boundsMin, verts[i].position);
            model.boundsMax = glm::max(model.boundsMax, verts[i].position);
        }

        std::vector<unsigned> idx;
        idx.reserve(am->mNumFaces * 3);
        for (unsigned f = 0; f < am->mNumFaces; ++f) {
            const aiFace& face = am->mFaces[f];
            if (face.mNumIndices != 3) continue;   // points/lines survive triangulation
            idx.insert(idx.end(), face.mIndices, face.mIndices + 3);
        }

        const aiMaterial* mat = scene->mMaterials[am->mMaterialIndex];
        GLuint tex = whiteTexture();
        aiString texPath;
        if (aiGetMaterialTexture(mat, aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS && texPath.length > 0
            && texPath.data[0] != '*') {   // "*N" means embedded texture, which we do not support
            std::string rel = texPath.C_Str();
            for (char& c : rel) if (c == '\\') c = '/';
            tex = loadTexture((dir / rel).lexically_normal().string());
        }

        Mesh m = uploadMesh(verts, idx, tex);
        aiColor4D c;
        if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &c) == AI_SUCCESS) m.baseColor = {c.r, c.g, c.b};
        if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_EMISSIVE, &c) == AI_SUCCESS) m.emissive = {c.r, c.g, c.b};
        return m;
    }

    void visit(const aiNode* node, const glm::mat4& parent) {
        const glm::mat4 xf = parent * toGlm(node->mTransformation);
        for (unsigned i = 0; i < node->mNumMeshes; ++i) {
            const aiMesh* am = scene->mMeshes[node->mMeshes[i]];
            if (am->mNumVertices && am->mNumFaces) model.meshes.push_back(buildMesh(am, xf));
        }
        for (unsigned i = 0; i < node->mNumChildren; ++i) visit(node->mChildren[i], xf);
    }
};

} // namespace

bool Model::load(const std::string& path) {
    const std::string resolved = resolvePath(path);
    // The C API keeps std:: types off the DLL boundary (the exe links libstdc++ statically).
    const aiScene* scene = aiImportFile(resolved.c_str(),
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality | aiProcess_OptimizeMeshes);
    if (!scene || !scene->mRootNode || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        std::cerr << "[Model] failed to load " << path << ": " << aiGetErrorString() << "\n";
        if (scene) aiReleaseImport(scene);
        return false;
    }

    meshes.clear();
    boundsMin = glm::vec3(std::numeric_limits<float>::max());
    boundsMax = glm::vec3(std::numeric_limits<float>::lowest());
    Loader{scene, std::filesystem::path(resolved).parent_path(), *this}.visit(scene->mRootNode, glm::mat4(1.0f));
    aiReleaseImport(scene);

    if (meshes.empty()) {
        std::cerr << "[Model] " << path << " contains no triangles\n";
        boundsMin = boundsMax = glm::vec3(0.0f);
        return false;
    }
    return true;
}

void Model::draw(const Shader& s) const {
    for (const Mesh& m : meshes) m.draw(s);
}

Model* ModelCache::get(const std::string& id) {
    if (auto it = models_.find(id); it != models_.end()) return it->second.get();

    auto model = std::make_unique<Model>();
    if (!model->load("assets/models/" + id + ".obj")) {
        std::cerr << "[Model] using placeholder cube for '" << id << "'\n";
        model->meshes = {makeBox(glm::vec3(1.0f), 1.0f, whiteTexture())};
        model->meshes[0].baseColor = {1.0f, 0.0f, 1.0f};   // loud magenta so missing assets are obvious
        model->boundsMin = glm::vec3(-0.5f);
        model->boundsMax = glm::vec3(0.5f);
    }
    return models_.emplace(id, std::move(model)).first->second.get();
}
