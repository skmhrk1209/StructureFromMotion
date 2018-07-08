#pragma once

#include "structure_from_motion.hpp"

class StructureFromMotionApp : public ci::app::App {
   public:
    StructureFromMotionApp() {}
    virtual ~StructureFromMotionApp() {
        if (process && process->joinable()) process->detach();
    }

    StructureFromMotionApp(const StructureFromMotionApp&) = delete;
    StructureFromMotionApp(StructureFromMotionApp&&) = delete;

    StructureFromMotionApp& operator=(const StructureFromMotionApp&) = delete;
    StructureFromMotionApp& operator=(StructureFromMotionApp&&) = delete;

    void setup() override;
    void update() override;
    void draw() override;

    void setupShaders();
    void setupCallbacks();
    void setupProcessors();

   private:
    StructureFromMotion::Ptr structureFromMotion;
    boost::signals2::connection structureFromMotionConnection;

    std::unique_ptr<std::thread> process;
    ci::gl::ContextRef context;

    float inradius;
    float wheelAngle;

    std::vector<ci::gl::TextureRef> textures;
    std::vector<ci::gl::BatchRef> planes;
    ci::gl::GlslProgRef planeShader;

    ci::gl::BatchRef circle;
    ci::gl::GlslProgRef circleShader;

    ci::gl::VboMeshRef pointCloud;

    ci::CameraPersp camera;
    ci::CameraUi cameraUi;

    std::unordered_map<std::string, std::function<void()>> callbacks;
};
