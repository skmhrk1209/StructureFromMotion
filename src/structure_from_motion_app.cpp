#include "structure_from_motion_app.hpp"

// #define DEBUG

void StructureFromMotionApp::setupShaders() {
    planeShader = ci::gl::GlslProg::create(
        ci::gl::GlslProg::Format()

            .vertex(CI_GLSL(
                150,

                uniform mat4 ciModelViewProjection;
                uniform mat4 ciModelMatrix;

                in vec4 ciPosition; in vec2 ciTexCoord0; out vec4 position;
                out vec2 texCoord;

                void main(void) {
                    position = ciModelMatrix * ciPosition;
                    texCoord = ciTexCoord0;

                    gl_Position = ciModelViewProjection * ciPosition;
                }))

            .fragment(CI_GLSL(
                150,

                uniform sampler2D texUnit;
                uniform float min; uniform float max;

                in vec4 position; in vec2 texCoord; out vec4 color;

                float scale(float inputVal, float inputMin, float inputMax,
                            float outputMin, float outputMax) {
                    return (inputVal - inputMin) / (inputMax - inputMin) *
                               (outputMax - outputMin) +
                           outputMin;
                }

                void main(void) {
                    color = texture(texUnit, texCoord) *
                            scale(position.z, min, max, 0.0, 1.0);
                })));

    circleShader = ci::gl::GlslProg::create(
        ci::gl::GlslProg::Format()

            .vertex(CI_GLSL(
                150,

                uniform mat4 ciModelViewProjection;

                in vec4 ciPosition; out vec4 position;

                void main(void) {
                    position = ciPosition;

                    gl_Position = ciModelViewProjection * ciPosition;
                }))

            .fragment(CI_GLSL(
                150,

                const float PI = 3.14159265358979323846;

                uniform float time;

                in vec4 position; out vec4 color;

                float scale(float inputVal, float inputMin, float inputMax,
                            float outputMin, float outputMax) {
                    return (inputVal - inputMin) / (inputMax - inputMin) *
                               (outputMax - outputMin) +
                           outputMin;
                }

                float angle(float time) { return 10.0 * time; }

                void main(void) {
                    float angleDiff =
                        atan(position.y, position.x) - angle(time);

                    while (angleDiff < 0) angleDiff += PI * 2.0;

                    while (angleDiff >= PI * 2.0) angleDiff -= PI * 2.0;

                    color = length(position) > 4.0
                                ? vec4(1.0, 1.0, 1.0,
                                       exp2(scale(angleDiff, 0.0, PI * 2.0,
                                                  -10.0, 0.0)))
                                : vec4(0.0, 0.0, 0.0, 1.0);
                })));
}

void StructureFromMotionApp::setupCallbacks() {
    getWindow()->getSignalMouseWheel().connect(
        [&](const ci::app::MouseEvent& event) {
            wheelAngle += event.getWheelIncrement() * 0.1;
        });

    getWindow()->getSignalKeyDown().connect(
        [&](const ci::app::KeyEvent& event) {
            auto it(callbacks.find({event.getChar()}));

            if (it != end(callbacks)) it->second();
        });

    getWindow()->getSignalFileDrop().connect(
        [&](const ci::app::FileDropEvent& event) {
            std::unordered_set<std::string> imageExtentions{".png", ".jpg",
                                                            ".jpeg"};
            std::unordered_set<std::string> parameterExtentions{".xml"};

            const auto& filePaths(event.getFiles());

            auto length(1.0);

            for (const auto& filePath : filePaths) {
                if (imageExtentions.find(filePath.extension().string()) !=
                    end(imageExtentions)) {
                    auto texture(
                        ci::gl::Texture::create(ci::loadImage(filePath)));

                    texture->bind(textures.size());

                    textures.push_back(texture);

                    planes.push_back(ci::gl::Batch::create(
                        ci::geom::Plane()
                            .size(ci::vec2(length, length))
                            .axes(ci::vec3(1.0, 0.0, 0.0),
                                  ci::vec3(0.0, 1.0, 0.0)),
                        planeShader));

                    structureFromMotion->loadImage(filePath.string());
                } else if (parameterExtentions.find(
                               filePath.extension().string()) !=
                           end(parameterExtentions)) {
                    structureFromMotion->loadParameter(filePath.string());
                }
            }

            inradius = length / 2.0 / tan(M_PI / (planes.size() * 1.5));

            planeShader->uniform("min", -inradius);
            planeShader->uniform("max", +inradius);
        });

    callbacks.emplace("r", [&]() {
        process = std::make_unique<std::thread>(
            [&]() { structureFromMotion->process(); });
    });

    callbacks.emplace("c", [&]() {
        textures.clear();
        planes.clear();
        structureFromMotion->clearImages();
    });
}

void StructureFromMotionApp::setupProcessors() {
    structureFromMotion = StructureFromMotion::create();

    structureFromMotionConnection =
        structureFromMotion->registerCallback([&](const auto& _pointCloud) {
            ci::ThreadSetup threadSetup;
            context->makeCurrent();

            if (!pointCloud) {
                camera.lookAt(ci::vec3(0.0, 0.0, -10.0),
                              ci::vec3(0.0, 0.0, 0.0),
                              ci::vec3(0.0, -1.0, 0.0));
                cameraUi.setCamera(&camera);
                cameraUi.connect(getWindow());
            }

            std::vector<ci::vec3> positions;
            std::vector<ci::Color> colors;
            for (const auto& point : _pointCloud) {
                positions.emplace_back(point.first.x, point.first.y,
                                       point.first.z);
                colors.emplace_back(point.second[2] / 255.0,
                                    point.second[1] / 255.0,
                                    point.second[0] / 255.0);
            }

            std::vector<ci::gl::VboMesh::Layout> layout{
                ci::gl::VboMesh::Layout()
                    .usage(GL_STATIC_DRAW)
                    .attrib(ci::geom::Attrib::POSITION, 3),
                ci::gl::VboMesh::Layout()
                    .usage(GL_STATIC_DRAW)
                    .attrib(ci::geom::Attrib::COLOR, 3),
            };

            ci::gl::Sync::create()->clientWaitSync();

            pointCloud = ci::gl::VboMesh::create(
                static_cast<uint32_t>(positions.size()), GL_POINTS, layout);

            pointCloud->bufferAttrib(ci::geom::Attrib::POSITION, positions);
            pointCloud->bufferAttrib(ci::geom::Attrib::COLOR, colors);
        });
}

void StructureFromMotionApp::setup() {
    ci::gl::enableDepthRead();
    ci::gl::enableDepthWrite();

    context = ci::gl::Context::create(ci::gl::context());

    setupShaders();

    setupCallbacks();

    setupProcessors();

    circle =
        ci::gl::Batch::create(ci::geom::Circle().radius(8.0), circleShader);

    camera.lookAt(ci::vec3(0.0, 10.0, 20.0), ci::vec3(0.0, 2.0, 0.0),
                  ci::vec3(0.0, 1.0, 0.0));
}

void StructureFromMotionApp::update() {}

void StructureFromMotionApp::draw() {
    ci::gl::clear(ci::ColorA(0.0, 0.0, 0.0, 1.0));

    if (pointCloud) {
        ci::gl::ScopedMatrices scopedMatrices;

        ci::gl::setMatrices(camera);

        ci::gl::draw(pointCloud);
    }

    else {
        {
            ci::gl::ScopedMatrices scopedMatrices;

            ci::gl::setMatricesWindow(getWindowSize());
            ci::gl::translate(getWindowCenter());

            if (process) {
                ci::gl::drawStringCentered("Now Recovering Images ...",
                                           ci::vec2(0, -80), ci::Color(1, 1, 1),
                                           ci::Font("Futura", 16));

                ci::gl::ScopedModelMatrix scopedMatrix;

                ci::gl::translate(ci::vec2(-120, -75));

                circleShader->uniform("time",
                                      static_cast<float>(getElapsedSeconds()));

                circle->draw();
            } else {
                ci::gl::drawStringCentered("Drag and Drop Image Files Here !",
                                           ci::vec2(0, -80), ci::Color(1, 1, 1),
                                           ci::Font("Futura", 16));

                if (!planes.empty()) {
                    ci::gl::drawStringCentered(
                        "Press 'r' to recover images.", ci::vec2(0, -40),
                        ci::Color(1, 1, 1), ci::Font("Futura", 16));

                    ci::gl::drawStringCentered(
                        "Press 'c' to clear images.", ci::vec2(0, 0),
                        ci::Color(1, 1, 1), ci::Font("Futura", 16));
                }
            }
        }

        {
            ci::gl::ScopedMatrices scopedMatrices;

            ci::gl::setMatrices(camera);

            ci::gl::rotate(angleAxis(wheelAngle, ci::vec3(0.0, 1.0, 0.0)));

            auto texUnit(0);
            for (const auto& plane : planes) {
                planeShader->uniform("texUnit", texUnit++);

                auto angle(M_PI * 2.0 / planes.size());

                ci::gl::rotate(angleAxis(static_cast<float>(angle),
                                         ci::vec3(0.0, 1.0, 0.0)));

                ci::gl::ScopedModelMatrix scopedMatrix;

                ci::gl::translate(ci::vec3(0.0, 0.0, inradius));

                plane->draw();
            }
        }
    }
}

CINDER_APP(StructureFromMotionApp, ci::app::RendererGl,
           [](ci::app::App::Settings* setting) {
               setting->setTitle("Structure From Motion");
               setting->setWindowSize(ci::ivec2(800, 450));
           })
