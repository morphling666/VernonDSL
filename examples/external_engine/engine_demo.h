#ifndef VERNON_EXTERNAL_ENGINE_DEMO_H
#define VERNON_EXTERNAL_ENGINE_DEMO_H

#include "VernonRHI.h"

#include <cstdint>
#include <memory>

class GraphicsHost;

class ExternalEngineDemo {
public:
    virtual ~ExternalEngineDemo() = default;
    virtual bool initialize(bool headless) = 0;
    virtual bool renderFrame(double elapsedSeconds) = 0;
    virtual bool shouldClose() const = 0;
    virtual void shutdown() = 0;
};

class ExternalEnginePanel {
public:
    virtual ~ExternalEnginePanel() = default;
    virtual bool initialize(GraphicsHost *graphics, bool headless) = 0;
    virtual bool renderFrame(double elapsedSeconds, uint32_t panelWidth, uint32_t panelHeight) = 0;
    virtual VernonRhiImage image() const = 0;
    virtual uint32_t imageWidth() const = 0;
    virtual uint32_t imageHeight() const = 0;
    virtual void shutdown() = 0;
};

std::unique_ptr<ExternalEnginePanel> createCpuFractalPanel();
std::unique_ptr<ExternalEnginePanel> createMandelbulbPanel();
std::unique_ptr<ExternalEngineDemo> createExternalEngineDemo();

#endif
