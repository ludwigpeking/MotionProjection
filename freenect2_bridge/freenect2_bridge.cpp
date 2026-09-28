// Kinect v2 through libfreenect2, published to Python over a Windows shared-memory mapping.
//
// Replaces the Microsoft Kinect driver path (which restarts this sensor every 7 s on
// its thermal reading; libfreenect2 ignores that reading). Each frame the bridge writes:
//   header   : sequence, sizes, timestamp, colour and IR camera intrinsics
//   colour   : 1920 x 1080 x 4 bytes (BGRX)
//   bigdepth : 1920 x 1082 floats, depth in mm registered onto the colour image (inf = unknown)
//   depth    : 512 x 424 floats, undistorted depth in mm
// The sequence number is written odd before the payload and even after it (a seqlock),
// so a reader that sees an even, unchanged sequence has a consistent frame.
//
// Build: see CMakeLists.txt next to this file. Run: freenect2_bridge.exe [cpu|opencl|cuda]

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

namespace {

const char* MAPPING_NAME = "Local\\freenect2_bridge_frames";
const char* STOP_EVENT_NAME = "Local\\freenect2_bridge_stop";
const unsigned int COLOUR_WIDTH = 1920;
const unsigned int COLOUR_HEIGHT = 1080;
const unsigned int BIGDEPTH_HEIGHT = 1082;
const unsigned int DEPTH_WIDTH = 512;
const unsigned int DEPTH_HEIGHT = 424;

#pragma pack(push, 1)
struct Header {
    unsigned int magic;               // 0x46324B42 "F2KB"
    unsigned int header_bytes;
    volatile unsigned int sequence;   // odd while a frame is being written, even when complete
    unsigned int frame_count;
    double timestamp_seconds;         // steady clock, seconds
    unsigned int colour_width, colour_height, bigdepth_height, depth_width, depth_height;
    float colour_fx, colour_fy, colour_cx, colour_cy;
    float ir_fx, ir_fy, ir_cx, ir_cy, ir_k1, ir_k2, ir_k3, ir_p1, ir_p2;
    unsigned int colour_offset, bigdepth_offset, depth_offset, total_bytes;
    char pipeline[32];
};
#pragma pack(pop)

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string requested_pipeline = argc > 1 ? argv[1] : "opencl";
    // Level from LIBFREENECT2_LOGGER_LEVEL (debug|info|warning|error), default Info: the OpenCL
    // device choice and USB stream problems are reported at Info.
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLoggerWithDefaultLevel());

    libfreenect2::Freenect2 freenect2;
    if (freenect2.enumerateDevices() == 0) {
        std::fprintf(stderr, "[bridge] no Kinect v2 found (is the libusbK driver installed for it?)\n");
        return 1;
    }
    std::string serial = freenect2.getDefaultDeviceSerialNumber();

    libfreenect2::PacketPipeline* pipeline = 0;
    std::string pipeline_name;
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
    if (requested_pipeline == "cuda") { pipeline = new libfreenect2::CudaPacketPipeline(); pipeline_name = "cuda"; }
#endif
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
    if (!pipeline && requested_pipeline != "cpu") { pipeline = new libfreenect2::OpenCLPacketPipeline(); pipeline_name = "opencl"; }
#endif
    if (!pipeline) { pipeline = new libfreenect2::CpuPacketPipeline(); pipeline_name = "cpu"; }

    libfreenect2::Freenect2Device* device = freenect2.openDevice(serial, pipeline);
    if (!device) {
        std::fprintf(stderr, "[bridge] could not open the Kinect %s\n", serial.c_str());
        return 1;
    }
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    if (!device->start()) {
        std::fprintf(stderr, "[bridge] the Kinect did not start\n");
        return 1;
    }
    libfreenect2::Freenect2Device::ColorCameraParams colour_params = device->getColorCameraParams();
    libfreenect2::Freenect2Device::IrCameraParams ir_params = device->getIrCameraParams();
    libfreenect2::Registration registration(ir_params, colour_params);
    std::printf("[bridge] Kinect %s started, pipeline %s, colour focal %.1f/%.1f centre %.1f/%.1f\n",
                serial.c_str(), pipeline_name.c_str(), colour_params.fx, colour_params.fy, colour_params.cx, colour_params.cy);
    std::printf("[bridge] ir params: fx %.3f fy %.3f cx %.3f cy %.3f k1 %.5f k2 %.5f k3 %.5f p1 %.5f p2 %.5f\n",
                ir_params.fx, ir_params.fy, ir_params.cx, ir_params.cy, ir_params.k1, ir_params.k2, ir_params.k3,
                ir_params.p1, ir_params.p2);
    std::printf("[bridge] colour params: shift_d %.3f shift_m %.3f mx_x3y0 %.6g mx_x0y3 %.6g mx_x2y1 %.6g mx_x1y2 %.6g "
                "mx_x2y0 %.6g mx_x0y2 %.6g mx_x1y1 %.6g mx_x1y0 %.6g mx_x0y1 %.6g mx_x0y0 %.6g\n",
                colour_params.shift_d, colour_params.shift_m, colour_params.mx_x3y0, colour_params.mx_x0y3,
                colour_params.mx_x2y1, colour_params.mx_x1y2, colour_params.mx_x2y0, colour_params.mx_x0y2,
                colour_params.mx_x1y1, colour_params.mx_x1y0, colour_params.mx_x0y1, colour_params.mx_x0y0);
    std::printf("[bridge] colour params: my_x3y0 %.6g my_x0y3 %.6g my_x2y1 %.6g my_x1y2 %.6g my_x2y0 %.6g my_x0y2 %.6g "
                "my_x1y1 %.6g my_x1y0 %.6g my_x0y1 %.6g my_x0y0 %.6g\n",
                colour_params.my_x3y0, colour_params.my_x0y3, colour_params.my_x2y1, colour_params.my_x1y2,
                colour_params.my_x2y0, colour_params.my_x0y2, colour_params.my_x1y1, colour_params.my_x1y0,
                colour_params.my_x0y1, colour_params.my_x0y0);
    std::fflush(stdout);

    Header header;
    std::memset(&header, 0, sizeof(header));
    header.magic = 0x46324B42;
    header.header_bytes = sizeof(Header);
    header.colour_width = COLOUR_WIDTH;
    header.colour_height = COLOUR_HEIGHT;
    header.bigdepth_height = BIGDEPTH_HEIGHT;
    header.depth_width = DEPTH_WIDTH;
    header.depth_height = DEPTH_HEIGHT;
    header.colour_fx = colour_params.fx; header.colour_fy = colour_params.fy;
    header.colour_cx = colour_params.cx; header.colour_cy = colour_params.cy;
    header.ir_fx = ir_params.fx; header.ir_fy = ir_params.fy; header.ir_cx = ir_params.cx; header.ir_cy = ir_params.cy;
    header.ir_k1 = ir_params.k1; header.ir_k2 = ir_params.k2; header.ir_k3 = ir_params.k3;
    header.ir_p1 = ir_params.p1; header.ir_p2 = ir_params.p2;
    header.colour_offset = sizeof(Header);
    header.bigdepth_offset = header.colour_offset + COLOUR_WIDTH * COLOUR_HEIGHT * 4;
    header.depth_offset = header.bigdepth_offset + COLOUR_WIDTH * BIGDEPTH_HEIGHT * 4;
    // After the undistorted depth: the colour registered onto the depth grid (512 x 424 BGRX).
    header.total_bytes = header.depth_offset + DEPTH_WIDTH * DEPTH_HEIGHT * 4 + DEPTH_WIDTH * DEPTH_HEIGHT * 4;
    std::strncpy(header.pipeline, pipeline_name.c_str(), sizeof(header.pipeline) - 1);

    HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, header.total_bytes, MAPPING_NAME);
    if (!mapping) { std::fprintf(stderr, "[bridge] CreateFileMapping failed\n"); return 1; }
    unsigned char* shared = static_cast<unsigned char*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, header.total_bytes));
    if (!shared) { std::fprintf(stderr, "[bridge] MapViewOfFile failed\n"); return 1; }
    HANDLE stop_event = CreateEventA(NULL, TRUE, FALSE, STOP_EVENT_NAME);
    Header* shared_header = reinterpret_cast<Header*>(shared);
    std::memcpy(shared, &header, sizeof(header));

    libfreenect2::Frame undistorted(DEPTH_WIDTH, DEPTH_HEIGHT, 4);
    libfreenect2::Frame registered(DEPTH_WIDTH, DEPTH_HEIGHT, 4);
    libfreenect2::Frame bigdepth(COLOUR_WIDTH, BIGDEPTH_HEIGHT, 4);
    libfreenect2::FrameMap frames;
    unsigned int frame_count = 0;
    double report_time = now_seconds();
    unsigned int report_frames = 0;
    unsigned int sequence = 0;

    while (WaitForSingleObject(stop_event, 0) != WAIT_OBJECT_0) {
        if (!listener.waitForNewFrame(frames, 2000)) {
            std::printf("[bridge] no frame for 2 s\n");
            std::fflush(stdout);
            continue;
        }
        libfreenect2::Frame* rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];
        registration.apply(rgb, depth, &undistorted, &registered, true, &bigdepth);

        shared_header->sequence = ++sequence;        // odd: writing
        MemoryBarrier();
        std::memcpy(shared + header.colour_offset, rgb->data, COLOUR_WIDTH * COLOUR_HEIGHT * 4);
        std::memcpy(shared + header.bigdepth_offset, bigdepth.data, COLOUR_WIDTH * BIGDEPTH_HEIGHT * 4);
        std::memcpy(shared + header.depth_offset, undistorted.data, DEPTH_WIDTH * DEPTH_HEIGHT * 4);
        std::memcpy(shared + header.depth_offset + DEPTH_WIDTH * DEPTH_HEIGHT * 4, registered.data,
                    DEPTH_WIDTH * DEPTH_HEIGHT * 4);
        shared_header->timestamp_seconds = now_seconds();
        shared_header->frame_count = ++frame_count;
        MemoryBarrier();
        shared_header->sequence = ++sequence;        // even: complete
        listener.release(frames);

        ++report_frames;
        double now = now_seconds();
        if (now - report_time >= 5.0) {
            std::printf("[bridge] %.1f fps\n", report_frames / (now - report_time));
            std::fflush(stdout);
            report_time = now;
            report_frames = 0;
        }
    }
    device->stop();
    device->close();
    UnmapViewOfFile(shared);
    CloseHandle(mapping);
    std::printf("[bridge] stopped after %u frames\n", frame_count);
    return 0;
}
