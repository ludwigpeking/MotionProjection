cmake_minimum_required(VERSION 3.20)
project(freenect2_bridge CXX)

# libfreenect2's generated freenect2Config.cmake is not usable on Windows (backslash
# paths), so the install prefix is taken directly.
set(freenect2_ROOT "C:/Users/qli/tools/libfreenect2/install" CACHE PATH "libfreenect2 install prefix")

add_executable(freenect2_bridge freenect2_bridge.cpp)
target_include_directories(freenect2_bridge PRIVATE "${freenect2_ROOT}/include")
target_link_libraries(freenect2_bridge PRIVATE "${freenect2_ROOT}/lib/freenect2.lib")
set_property(TARGET freenect2_bridge PROPERTY CXX_STANDARD 14)

# Runtime DLLs next to the executable (turbojpeg from vcpkg pulls in spng and zlib).
set(vcpkg_BIN "C:/Users/qli/tools/vcpkg/installed/x64-windows/bin" CACHE PATH "vcpkg bin directory")
add_custom_command(TARGET freenect2_bridge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${freenect2_ROOT}/bin/freenect2.dll" "${freenect2_ROOT}/bin/libusb-1.0.dll" "${freenect2_ROOT}/bin/turbojpeg.dll"
        "${vcpkg_BIN}/spng.dll" "${vcpkg_BIN}/z.dll"
        "$<TARGET_FILE_DIR:freenect2_bridge>")
