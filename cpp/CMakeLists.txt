cmake_minimum_required(VERSION 3.12)
project(HyperConformal LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for embedded targets
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -fomit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -DDEBUG")

# Platform-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "ARM")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif()

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/hyperconformal.c
)

# Create library
add_library(hyperconformal STATIC ${SOURCES})

# Set library properties
set_target_properties(hyperconformal PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER "include/hyperconformal.h"
)

# Install targets
install(TARGETS hyperconformal
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Examples (optional)
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests (optional)
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS AND NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Arduino support
option(ARDUINO_BUILD "Build for Arduino" OFF)
if(ARDUINO_BUILD)
    # Arduino-specific flags and settings
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DARDUINO")
    # Remove standard library dependencies
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -nostdlib")
endif()

# Packaging
include(CPack)
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HyperConformal - Embedded HDC with Conformal Prediction")
set(CPACK_PACKAGE_VENDOR "Terragon Labs")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")