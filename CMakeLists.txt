# Compile and run instructions:
# rm -rf build
# cmake -DAscent_DIR=/your/path/to/ascent -B build -S .
# cp ascent_actions.yaml build 
# cd build
# make
# mpiexec -n 4 ./lbmcfd

cmake_minimum_required(VERSION 3.9)

project(lbmcfd)

option(BUILD_ASCENT "Build with ASCENT support" ON)

if (BUILD_ASCENT)
    add_compile_definitions(ASCENT_ENABLED)
    find_package(Ascent REQUIRED
                NO_DEFAULT_PATH
                PATHS ${Ascent_DIR})
endif()

add_executable(lbmcfd src/main.cpp)

target_include_directories(lbmcfd PUBLIC include)

target_link_libraries(lbmcfd ascent::ascent_mpi)