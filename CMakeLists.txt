cmake_minimum_required(VERSION 3.9)

project(lbmcfd)

add_compile_definitions(ASCENT_ENABLED)
find_package(Ascent REQUIRED NO_DEFAULT_PATH PATHS ${Ascent_DIR})

add_executable(lbmcfd src/main.cpp)

target_include_directories(lbmcfd PUBLIC include)

target_link_libraries(lbmcfd ascent::ascent_mpi)
