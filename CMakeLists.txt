cmake_minimum_required(VERSION 2.8)
project(random_walk)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(random_walk rwalk_kernel.cc)