cmake_minimum_required(VERSION 3.0.0)
project(fastklt VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(fastklt main.cpp)
target_link_libraries( fastklt ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
