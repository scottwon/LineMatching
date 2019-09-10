cmake_minimum_required(VERSION 2.8)

find_package(OpenCV PATHS "/usr/local/share/OpenCV")
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS})

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

SET(BUILD_SHARED_LIBS ON)

SET(ARPACK_INCLUDE_DIR "/usr/include/arpack++")
SET(ARPACK_LINK_DIR "/usr/lib")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake_modules/")

# Default locations to search for on various platforms.
#LIST(APPEND SEARCH_LIBS /usr/lib)
#LIST(APPEND SEARCH_LIBS /usr/local/lib)
#LIST(APPEND SEARCH_LIBS /opt/local/lib)

#LIST(APPEND SEARCH_HEADERS /usr/include)
#LIST(APPEND SEARCH_HEADERS /usr/local/include)
#LIST(APPEND SEARCH_HEADERS /opt/local/include)

#message("-- Check for Google Flags")
#find_library(GFLAGS_LIB NAMES gflags PATHS ${SEARCH_LIBS})
#find_path(GFLAGS_INCLUDE NAMES gflags/gflags.h PATHS ${SEARCH_HEADERS})

#set(GOOGLE_LIBRARIES ${GFLAGS_LIB})

include_directories(${linedetection_SOURCE_DIR}/include)
include_directories(library ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
#include_directories(${GFLAGS_INCLUDE})
#include_directories(.)
INCLUDE_DIRECTORIES(${ARPACK_INCLUDE_DIR} /usr/local/include/opencv)
LINK_DIRECTORIES(${ARPACK_LINK_DIR} /usr/local/lib)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3") #-fno-omit-frame-pointer")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -O3") #-fno-omit-frame-pointer")

# source files of library "LineMatchingLib" to be created
SET(LineMatchingLib_SRCS
    linedetection.cpp
    linedetector.cpp
)
# header files to be installed
SET(LineMatchingLib_HEADER
    linedetector.h
    system.h
)

ADD_LIBRARY(LineMatchingLib
            ${LineMatchingLib_SRCS}
            ${LineMatchingLib_HEADER})
TARGET_LINK_LIBRARIES(LineMatchingLib ${OpenCV_LIBS} arpack /usr/lib/x86_64-linux-gnu/libsuperlu.so arpack++)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(lineMatching Main.cpp ) 
target_link_libraries(lineMatching LineMatchingLib ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})
