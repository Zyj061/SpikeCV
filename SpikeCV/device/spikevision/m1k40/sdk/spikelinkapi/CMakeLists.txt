cmake_minimum_required(VERSION 3.0.0)
project(spikelinkapi VERSION 0.1.0)

message( STATUS "compiling " ${PROJECT_NAME})
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    message( STATUS "the system is Linux ... O.K.")
else()
    message( STATUS "the system is: ${CMAKE_SYSTEM_NAME} ... NOT O.K. !!! " )
 #   message(FATAL_ERROR "only linux is supported")
endif()

set(CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SHARED_LIB_SOURCES
    spikelinkapi.cpp 
    spikelinkinput.cpp 
    refcount.cpp)

set( EXE_SOURCES
    main.cpp refcount.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
add_definitions(-D_HAS_STD_BYTE=0)
add_definitions(-DSV_EXPORTS)
set(ROOT_DIR ${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../JR6101_v2/include)

add_library(spikelinkapi SHARED ${SHARED_LIB_SOURCES})
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    TARGET_LINK_LIBRARIES(spikelinkapi pthread stdc++fs dl)
endif()


add_executable(spikelinkapitest ${EXE_SOURCES})
TARGET_LINK_LIBRARIES(spikelinkapitest spikelinkapi)
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    TARGET_LINK_LIBRARIES(spikelinkapitest pthread stdc++fs dl) 
endif()
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../lib) 