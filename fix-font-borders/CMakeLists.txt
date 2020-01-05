file(GLOB SRCS *.cpp *.h)
add_executable(fix-font-borders ${SRCS})
include_directories(../common)
set_property(TARGET fix-font-borders PROPERTY CXX_STANDARD 17)
