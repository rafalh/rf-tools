file(GLOB SRCS *.cpp *.h)
add_executable(vf-exporter ${SRCS})
include_directories(
    ../common
    ../vendor/rf-reversed
)
set_property(TARGET vf-exporter PROPERTY CXX_STANDARD 17)
