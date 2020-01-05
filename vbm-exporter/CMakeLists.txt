file(GLOB SRCS *.cpp *.h)
add_executable(vbm-exporter ${SRCS})
include_directories(
    ../common
    ../vendor/rf-reversed
)
set_property(TARGET vbm-exporter PROPERTY CXX_STANDARD 17)
