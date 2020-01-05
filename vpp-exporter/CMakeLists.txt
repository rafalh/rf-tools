file(GLOB SRCS *.cpp *.h)
add_executable(vpp-exporter ${SRCS})
include_directories (../vendor/rf-reversed)
set_property(TARGET vbm-exporter PROPERTY CXX_STANDARD 17)
