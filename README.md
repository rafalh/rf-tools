RF Tools
========

Tools for Red Faction game:

* vmesh - converts GLTF to V3M files (3D models)
* vbm-exporter - exports content of VBM files into series of TGA images
* vf-exporter - exports content of VF file (font) into TGA image
* vpp-exporter - unpacks packfiles (files with `.vpp` extension)
* makevbm - creates VBM files from a series of images
* vpp - creates or extracts packfiles (files with `.vpp` extension)
* vsound - converts VSE/VMU files (used by RF in PS2 version) to WAV
* peg - extracts bitmaps from PEG files (used by RF in PS2 version)

All provided tools use command line interface.
Use them on your own risk.

Build
-----

Tools written in C++ use CMake for building.

Windows:

Generate project files in CMake and build using your favorite IDE.

Ubuntu (MinGW):

    # Install MinGW
    sudo apt install mingw-w64
    # Configure build
    mkdir -p build
    cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/mingw-ubuntu.cmake -DCMAKE_BUILD_TYPE=Release
    # Build
    make -j8

Tools written in Rust use Cargo for building.

You need Rust installed.

Build from repository root directory:

    cargo build --release

To cross-compile for Windows (e.g. from Linux):

    cargo build --release --target=i686-pc-windows-gnu

Note: you may need additional packages (e.g. `mingw-w64` in Ubuntu).


License
-------
The GPL-3 license. See LICENSE.txt.
