RF Tools
========

Tools for Red Faction game:

* v3d-tool - generates V3M files based on GLTF (3D models)
* vbm-exporter - exports content of VBM files into series of TGA images
* vf-exporter - exports content of VF file (font) into TGA image
* vpp-exporter - unpacks packfiles (files with `.vpp` extension)
* makevbm - creates VBM files from a series of images
* vpp - creates or extracts packfiles (files with `.vpp` extension)

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

**v3d-tool**, **makevbm**, **vpp** are Rust applications - they have to be built separately.

You need Rust installed.

To build a single project run following command in project directory:

    cargo build

To cross-compile for Windows (e.g. from Linux):

    cargo build --release --target=i686-pc-windows-gnu

Note: you may need additional packages (e.g. mingw-w64 in Ubuntu).


License
-------
The GPL-3 license. See LICENSE.txt.
