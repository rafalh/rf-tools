RF Tools
========

Tools for Red Faction game:

* v3d-tool - generates V3M files based on GLTF (3D models)
* vbm-exporter - exports content of VBM files into series of TGA images
* vf-exporter - exports content of VF file (font) into TGA image
* vpp-exporter - unpacks packfiles (.vpp files)
* makevbm - creates VBM files from a series of images

All provided tools use command line interface.
Use them on your own risk.

Build
-----

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

**v3d-tool** and **makevbm** are Rust applications - they have to be built differently. Check README in v3d-tool directory.

License
-------
The GPL-3 license. See LICENSE.txt.
