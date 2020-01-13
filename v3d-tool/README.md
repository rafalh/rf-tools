
V3D Tool
========

V3D Tool converts 3D meshes in GLTF format to V3D format (.v3m file extension). Only static meshes are supported right now.
V3D format is used by Red Faction game.

Usage
-----

    v3d-tool input.gltf output.v3m

Limitations
-----------

* Maximal number of vertices in a primitive is 5232.
* Maximal number of indices in a primitive is 9232.
* Maximal number of textures in a mesh is 7.
* All nodes with meshes attached are exported as submeshes in V3D.
* LOD (level of detail) meshes are not supported.
* Only direct node transformations are applied to the mesh. Node hierarchy is completly ignored by this tool.
* Base color texture is used as diffuse map. Other maps are not supported (V3D limitation).
* For emissive materials only maximal value (channel) of RGB factor is used (e.g. if emissive factor is #FF0000 resulting V3D will have full emission).
* Double sided material property is supported. If not enabled back-face culling is used for V3D rendering.

Building
--------

You need Rust installed to build this project.

To build this project run:

    cargo build

To cross-compile for Windows (e.g. from Linux):

    cargo build --release --target=i686-pc-windows-gnu

Note: you may need additional packages (e.g. mingw-w64 in Ubuntu).
