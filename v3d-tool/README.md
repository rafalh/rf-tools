
V3D Tool
========

V3D Tool converts 3D meshes in GLTF format to V3M format (.v3m file extension). Only static meshes are supported right now.
V3M format is used by Red Faction game on PC platform.

Collision spheres are supported (they are used for collisions with vehicles and other non-player objects). To make a collision sphere create a node without a mesh in the root of the object hierarchy (Blender: Add -> Empty -> Sphere). Object scale determines collision sphere radius (maximal component is used). Object name must start with "csphere_".

Usage
-----

    v3d-tool input.gltf output.v3m

Limitations
-----------

* Maximal number of vertices in a primitive is 5232.
* Maximal number of indices in a primitive is 9231 (3077 triangles).
* Maximal number of textures in a mesh is 7.
* Maximal length of node name is 23 characters (ASCII).
* Maximal length of texture file name is 31 characters (ASCII).
* All nodes with meshes attached are exported as submeshes in V3D.
* LOD (level of detail) meshes are not supported.
* Only direct node transformations are applied to the mesh. Node hierarchy is completly ignored by this tool.
* Base color texture is used as diffuse map. Other maps are not supported (V3D limitation).
* For emissive materials only maximal value (channel) of RGB factor is used (e.g. if emissive factor is #FF0000 resulting V3D will have full emission).
* Double sided material property is supported. If not enabled back-face culling is used for V3D rendering.
* Child nodes without mesh attached are exported as prop points (e.g. for glares)
