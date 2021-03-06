
VMesh Tool
==========

VMesh Tool converts 3D meshes in GLTF format to V3M format (.v3m file extension). Only static meshes are supported right now.
V3M format is used by Red Faction game on PC platform.

Collision spheres
-----------------
Collision spheres are used for collisions with vehicles and other non-player objects. To make a collision sphere create
a node without a mesh in the root of the object hierarchy (Blender: Add -> Empty -> Sphere). Object scale determines
collision sphere radius (axis with maximal value is used). Object name must start with the string "csphere_".

Level of detail (LOD)
---------------------
Create the most detailed mesh as root level node and create less detailed meshes as its children.
Add custom property `LOD_distance` to child nodes and set it to the minimal distance at
which mesh should be rendered in game units (meters). Parent mesh (the most detailed one) has implicit distance of 0.
Child meshes should not use any transformations relative to the parent.
Be aware that Blender plugin by default does not export custom properties. You must enable them in the export options.
Keep in mind that RF uses the least detailed mesh for detection of collisions with player character.

Usage
-----

    vmesh input.gltf output.v3m

Limitations
-----------

* Maximal number of vertices in a primitive is 5232.
* Maximal number of indices in a primitive is 9231 (3077 triangles).
* Maximal number of textures in a mesh is 7.
* Maximal length of node name is 23 characters (ASCII).
* Maximal length of texture file name is 31 characters (ASCII).
* All nodes with meshes attached are exported as submeshes in V3M.
* Only direct node transformations are applied to the mesh. Node hierarchy is completly ignored by this tool
  (except for LOD meshes).
* Base color texture is used as diffuse map. Other maps are not supported (V3M limitation).
* For emissive materials only maximal value (channel) of RGB factor is used (e.g. if emissive factor is #FF0000
  converted mesh will have full emission).
* Double sided material property is supported. If not enabled back-face culling is used for V3M rendering.
* Child nodes without mesh attached are exported as prop points (e.g. for glares).
* Child nodes with meshes are exported as LOD levels and should have `LOD_distance` user property (see above).
