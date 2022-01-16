
VMesh
=====

VMesh tool converts 3D meshes in GLTF format to V3M (static mesh) and V3C (character mesh) formats.
V3M and V3C formats are used by Red Faction game on PC platform.

Collision spheres
-----------------
Collision spheres are used for collisions with vehicles and other non-player objects. To make a collision sphere create
a top-level node without a mesh (Blender: Add -> Empty -> Sphere). Object scale determines
collision sphere radius (axis with maximal value is used). Object name must start with the string "csphere_".
Collision sphere can be parented to a joint/bone node in case of a character mesh.

Level of detail (LOD)
---------------------
Create the most detailed mesh as root level node and create less detailed meshes as its children.
Add custom property `LOD_distance` to child nodes and set it to the minimal distance at
which mesh should be rendered in game units (meters). Parent mesh (the most detailed one) has implicit distance of 0.
Child meshes should not use any transformations relative to the parent.
Be aware that Blender plugin by default does not export custom properties. You must enable them in the export options.
Keep in mind that RF uses the least detailed mesh for detection of collisions with player character.

Character
---------
If GLTF file contains a skin tool exports a character mesh (V3C). Only one skin is allowed.

When working with Blender please note that mesh object should not be parented to armature object.
Blender does it automatically when assigning automatic vertex weights so it may be necessary to manually
unparent after this operation.

All animations contained in GLTF file are exported as RFA files with names based on animation name.

Every animation has ramp in and ramp out times. They determine how animation is blended with other animations after start and before end. The tool generates those times based on animation name but user can overwrite them by `ramp_in_time.<animation name>` and `ramp_out_time.<animation name>` extras (custom properties) in `root` joint (bone). Value is specified in seconds.
Good starting value is `0.1`.

Every joint (bone) has animation specific weight that determines how animation of that specific joint blends with other
animations. Weight is in 0-10 range. Weight can be defined in `weight.<animation name>` extra (custom property) in
joint (bone) node. Weights are especially important in action animations because they are always mixed with state
animations. Weight of 10 removes state animation influence on the bone.

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
