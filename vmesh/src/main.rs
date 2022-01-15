mod import;
mod io_utils;
mod math_utils;
mod v3mc;
mod rfa;
mod char_anim;
mod material;

use std::io::Cursor;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Mul;
use std::vec::Vec;
use std::env;
use std::convert::TryInto;
use std::f32;
use std::iter;
use std::error::Error;
use std::path::Path;
use serde_derive::Deserialize;
use import::BufferData;
use io_utils::new_custom_error;
use material::{convert_material, create_mesh_material_ref};
use math_utils::{
    Matrix3, Matrix4, Vector3, compute_triangle_plane, generate_uv, get_vector_len, transform_normal, transform_point
};

// glTF defines -X as right, RF defines +X as right
// Both glTF and RF defines +Y as up, +Z as forward

pub(crate) fn gltf_to_rf_vec(vec: [f32; 3]) -> [f32; 3] {
    // in GLTF negative X is right, in RF positive X is right
    [-vec[0], vec[1], vec[2]]
}

pub(crate) fn gltf_to_rf_quat(quat: [f32; 4]) -> [f32; 4] {
    // convert to RF coordinate system
    // it seems RF expects inverted quaternions...
    [-quat[0], quat[1], quat[2], quat[3]]
}

fn gltf_to_rf_face(vindices: [u16; 3]) -> [u16; 3] {
    // because we convert from right-handed to left-handed order of vertices must be flipped to
    // fix backface culling
    [vindices[0], vindices[2], vindices[1]]
}

fn build_child_nodes_indices(doc: &gltf::Document) -> Vec<usize> {
    let mut child_indices: Vec<usize> = doc.nodes().flat_map(|n| n.children().map(|n| n.index())).collect();
    child_indices.dedup();
    child_indices
}

fn get_submesh_nodes(doc: &gltf::Document) -> Vec<gltf::Node> {
    let child_indices = build_child_nodes_indices(doc);
    doc.nodes().filter(|n| n.mesh().is_some() && !child_indices.contains(&n.index())).collect()
}

fn get_mesh_materials<'a>(mesh: &gltf::Mesh<'a>) -> Vec<gltf::Material<'a>> {
    let mut materials = mesh.primitives().map(|prim| prim.material())
        .collect::<Vec<_>>();
    materials.dedup_by_key(|m| m.index());
    materials
}

fn create_v3mc_file_header(lod_meshes: &[v3mc::LodMesh], cspheres: &[v3mc::ColSphere], is_character: bool) -> v3mc::FileHeader {
    v3mc::FileHeader {
        signature: if is_character { v3mc::V3C_SIGNATURE } else { v3mc::V3M_SIGNATURE },
        version: v3mc::VERSION,
        num_lod_meshes: lod_meshes.len() as i32,
        num_all_materials: lod_meshes.iter().map(|lm| lm.materials.len()).sum::<usize>() as i32,
        num_cspheres: cspheres.len() as i32,
        ..v3mc::FileHeader::default()
    }
}

fn get_primitive_vertex_count(prim: &gltf::Primitive) -> usize {
    prim.attributes().find(|p| p.0 == gltf::mesh::Semantic::Positions).map_or(0, |a| a.1.count())
}

fn count_mesh_vertices(mesh: &gltf::Mesh) -> usize {
    mesh.primitives()
        .map(|p| get_primitive_vertex_count(&p))
        .sum()
}

fn compute_mesh_bbox(mesh: &gltf::Mesh, buffers: &[BufferData], transform: &Matrix3) -> gltf::mesh::BoundingBox {
    // Note: primitive AABB from gltf cannot be used because vertices are being transformed
    if count_mesh_vertices(mesh) == 0 {
        // Mesh has no vertices so return empty AABB
        return gltf::mesh::BoundingBox {
            min: [0_f32; 3],
            max: [0_f32; 3],
        };
    }
    let mut aabb = gltf::mesh::BoundingBox {
        min: [f32::MAX; 3],
        max: [f32::MIN; 3],
    };
    // Calculate AABB manually using vertex position data
    for prim in mesh.primitives() {
        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_positions() {
            for pos in iter {
                let tpos = transform_point(&pos, transform);
                #[allow(clippy::needless_range_loop)]
                for i in 0..3 {
                    aabb.min[i] = aabb.min[i].min(tpos[i]);
                    aabb.max[i] = aabb.max[i].max(tpos[i]);
                }
            }
        }
    }
    aabb
}

fn compute_mesh_bounding_sphere_radius(mesh: &gltf::Mesh, buffers: &[BufferData], transform: &Matrix3) -> f32 {
    let mut radius = 0_f32;
    for prim in mesh.primitives() {
        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_positions() {
            for pos in iter {
                let tpos = transform_point(&pos, transform);
                let diff = [tpos[0], tpos[1], tpos[2]];
                let dist = get_vector_len(&diff);
                radius = radius.max(dist);
            }
        } else {
            panic!("mesh has no positions");
        }
    }
    radius
}

fn extract_translation_from_matrix(transform: &Matrix4) -> (Vector3, Matrix3) {
    let mut translation = [0_f32; 3];
    translation.copy_from_slice(&transform[3][0..3]);
    let mut rot_scale_mat = [[0_f32; 3]; 3];
    rot_scale_mat[0].copy_from_slice(&transform[0][0..3]);
    rot_scale_mat[1].copy_from_slice(&transform[1][0..3]);
    rot_scale_mat[2].copy_from_slice(&transform[2][0..3]);
    (translation, rot_scale_mat)
}

fn create_mesh_chunk_info(prim: &gltf::Primitive, materials: &[gltf::Material]) -> v3mc::MeshDataBlockChunkInfo {
    // write texture index in LOD model textures array
    let prim_mat = prim.material();
    let texture_index = materials.iter()
        .position(|m| m.index() == prim_mat.index())
        .expect("cannot find texture") as i32;

    v3mc::MeshDataBlockChunkInfo{
        texture_index,
    }
}

fn create_mesh_chunk_data(prim: &gltf::Primitive, buffers: &[BufferData],
    transform: &Matrix3, is_character: bool) -> v3mc::MeshChunkData {
    
    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

    let vecs: Vec<_> = reader.read_positions()
        .expect("mesh has no positions")
        .map(|pos| gltf_to_rf_vec(transform_point(&pos, transform)))
        .collect();
    let norms: Vec<_> = reader.read_normals()
        // FIXME: according to GLTF spec we should generate flat normals here
        .expect("mesh has no normals")
        .map(|norm| gltf_to_rf_vec(transform_normal(&norm, transform)))
        .collect();
    let uvs: Vec<_> = reader.read_tex_coords(0)
        .map_or_else(
            || (0..vecs.len()).map(|i| generate_uv(&vecs[i], &norms[i])).collect(),
            |iter| iter.into_f32().collect(),
        );
    let indices: Vec<u16> = reader.read_indices()
        .expect("mesh has no indices")
        .into_u32()
        .map(|vindex| TryInto::<u16>::try_into(vindex).expect("vertex index does not fit in 16 bits"))
        .collect();
    // Sanity checks
    assert!(indices.len() % 3 == 0, "number of indices is not a multiple of three: {}", indices.len());
    assert!(vecs.len() == norms.len());
    let nv = vecs.len();
    let face_flags = if prim.material().double_sided() { v3mc::MeshFace::DOUBLE_SIDED } else { 0 };

    let faces: Vec<_> = indices
        .chunks(3)
        .map(|tri| gltf_to_rf_face([tri[0], tri[1], tri[2]]))
        .map(|vindices| v3mc::MeshFace{
            vindices,
            flags: face_flags,
        })
        .collect();

    let face_planes: Vec::<_> = if is_character {
        Vec::new()
    } else {
        indices.chunks(3)
            .map(|tri| (tri[0] as usize, tri[1] as usize, tri[2] as usize))
            .map(|(i, j, k)| compute_triangle_plane(&vecs[i], &vecs[j], &vecs[k]))
            .collect()
    };

    let same_pos_vertex_offsets: Vec<i16> = (0..nv).map(|_| 0).collect();
    
    let wis: Vec<_> = if let Some(joints) = reader.read_joints(0) {
        joints.into_u16()
            .zip(reader.read_weights(0).expect("mesh has no weights").into_u8())
            .map(|(indices_u16, weights)| {
                let indices = indices_u16
                    .map(|x| x.try_into().expect("joint index should fit in u8"));
                v3mc::WeightIndexArray { weights, indices }
            })
            .collect()
    } else {
        (0..nv).map(|_| v3mc::WeightIndexArray::default()).collect()
    };

    v3mc::MeshChunkData{
        vecs,
        norms,
        uvs,
        faces,
        face_planes,
        same_pos_vertex_offsets,
        wi: wis,
    }
}

fn create_mesh_data_block(
    mesh: &gltf::Mesh, 
    buffers: &[BufferData], 
    transform: &Matrix3, 
    mesh_materials: &[gltf::Material], 
    prop_points: &[v3mc::PropPoint],
    is_character: bool
) -> v3mc::MeshDataBlock {
    v3mc::MeshDataBlock{
        chunks: mesh.primitives().map(|prim| create_mesh_chunk_info(&prim, mesh_materials)).collect(),
        chunks_data: mesh.primitives().map(|prim| create_mesh_chunk_data(&prim, buffers, transform, is_character)).collect(),
        prop_points: prop_points.to_vec(),
    }
}

fn create_mesh_chunk(prim: &gltf::Primitive) -> std::io::Result<v3mc::MeshChunk> {
    
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return Err(new_custom_error("only triangle list primitives are supported"));
    }
    if prim.indices().is_none() {
        return Err(new_custom_error("not indexed geometry is not supported"));
    }

    let index_count = prim.indices().unwrap().count();
    assert!(index_count % 3 == 0, "number of indices is not a multiple of three: {}", index_count);
    println!("Index count: {}", index_count);

    let tri_count = index_count / 3;
    let vertex_count = get_primitive_vertex_count(prim);
    println!("Vertex count: {}", vertex_count);

    if env::var("IGNORE_GEOMETRY_LIMITS").is_err() {
        let index_limit = 10000 - 768;
        if index_count > index_limit {
            return Err(new_custom_error(format!("primitive has too many indices: {} (limit {})", index_count, index_limit)));
        }

        let vertex_limit = 6000 - 768;
        if vertex_count > vertex_limit {
            return Err(new_custom_error(format!("primitive has too many vertices: {} (limit {})", vertex_count, vertex_limit)));
        }
    }

    let num_vecs = vertex_count.try_into().unwrap();
    let num_faces = tri_count.try_into().unwrap();
    let vecs_alloc = (vertex_count * 3 * 4).try_into().unwrap();
    let faces_alloc = (tri_count * 4 * 2).try_into().unwrap();
    let same_pos_vertex_offsets_alloc = (vertex_count * 2).try_into().unwrap();
    let wi_alloc = (vertex_count * 2 * 4).try_into().unwrap();
    let uvs_alloc = (vertex_count * 2 * 4).try_into().unwrap();
    let render_mode = material::compute_render_mode_for_material(&prim.material());
    Ok(v3mc::MeshChunk{
        num_vecs,
        num_faces,
        vecs_alloc,
        faces_alloc,
        same_pos_vertex_offsets_alloc,
        wi_alloc,
        uvs_alloc,
        render_mode,
    })
}

fn convert_mesh(
    node: &gltf::Node, 
    buffers: &[BufferData], 
    lod_mesh_materials: &[gltf::Material],
    prop_points: &[v3mc::PropPoint],
    transform: &Matrix3,
    is_character: bool
) -> std::io::Result<v3mc::Mesh> {

    let mesh = node.mesh().unwrap();
    let flags = if is_character { v3mc::VIF_MESH_FLAG_CHARACTER } else { v3mc::VIF_MESH_FLAG_FACE_PLANES };
    let num_vecs = count_mesh_vertices(&mesh) as i32;

    let materials: Vec<_> = get_mesh_materials(&mesh);
    if materials.len() > v3mc::Mesh::MAX_TEXTURES {
        return Err(new_custom_error(format!("found {} materials in a submesh but only {} are allowed",
        materials.len(), v3mc::Mesh::MAX_TEXTURES)));
    }

    let mut chunks = Vec::new();
    for prim in mesh.primitives() {
        chunks.push(create_mesh_chunk(&prim)?);
    }

    let mut data_block_cur = Cursor::new(Vec::<u8>::new());
    create_mesh_data_block(&mesh, buffers, transform, &materials, prop_points, is_character)
        .write(&mut data_block_cur)?;
    let data_block: Vec::<u8> = data_block_cur.into_inner();
    
    let num_prop_points = prop_points.len() as i32;
    let tex_refs: Vec<_> = materials.iter()
        .map(|m| create_mesh_material_ref(m, lod_mesh_materials))
        .collect();

    Ok(v3mc::Mesh{
        flags,
        num_vecs,
        chunks,
        data_block,
        num_prop_points,
        textures: tex_refs,
    })
}

#[derive(Deserialize, Debug, Default)]
struct NodeExtras {
    #[serde(rename = "LOD_distance")]
    lod_distance: Option<f32>,
}

fn get_node_extras(node: &gltf::Node) -> NodeExtras {
    node.extras().as_ref()
        .and_then(|raw| serde_json::from_str::<NodeExtras>(raw.get()).ok())
        .unwrap_or_default()
}

fn find_lod_nodes<'a>(node: &'a gltf::Node) -> Vec<(gltf::Node<'a>, f32)> {
    let mut child_node_dist_vec: Vec<(gltf::Node, f32)> = node.children()
        .filter(|n| n.mesh().is_some())
        .map(|n| {
            let dist_opt = get_node_extras(&n).lod_distance;
            (n, dist_opt)
        })
        .filter_map(|(n, dist_opt)| {
            if dist_opt.is_none() {
                eprintln!("Warning! Expected LOD_distance in child node {}", n.name().unwrap_or("None"));
            }
            dist_opt.map(|d| (n, d))
        })
        .chain(iter::once((node.clone(), 0_f32)))
        .collect();
    child_node_dist_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    child_node_dist_vec
}

fn get_node_local_transform(node: &gltf::Node) -> glam::Mat4 {
    glam::Mat4::from_cols_array_2d(&node.transform().matrix())
}

fn convert_prop_point(node: &gltf::Node, transform: &glam::Mat4, parent_index: i32) -> v3mc::PropPoint {
    let local_transform = get_node_local_transform(node);
    let (_scale, rotation, translation) = transform.mul(local_transform)
        .to_scale_rotation_translation();

    v3mc::PropPoint{
        name: node.name().expect("prop point name is missing").to_string(),
        orient: gltf_to_rf_quat(rotation.into()),
        pos: gltf_to_rf_vec(translation.into()),
        parent_index,
    }
}

fn is_joint(node: &gltf::Node, skin: &gltf::Skin) -> bool {
    skin.joints().any(|joint| node.index() == joint.index())
}

fn get_prop_points(parent: &gltf::Node, transform: &glam::Mat4) -> Vec<v3mc::PropPoint> {
    let mut prop_points = parent.children()
        .filter(|n| n.mesh().is_none())
        .map(|n| convert_prop_point(&n, transform, -1))
        .collect::<Vec<_>>();
    if let Some(skin) = parent.skin() {
        prop_points.extend(skin.joints()
            .flat_map(|joint| joint.children().map(move |n| (n, joint.index())))
            .filter(|(node, _)| !is_joint(node, &skin))
            .filter(|(node, _)| node.mesh().is_none())
            .filter(|(node, _)| node.name().is_some())
            .map(|(node, parent_index)|
                convert_prop_point(&node, &glam::Mat4::IDENTITY, parent_index as i32)
            )
        );
    }
    prop_points
}

fn is_csphere(node: &gltf::Node) -> bool {
    node.mesh().is_none() && node.name().unwrap_or_default().starts_with("csphere_")
}

fn get_cspheres(doc: &gltf::Document) -> Vec<v3mc::ColSphere> {
    let mut cspheres = doc.nodes()
        .filter(is_csphere)
        .map(|n| convert_csphere(&n, -1))
        .collect::<Vec<_>>();
    if let Some(skin) = doc.skins().next() {
        cspheres.extend(skin.joints()
            .flat_map(|joint| joint.children().map(move |n| (n, joint.index())))
            .filter(|(node, _)| !is_joint(node, &skin))
            .filter(|(node, _)| is_csphere(node))
            .filter(|(node, _)| node.name().is_some())
            .map(|(node, parent_index)|
                convert_csphere(&node, parent_index as i32)
            )
        );
    }
    cspheres
}

fn convert_csphere(node: &gltf::Node, parent_index: i32) -> v3mc::ColSphere {
    let name = node.name().expect("csphere name is missing").to_owned();
    let transform = get_node_local_transform(node);
    let (scale, _rotation, translation) = transform.to_scale_rotation_translation();
    let radius = scale.max_element();
    v3mc::ColSphere{
        name,
        parent_index,
        pos: gltf_to_rf_vec(translation.into()),
        radius,
    }
}

fn convert_lod_mesh(node: &gltf::Node, buffers: &[BufferData], is_character: bool) -> Result<v3mc::LodMesh, Box<dyn Error>> {
    let node_transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix()).to_cols_array_2d();

    let mesh = node.mesh().unwrap();
    let name = node.name().unwrap_or("Default").to_string();
    println!("Processing top-level node {} ({})", node.index(), name);

    let parent_name = "None".to_string();
    let version = v3mc::MeshDataBlock::VERSION;
    let child_node_dist_vec = find_lod_nodes(node);
    let distances = child_node_dist_vec.iter().map(|(_, dist)| *dist).collect();
    let (origin, rot_scale_mat) = extract_translation_from_matrix(&node_transform);

    let bbox = compute_mesh_bbox(&mesh, buffers, &rot_scale_mat);
    let (bbox_min, bbox_max) = (
        gltf_to_rf_vec(bbox.min),
        gltf_to_rf_vec(bbox.max),
    );

    let offset = gltf_to_rf_vec(origin);
    let radius = compute_mesh_bounding_sphere_radius(&mesh, buffers, &rot_scale_mat);

    let transform = glam::Mat4::from_mat3(glam::Mat3::from_cols_array_2d(&rot_scale_mat));
    let prop_points = get_prop_points(node, &transform);

    let mut gltf_materials: Vec<_> = child_node_dist_vec.iter()
        .flat_map(|(n, _)| get_mesh_materials(&n.mesh().unwrap()))
        .collect(); 
    gltf_materials.dedup_by_key(|m| m.index());
    let materials: Vec<_> = gltf_materials.iter().map(convert_material).collect();

    let mut meshes: Vec<_> = Vec::with_capacity(child_node_dist_vec.len());
    for (n, d) in &child_node_dist_vec {
        println!("Processing node {} name {} distance {}", n.index(), n.name().unwrap_or("<unnamed>"), d);
        meshes.push(convert_mesh(n, buffers, &gltf_materials, &prop_points, &rot_scale_mat, is_character)?);
    }

    Ok(v3mc::LodMesh{
        name,
        parent_name,
        version,
        distances,
        offset,
        radius,
        bbox_min,
        bbox_max,
        meshes,
        materials,
    })
}

fn make_v3mc_file(doc: &gltf::Document, buffers: &[BufferData], is_character: bool) -> Result<v3mc::File, Box<dyn Error>> {
    if doc.skins().count() > 1 {
        eprintln!("Warning! There is more than one skin defined. Only first skin will be used.");
    }

    let submesh_nodes = get_submesh_nodes(doc);
    println!("Found {} top-level mesh nodes", submesh_nodes.len());
    let mut lod_meshes = Vec::with_capacity(submesh_nodes.len());
    for n in &submesh_nodes {
        lod_meshes.push(convert_lod_mesh(n, buffers, is_character)?);
    }

    let cspheres = get_cspheres(doc);

    let bones = if let Some(skin) = doc.skins().next() {
        char_anim::convert_bones(&skin, buffers)?
    } else {
        Vec::new()
    };

    Ok(v3mc::File{
        header: create_v3mc_file_header(&lod_meshes, &cspheres, is_character),
        lod_meshes,
        cspheres,
        bones,
    })
}

fn generate_output_file_name(input_file_name: &str, output_file_name_opt: Option<&str>, is_character: bool) -> String {
    output_file_name_opt
        .map_or_else(|| {
            let base_file_name = input_file_name.strip_suffix(".gltf")
            .unwrap_or_else(|| input_file_name.strip_suffix(".glb").unwrap_or(input_file_name));
            let ext = if is_character { "v3c" } else { "v3m" };
            format!("{}.{}", base_file_name, ext)
        }, &str::to_owned)
}

fn convert_gltf_to_v3mc(input_file_name: &str, output_file_name_opt: Option<&str>) -> Result<(), Box<dyn Error>> {
    println!("Importing GLTF file {}...", input_file_name);
    let input_path = Path::new(input_file_name);
    let gltf = gltf::Gltf::open(input_path)?;
    let gltf::Gltf { document, blob } = gltf;

    println!("Importing GLTF buffers...");
    let buffers = import::import_buffer_data(&document, input_path.parent(), blob)?;
    
    println!("Converting...");
    let skin_opt = document.skins().next();
    let is_character = skin_opt.is_some();
    let output_file_name = generate_output_file_name(input_file_name, output_file_name_opt, is_character);
    let v3m = make_v3mc_file(&document, &buffers, is_character)?;
    let file = File::create(&output_file_name)?;
    let mut wrt = BufWriter::new(file);
    v3m.write(&mut wrt)?;

    let output_dir = Path::new(&output_file_name).parent().unwrap();
    if let Some(skin) = skin_opt {
        for (i, anim) in document.animations().enumerate() {
            char_anim::convert_animation_to_rfa(&anim, i, &skin, &buffers, output_dir)?;
        }
    }

    println!("Converted successfully.");
    Ok(())
}

fn main() {
    println!("GLTF to V3M/V3C converter {} by Rafalh", env!("CARGO_PKG_VERSION"));

    let mut args = env::args();
    let app_name = args.next().unwrap();
    if env::args().len() < 2 {
        println!("Usage: {} input_file_name.gltf [output_file_name.v3m]", app_name);
        std::process::exit(1);
    }

    let input_file_name = args.next().unwrap();
    let output_file_name = args.next();

    if let Err(e) = convert_gltf_to_v3mc(&input_file_name, output_file_name.as_deref()) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
