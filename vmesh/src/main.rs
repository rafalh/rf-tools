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
use std::path::PathBuf;
use std::vec::Vec;
use std::env;
use std::convert::TryInto;
use std::f32;
use std::iter;
use std::error::Error;
use std::path::Path;
use gltf::Buffer;
use serde_derive::Deserialize;
use clap::Parser;
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

fn compute_mesh_bbox(mesh: &gltf::Mesh, transform: &Matrix3, ctx: &Context) -> gltf::mesh::BoundingBox {
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
        let reader = prim.reader(|buffer| ctx.get_buffer_data(buffer));
        if let Some(iter) = reader.read_positions() {
            for pos in iter {
                let tpos = gltf_to_rf_vec(transform_point(&pos, transform));
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

fn compute_mesh_bounding_sphere_radius(mesh: &gltf::Mesh, transform: &Matrix3, ctx: &Context) -> f32 {
    let mut radius = 0_f32;
    for prim in mesh.primitives() {
        let reader = prim.reader(|buffer| ctx.get_buffer_data(buffer));
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

fn create_mesh_chunk_data(prim: &gltf::Primitive, transform: &Matrix3, ctx: &Context) -> v3mc::MeshChunkData {
    
    let reader = prim.reader(|buffer| ctx.get_buffer_data(buffer));

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

    let face_planes: Vec::<_> = if ctx.is_character {
        Vec::new()
    } else {
        faces.iter()
            .map(|face| face.vindices.map(usize::from))
            .map(|[i, j, k]| compute_triangle_plane(&vecs[i], &vecs[j], &vecs[k]))
            .collect()
    };

    let same_pos_vertex_offsets: Vec<i16> = vec![0; nv];
    
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
        vec![v3mc::WeightIndexArray::default(); nv]
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
    transform: &Matrix3, 
    mesh_materials: &[gltf::Material],
    prop_points: &[v3mc::PropPoint],
    ctx: &Context
) -> v3mc::MeshDataBlock {
    v3mc::MeshDataBlock{
        chunks: mesh.primitives().map(|prim| create_mesh_chunk_info(&prim, mesh_materials)).collect(),
        chunks_data: mesh.primitives().map(|prim| create_mesh_chunk_data(&prim, transform, ctx)).collect(),
        prop_points: prop_points.to_vec(),
    }
}

fn create_mesh_chunk(prim: &gltf::Primitive, index: usize) -> std::io::Result<v3mc::MeshChunk> {
    
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return Err(new_custom_error("only triangle list primitives are supported"));
    }
    if prim.indices().is_none() {
        return Err(new_custom_error("not indexed geometry is not supported"));
    }

    let vertex_count = get_primitive_vertex_count(prim);
    let vertex_limit = 6000 - 768;

    let index_count = prim.indices().unwrap().count();
    let index_limit = 10000 - 768;
    assert!(index_count % 3 == 0, "number of indices is not a multiple of three: {}", index_count);
    let tri_count = index_count / 3;

    println!("Primitive #{}: vertices {}/{}, indices {}/{}", index, vertex_count, vertex_limit, index_count, index_limit);

    if env::var("IGNORE_GEOMETRY_LIMITS").is_err() {
        if vertex_count > vertex_limit {
            return Err(new_custom_error(format!("primitive has too many vertices: {} (limit {})", vertex_count, vertex_limit)));
        }
        if index_count > index_limit {
            return Err(new_custom_error(format!("primitive has too many indices: {} (limit {})", index_count, index_limit)));
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
    lod_mesh_materials: &[gltf::Material],
    prop_points: &[v3mc::PropPoint],
    transform: &Matrix3,
    ctx: &Context
) -> std::io::Result<v3mc::Mesh> {

    let mesh = node.mesh().unwrap();
    let flags = if ctx.is_character { v3mc::VIF_MESH_FLAG_CHARACTER } else { v3mc::VIF_MESH_FLAG_FACE_PLANES };
    let num_vecs = count_mesh_vertices(&mesh) as i32;

    let materials: Vec<_> = get_mesh_materials(&mesh);
    if materials.len() > v3mc::Mesh::MAX_TEXTURES {
        return Err(new_custom_error(format!("found {} materials in a submesh but only {} are allowed",
        materials.len(), v3mc::Mesh::MAX_TEXTURES)));
    }

    let mut chunks = Vec::new();
    for (i, prim) in mesh.primitives().enumerate() {
        chunks.push(create_mesh_chunk(&prim, i)?);
    }

    let mut data_block_cur = Cursor::new(Vec::<u8>::new());
    create_mesh_data_block(&mesh, transform, &materials, prop_points, ctx)
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

pub(crate) fn get_node_extras<'a, T: serde::Deserialize<'a> + Default>(node: &'a gltf::Node) -> T {
    node.extras().as_ref()
        .and_then(|raw| serde_json::from_str::<T>(raw.get()).ok())
        .unwrap_or_default()
}

fn find_lod_nodes<'a>(node: &'a gltf::Node) -> Vec<(gltf::Node<'a>, f32)> {
    let mut child_node_dist_vec: Vec<(gltf::Node, f32)> = node.children()
        .filter(|n| n.mesh().is_some())
        .map(|n| {
            let dist_opt = get_node_extras::<NodeExtras>(&n).lod_distance;
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

fn get_prop_points(parent: &gltf::Node, transform: &glam::Mat4) -> Vec<v3mc::PropPoint> {
    let mut prop_points = parent.children()
        .filter(|n| n.mesh().is_none())
        .map(|n| convert_prop_point(&n, transform, -1))
        .collect::<Vec<_>>();
    if let Some(skin) = parent.skin() {
        prop_points.extend(char_anim::get_nodes_parented_to_bones(&skin)
            .filter(|(node, _)| node.mesh().is_none())
            .filter(|(node, _)| node.name().is_some())
            .map(|(node, parent_index)|
                convert_prop_point(&node, &glam::Mat4::IDENTITY, parent_index)
            )
        );
    }
    println!("Found {} prop points", prop_points.len());
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
        cspheres.extend(char_anim::get_nodes_parented_to_bones(&skin)
            .filter(|(node, _)| is_csphere(node))
            .filter(|(node, _)| node.name().is_some())
            .map(|(node, parent_index)|
                convert_csphere(&node, parent_index)
            )
        );
    }
    println!("Found {} cspheres", cspheres.len());
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

fn convert_lod_mesh(node: &gltf::Node, ctx: &Context) -> Result<v3mc::LodMesh, Box<dyn Error>> {
    let node_transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix()).to_cols_array_2d();

    let mesh = node.mesh().unwrap();
    let name = node.name().unwrap_or("Default").to_string();
    println!("Processing LOD group: node #{} '{}'", node.index(), name);

    let parent_name = "None".to_string();
    let version = v3mc::MeshDataBlock::VERSION;
    let child_node_dist_vec = find_lod_nodes(node);
    let distances = child_node_dist_vec.iter().map(|(_, dist)| *dist).collect();
    let (origin, rot_scale_mat) = extract_translation_from_matrix(&node_transform);

    let bbox = compute_mesh_bbox(&mesh, &rot_scale_mat, ctx);
    let (bbox_min, bbox_max) = (bbox.min, bbox.max);

    let offset = gltf_to_rf_vec(origin);
    let radius = compute_mesh_bounding_sphere_radius(&mesh, &rot_scale_mat, ctx);

    let transform = glam::Mat4::from_mat3(glam::Mat3::from_cols_array_2d(&rot_scale_mat));
    let prop_points = get_prop_points(node, &transform);

    let mut gltf_materials: Vec<_> = child_node_dist_vec.iter()
        .flat_map(|(n, _)| get_mesh_materials(&n.mesh().unwrap()))
        .collect(); 
    gltf_materials.dedup_by_key(|m| m.index());
    let materials: Vec<_> = gltf_materials.iter().map(convert_material).collect();

    let mut meshes: Vec<_> = Vec::with_capacity(child_node_dist_vec.len());
    for (i, (n, d)) in child_node_dist_vec.iter().enumerate() {
        println!("Processing LOD{} mesh: node #{} '{}', distance {}", i, n.index(), n.name().unwrap_or("<unnamed>"), d);
        meshes.push(convert_mesh(n, &gltf_materials, &prop_points, &rot_scale_mat, ctx)?);
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

fn make_v3mc_file(doc: &gltf::Document, ctx: &Context) -> Result<v3mc::File, Box<dyn Error>> {
    if doc.skins().count() > 1 {
        eprintln!("Warning! There is more than one skin defined. Only first skin will be used.");
    }

    let submesh_nodes = get_submesh_nodes(doc);
    let mut lod_meshes = Vec::with_capacity(submesh_nodes.len());
    for n in &submesh_nodes {
        lod_meshes.push(convert_lod_mesh(n, ctx)?);
    }

    let cspheres = get_cspheres(doc);

    let bones = if let Some(skin) = doc.skins().next() {
        char_anim::convert_bones(&skin, ctx)?
    } else {
        Vec::new()
    };

    Ok(v3mc::File{
        header: create_v3mc_file_header(&lod_meshes, &cspheres, ctx.is_character),
        lod_meshes,
        cspheres,
        bones,
    })
}

struct Context {
    buffers: Vec<BufferData>,
    is_character: bool,
    args: Args,
    output_dir: PathBuf,
}

impl Context {
    fn get_buffer_data(&self, buffer: Buffer) -> Option<&[u8]> {
        Some(&*self.buffers[buffer.index()])
    }
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

fn convert_gltf_to_v3mc(args: Args) -> Result<(), Box<dyn Error>> {
    println!("Importing GLTF file: {}", args.input_file);
    let input_path = Path::new(&args.input_file);
    let gltf = gltf::Gltf::open(input_path)?;
    let gltf::Gltf { document, blob } = gltf;

    println!("Importing GLTF buffers");
    let buffers = import::import_buffer_data(&document, input_path.parent(), blob)?;
    let skin_opt = document.skins().next();
    let is_character = skin_opt.is_some();
    let output_file_name = generate_output_file_name(&args.input_file, args.output_file.as_deref(), is_character);
    let output_dir = Path::new(&output_file_name).parent().unwrap().to_owned();

    println!("Exporting mesh: {}", output_file_name);
    let ctx = Context { buffers, is_character, args, output_dir };
    let v3m = make_v3mc_file(&document, &ctx)?;
    let file = File::create(&output_file_name)?;
    let mut wrt = BufWriter::new(file);
    v3m.write(&mut wrt)?;

    if let Some(skin) = skin_opt {
        for (i, anim) in document.animations().enumerate() {
            char_anim::convert_animation_to_rfa(&anim, i, &skin, &ctx)?;
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Input GLTF filename
    input_file: String,

    /// Input GLTF file
    output_file: Option<String>,

    /// Default animation weight to be used when it is not defined in bone extras.
    /// Default is 10 if bone is animated, 2 otherwise
    #[clap(short = 'w', long)]
    anim_weight: Option<f32>,
}

fn main() {
    let args = Args::parse();

    println!("GLTF to V3M/V3C converter {} by Rafalh", env!("CARGO_PKG_VERSION"));

    if let Err(e) = convert_gltf_to_v3mc(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
