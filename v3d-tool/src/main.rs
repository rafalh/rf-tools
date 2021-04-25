
mod import;
mod io_utils;
mod math_utils;
mod v3mc;

use std::io::{prelude::*, Cursor};
use std::fs::File;
use std::io::BufWriter;
use std::vec::Vec;
use std::env;
use std::convert::TryInto;
use std::f32;
use gltf;
use import::BufferData;
use io_utils::new_custom_error;
use math_utils::*;

fn get_submesh_nodes(doc: &gltf::Document) -> impl Iterator<Item = gltf::Node> {
    doc.nodes().filter(|n| n.mesh().is_some())
}

fn get_csphere_nodes(doc: &gltf::Document) -> impl Iterator<Item = gltf::Node> {
    doc.nodes().filter(|n| n.mesh().is_none()).filter(|n| n.name().unwrap_or("").starts_with("csphere_"))
}

fn get_prop_point_nodes<'a>(parent: &gltf::Node<'a>) -> impl Iterator<Item = gltf::Node<'a>> {
    parent.children().filter(|n| n.mesh().is_none() && n.name().is_some())
}

fn get_submesh_textures(node: &gltf::Node) -> Vec<String> {
    let mesh = node.mesh().unwrap();
    let mut textures = mesh.primitives()
        .map(|prim| get_material_base_color_texture_name(&prim.material()))
        .collect::<Vec<_>>();
    textures.sort();
    textures.dedup();
    textures
}

fn create_v3m_file_header(lod_meshes: &Vec<v3mc::LodMesh>, cspheres: &Vec<v3mc::ColSphere>) -> v3mc::FileHeader {
    v3mc::FileHeader {
        signature: v3mc::V3M_SIGNATURE,
        version: v3mc::VERSION,
        num_lod_meshes: lod_meshes.len() as i32,
        num_all_materials: lod_meshes.into_iter().map(|lm| lm.materials.len()).sum::<usize>() as i32,
        num_cspheres: cspheres.len() as i32,
        ..v3mc::FileHeader::default()
    }
}

fn get_primitive_vertex_count(prim: &gltf::Primitive) -> usize {
    prim.attributes().find(|p| p.0 == gltf::mesh::Semantic::Positions).map(|a| a.1.count()).unwrap_or(0)
}

fn count_mesh_vertices(mesh: &gltf::Mesh) -> usize {
    mesh.primitives()
        .map(|p| get_primitive_vertex_count(&p))
        .sum()
}

fn compute_mesh_bbox(mesh: &gltf::Mesh, buffers: &Vec<BufferData>, transform: &Matrix3) -> gltf::mesh::BoundingBox {
    // Note: primitive AABB from gltf cannot be used because vertices are being transformed
    if count_mesh_vertices(mesh) == 0 {
        // Mesh has no vertices so return empty AABB
        return gltf::mesh::BoundingBox {
            min: [0f32; 3],
            max: [0f32; 3],
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
                for i in 0..3 {
                    aabb.min[i] = aabb.min[i].min(tpos[i]);
                    aabb.max[i] = aabb.max[i].max(tpos[i]);
                }
            }
        }
    }
    aabb
}

fn compute_mesh_bounding_sphere_radius(mesh: &gltf::Mesh, buffers: &Vec<BufferData>, transform: &Matrix3) -> f32 {
    let mut radius = 0f32;
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
    let mut translation = [0f32; 3];
    translation.copy_from_slice(&transform[3][0..3]);
    let mut rot_scale_mat = [[0f32; 3]; 3];
    rot_scale_mat[0].copy_from_slice(&transform[0][0..3]);
    rot_scale_mat[1].copy_from_slice(&transform[1][0..3]);
    rot_scale_mat[2].copy_from_slice(&transform[2][0..3]);
    (translation, rot_scale_mat)
}

fn create_mesh_chunk_info(prim: &gltf::Primitive, textures: &Vec::<String>) -> v3mc::MeshDataBlockChunkInfo {
    // write texture index in LOD model textures array
    let texture_name = get_material_base_color_texture_name(&prim.material());
    let texture_index = textures.iter().position(|t| t == &texture_name).expect("find texture") as i32;

    v3mc::MeshDataBlockChunkInfo{
        texture_index,
    }
}

fn create_mesh_chunk_data(prim: &gltf::Primitive, buffers: &Vec<BufferData>,
    transform: &Matrix3) -> v3mc::MeshChunkData {
    
    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

    let vecs: Vec<_> = reader.read_positions()
        .expect("mesh has no positions")
        .map(|pos| transform_point(&pos, transform))
        .collect();
    let norms: Vec<_> = reader.read_normals()
        .expect("mesh has no normals")
        .map(|norm| transform_normal(&norm, transform))
        .collect();
    let uvs: Vec<_> = reader.read_tex_coords(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_else(|| (0..vecs.len()).map(|i| generate_uv(&vecs[i], &norms[i])).collect());

    let indices: Vec<_> = reader.read_indices().expect("mesh has no indices").into_u32().collect();
    // Sanity checks
    assert!(indices.len() % 3 == 0, "number of indices is not a multiple of three: {}", indices.len());
    assert!(vecs.len() == norms.len());
    let nv = vecs.len();

    let faces: Vec<_> = indices.chunks(3)
        .map(|tri| [tri[0].try_into().unwrap(), tri[1].try_into().unwrap(), tri[2].try_into().unwrap()])
        .map(|vindices| v3mc::MeshFace{
            vindices,
            flags: if prim.material().double_sided() { v3mc::MeshFace::DOUBLE_SIDED } else { 0 },
        })
        .collect();

    let face_planes: Vec::<_> = indices.chunks(3)
        .map(|tri| (tri[0] as usize, tri[1] as usize, tri[2] as usize))
        .map(|(i, j, k)| compute_triangle_plane(&vecs[i], &vecs[j], &vecs[k]))
        .collect();

    let same_pos_vertex_offsets: Vec<i16> = (0..nv).map(|_| 0).collect();
    let wis: Vec<_> = (0..nv).map(|_| v3mc::WeightIndexArray::default()).collect();

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

fn create_prop_point(node: &gltf::Node, transform: &Matrix3) -> v3mc::PropPoint {
    let (translation, rotation, _scale) = node.transform().decomposed();
    v3mc::PropPoint{
        name: node.name().expect("prop point name").to_string(),
        orient: rotation,
        pos: transform_point(&translation, &transform),
        parent_index: -1,
    }
}

fn create_mesh_data_block<'a>(
    mesh: &gltf::Mesh, 
    buffers: &Vec<BufferData>, 
    transform: &Matrix3, 
    textures: &Vec::<String>, 
    prop_points_nodes: impl Iterator<Item = &'a gltf::Node<'a>>
) -> v3mc::MeshDataBlock {
    v3mc::MeshDataBlock{
        chunks: mesh.primitives().map(|prim| create_mesh_chunk_info(&prim, textures)).collect(),
        chunks_data: mesh.primitives().map(|prim| create_mesh_chunk_data(&prim, buffers, transform)).collect(),
        prop_points: prop_points_nodes.map(|prop| create_prop_point(&prop, transform)).collect(),
    }
}

fn compute_render_mode_for_material(material: &gltf::material::Material) -> u32 {
    // for example 0x518C41: tex_src = 1, color_op = 2, alpha_op = 3, alpha_blend = 3, zbuffer_type = 5, fog = 0
    let mut tex_src = v3mc::TextureSource::Wrap;
    if let Some(tex_info) = material.pbr_metallic_roughness().base_color_texture() {
        use gltf::texture::WrappingMode;
        let sampler = tex_info.texture().sampler();
        if sampler.wrap_t() != sampler.wrap_s() {
            eprintln!("Ignoring wrapT - wrapping mode must be the same for T and S");
        }
        if sampler.wrap_s() == WrappingMode::MirroredRepeat {
            eprintln!("MirroredRepeat wrapping mode is not supported");
        }

        tex_src = if sampler.wrap_s() == WrappingMode::ClampToEdge {
            v3mc::TextureSource::Clamp
        } else {
            v3mc::TextureSource::Wrap
        };
    }

    let color_op = v3mc::ColorOp::Mul;
    let alpha_op = v3mc::AlphaOp::Mul;

    use gltf::material::AlphaMode;
    let alpha_blend = match material.alpha_mode() {
        AlphaMode::Blend => v3mc::AlphaBlend::AlphaBlendAlpha,
        _ => v3mc::AlphaBlend::None,
    };
    let zbuffer_type = match material.alpha_mode() {
        AlphaMode::Opaque => v3mc::ZbufferType::Full,
        _ => v3mc::ZbufferType::FullAlphaTest,
    };
    let fog = v3mc::FogType::Type0;
    v3mc::encode_render_mode(tex_src, color_op, alpha_op, alpha_blend, zbuffer_type, fog)
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

    let tri_count = index_count / 3;
    let vertex_count = get_primitive_vertex_count(prim);

    if env::var("IGNORE_GEOMETRY_LIMITS").is_err() {
        let index_limit = 10000 - 768;
        if index_count > index_limit {
            return Err(new_custom_error(format!("primitive has too many indices: {} (limit {})", index_count, index_limit)));
        }

        let vertex_limit = 6000 - 768;
        if vertex_count > 6000 {
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
    let render_mode = compute_render_mode_for_material(&prim.material());
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

fn create_mesh_texture_ref(tex_name: &str, textures: &Vec::<String>) -> v3mc::MeshTextureRef {
    let material_index = textures.iter().position(|n| n == tex_name).unwrap().try_into().unwrap();
    v3mc::MeshTextureRef{
        material_index,
        tex_name: tex_name.into(),
    }
}

fn convert_mesh(
    node: &gltf::Node, 
    buffers: &Vec<BufferData>, 
    lod_mesh_tex_names: &Vec::<String>,
    transform: &Matrix3
) -> std::io::Result<v3mc::Mesh> {

    let mesh = node.mesh().unwrap();
    let flags = 0x20; // 0x1|0x02 - characters, 0x20 - static meshes, 0x10 only driller01.v3m
    let num_vecs = count_mesh_vertices(&mesh) as i32;

    let tex_names: Vec<_> = mesh.primitives()
        .map(|prim| get_material_base_color_texture_name(&prim.material()))
        .collect();
    if tex_names.len() > v3mc::Mesh::MAX_TEXTURES {
        return Err(new_custom_error(format!("found {} textures in a submesh but only {} are allowed",
        tex_names.len(), v3mc::Mesh::MAX_TEXTURES)));
    }

    let prop_point_nodes: Vec<_> = get_prop_point_nodes(node).collect();

    let mut chunks = Vec::new();
    for prim in mesh.primitives() {
        chunks.push(create_mesh_chunk(&prim)?);
    }

    let mut data_block_cur = Cursor::new(Vec::<u8>::new());
    create_mesh_data_block(&mesh, buffers, transform, &tex_names, prop_point_nodes.iter())
        .write(&mut data_block_cur)?;
    let data_block: Vec::<u8> = data_block_cur.into_inner();
    
    let num_prop_points = prop_point_nodes.len() as i32;
    let tex_refs: Vec<_> = tex_names.into_iter()
        .map(|tex_name| create_mesh_texture_ref(&tex_name, lod_mesh_tex_names))
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

fn create_material(tex_name: &str, emissive_factor: f32) -> v3mc::Material {
    v3mc::Material{
        tex_name: tex_name.to_string(),
        self_illumination: emissive_factor,
        flags: 0x11,
        ..v3mc::Material::default()
    }
}

fn change_texture_ext_to_tga(name: &str) -> String {
    let dot_offset = name.find('.').unwrap_or(name.len());
    let mut owned = name.to_owned();
    owned.replace_range(dot_offset.., ".tga");
    owned
}

fn get_material_base_color_texture_name(material: &gltf::material::Material) -> String {
    if let Some(tex_info) = material.pbr_metallic_roughness().base_color_texture() {
        let tex = tex_info.texture();
        let img = tex.source();
        if let Some(img_name) = img.name() {
            return change_texture_ext_to_tga(img_name);
        }
        if let gltf::image::Source::Uri { uri, .. } = img.source() {
            return change_texture_ext_to_tga(uri);
        }
    }
    const DEFAULT_TEXTURE: &'static str = "Rck_Default.tga";
    eprintln!("Cannot obtain texture name for material {} (materials without base color texture are not supported)",
        material.index().unwrap_or(0));
    DEFAULT_TEXTURE.into()
}

fn get_emissive_factor(mesh: &gltf::Mesh, texture: &str) -> f32 {
    mesh.primitives()
        .filter(|prim| get_material_base_color_texture_name(&prim.material()) == texture)
        .map(|prim| prim.material().emissive_factor())
        .map(|emissive_factor_rgb| emissive_factor_rgb.iter().cloned().fold(0f32, f32::max))
        .fold(0f32, f32::max)
}

fn convert_lod_mesh(node: &gltf::Node, buffers: &Vec<BufferData>) -> std::io::Result<v3mc::LodMesh> {
    let node_transform = node.transform().matrix();
    let mesh = node.mesh().unwrap();

    let name = node.name().unwrap_or("Default").to_string();
    let parent_name = "None".to_string();
    let version = v3mc::MeshDataBlock::VERSION;
    let distances = vec!(0.0);
    let (origin, rot_scale_mat) = extract_translation_from_matrix(&node_transform);

    let bbox = compute_mesh_bbox(&mesh, buffers, &rot_scale_mat);
    let (bbox_min, bbox_max) = (bbox.min, bbox.max);

    let offset = origin;
    let radius = compute_mesh_bounding_sphere_radius(&mesh, buffers, &rot_scale_mat);

    let textures = get_submesh_textures(node);
    let meshes: Vec<_> = vec!(convert_mesh(&node, buffers, &textures, &rot_scale_mat)?);

    let materials: Vec<_> = textures.into_iter().map(|tex_name| {
        let emissive_factor = get_emissive_factor(&mesh, &tex_name);
        create_material(&tex_name, emissive_factor)
    }).collect();

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

fn convert_csphere(node: &gltf::Node) -> v3mc::ColSphere {
    let name = node.name().unwrap_or("csphere");
    let (translation, _rotation, scale) = node.transform().decomposed();
    let radius = scale[0].max(scale[1]).max(scale[2]);
    v3mc::ColSphere{
        name: name.to_string(),
        parent_index: -1,
        pos: translation,
        radius,
    }
}

fn write_v3m_file<W: Write + Seek>(wrt: &mut W, doc: &gltf::Document, buffers: &Vec<BufferData>) -> std::io::Result<()> {

    if doc.nodes().filter(|n| n.children().count() > 0).count() > 0 {
        eprintln!("Node hierarchy is ignored!");
    }

    let lod_meshes = get_submesh_nodes(doc).map(|n| convert_lod_mesh(&n, buffers).unwrap()).collect();
    let cspheres = get_csphere_nodes(doc).map(|n| convert_csphere(&n)).collect();
    v3mc::File{
        header: create_v3m_file_header(&lod_meshes, &cspheres),
        lod_meshes,
        cspheres,
    }.write(wrt)
}

fn convert_gltf_to_v3m(input_file_name: &str, output_file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Importing GLTF file {}...", input_file_name);
    let gltf = gltf::Gltf::open(&input_file_name)?;
    let input_path: &std::path::Path = input_file_name.as_ref();
    let gltf::Gltf { document, blob } = gltf;

    println!("Importing GLTF buffers...");
    let buffers = import::import_buffer_data(&document, input_path.parent(), blob)?;
    
    println!("Converting...");
    let file = File::create(output_file_name)?;
    let mut wrt = BufWriter::new(file);
    write_v3m_file(&mut wrt, &document, &buffers)?;
    
    println!("Converted successfully.");
    Ok(())
}

fn main() {
    println!("GLTF to V3M converter {} by Rafalh", env!("CARGO_PKG_VERSION"));

    let mut args = env::args();
    let app_name = args.next().unwrap();
    if env::args().len() != 3 {
        println!("Usage: {} input_file_name.gltf output_file_name.v3m", app_name);
        std::process::exit(1);
    }

    let input_file_name = args.next().unwrap();
    let output_file_name = args.next().unwrap();

    if let Err(e) = convert_gltf_to_v3m(&input_file_name, &output_file_name) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
