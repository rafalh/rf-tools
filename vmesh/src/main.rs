mod import;
mod io_utils;
mod math_utils;
mod v3mc;

use std::io::Cursor;
use std::fs::File;
use std::io::BufWriter;
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
use math_utils::*;

fn build_child_nodes_indices(doc: &gltf::Document) -> Vec<usize> {
    let mut child_indices: Vec<usize> = doc.nodes().flat_map(|n| n.children().map(|n| n.index())).collect();
    child_indices.dedup();
    child_indices
}

fn get_submesh_nodes(doc: &gltf::Document) -> Vec<gltf::Node> {
    let child_indices = build_child_nodes_indices(doc);
    doc.nodes().filter(|n| n.mesh().is_some() && !child_indices.contains(&n.index())).collect()
}

fn get_csphere_nodes(doc: &gltf::Document) -> Vec<gltf::Node> {
    doc.nodes().filter(|n| n.mesh().is_none()).filter(|n| n.name().unwrap_or("").starts_with("csphere_")).collect()
}

fn get_prop_point_nodes<'a>(parent: &gltf::Node<'a>) -> impl Iterator<Item = gltf::Node<'a>> {
    parent.children().filter(|n| n.mesh().is_none() && n.name().is_some())
}

fn get_mesh_materials<'a>(mesh: &gltf::Mesh<'a>) -> Vec<gltf::Material<'a>> {
    let mut materials = mesh.primitives().map(|prim| prim.material())
        .collect::<Vec<_>>();
    materials.dedup_by_key(|m| m.index());
    materials
}

fn get_material_self_illumination(mat: &gltf::Material) -> f32 {
    mat.emissive_factor().iter().cloned().fold(0f32, f32::max)
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
    prim.attributes().find(|p| p.0 == gltf::mesh::Semantic::Positions).map(|a| a.1.count()).unwrap_or(0)
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

    let face_planes: Vec::<_> = if !is_character {
        indices.chunks(3)
            .map(|tri| (tri[0] as usize, tri[1] as usize, tri[2] as usize))
            .map(|(i, j, k)| compute_triangle_plane(&vecs[i], &vecs[j], &vecs[k]))
            .collect()
    } else {
        Vec::new()
    };

    let same_pos_vertex_offsets: Vec<i16> = (0..nv).map(|_| 0).collect();
    
    let wis: Vec<_> = if let Some(joints) = reader.read_joints(0) {
        joints.into_u16()
            .zip(reader.read_weights(0).expect("mesh has no weights").into_u8())
            .map(|(indices_u16, weights)| {
                let indices = indices_u16
                    .map(|x| x.try_into().expect("joint index should fit in u8"));
                v3mc::WeightIndexArray { indices, weights }
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

fn create_prop_point(node: &gltf::Node, transform: &Matrix3) -> v3mc::PropPoint {
    let (translation, rotation, _scale) = node.transform().decomposed();
    v3mc::PropPoint{
        name: node.name().expect("prop point name is missing").to_string(),
        orient: rotation,
        pos: transform_point(&translation, transform),
        parent_index: -1,
    }
}

fn create_mesh_data_block(
    mesh: &gltf::Mesh, 
    buffers: &[BufferData], 
    transform: &Matrix3, 
    mesh_materials: &[gltf::Material], 
    prop_points_nodes: &[gltf::Node],
    is_character: bool
) -> v3mc::MeshDataBlock {
    v3mc::MeshDataBlock{
        chunks: mesh.primitives().map(|prim| create_mesh_chunk_info(&prim, mesh_materials)).collect(),
        chunks_data: mesh.primitives().map(|prim| create_mesh_chunk_data(&prim, buffers, transform, is_character)).collect(),
        prop_points: prop_points_nodes.iter().map(|prop| create_prop_point(prop, transform)).collect(),
    }
}

fn compute_render_mode_for_material(material: &gltf::material::Material) -> u32 {
    // for example 0x400C41 (sofa1.v3m):
    //   tex_src = 1, color_op = 2, alpha_op = 3, alpha_blend = 0, zbuffer_type = 5, fog = 0
    // for example 0x518C41 (paper1.v3m, per1.v3m, ...):
    //   tex_src = 1, color_op = 2, alpha_op = 3, alpha_blend = 3, zbuffer_type = 5, fog = 0
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

fn create_mesh_material_ref(material: &gltf::Material, lod_mesh_materials: &[gltf::Material]) -> v3mc::MeshTextureRef {
    let material_index = lod_mesh_materials.iter()
        .position(|m| m.index() == material.index())
        .unwrap()
        .try_into()
        .unwrap();
    v3mc::MeshTextureRef{
        material_index,
        tex_name: get_material_base_color_texture_name(material),
    }
}

fn convert_mesh(
    node: &gltf::Node, 
    buffers: &[BufferData], 
    lod_mesh_materials: &[gltf::Material],
    prop_point_nodes: &[gltf::Node],
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
    create_mesh_data_block(&mesh, buffers, transform, &materials, prop_point_nodes, is_character)
        .write(&mut data_block_cur)?;
    let data_block: Vec::<u8> = data_block_cur.into_inner();
    
    let num_prop_points = prop_point_nodes.len() as i32;
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

fn change_texture_ext_to_tga(name: &str) -> String {
    String::from(Path::new(name).with_extension("tga").file_name().unwrap().to_string_lossy())
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
    const DEFAULT_TEXTURE: &str = "Rck_Default.tga";
    eprintln!("Cannot obtain texture name for material {} (materials without base color texture are not supported)",
        material.index().unwrap_or(0));
    DEFAULT_TEXTURE.into()
}

fn convert_material(mat: &gltf::Material) -> v3mc::Material {
    let tex_name = get_material_base_color_texture_name(mat);
    let self_illumination = get_material_self_illumination(mat);
    let specular_level = mat.pbr_specular_glossiness()
        .map(|spec_glos| spec_glos.specular_factor().iter().cloned().fold(0f32, f32::max))
        .unwrap_or_else(|| mat.pbr_metallic_roughness().metallic_factor());
    let glossiness = mat.pbr_specular_glossiness()
        .map(|spec_glos| spec_glos.glossiness_factor())
        .unwrap_or_else(|| 1.0 - mat.pbr_metallic_roughness().roughness_factor());

    v3mc::Material{
        tex_name,
        self_illumination,
        specular_level,
        glossiness,
        flags: 0x11,
        ..v3mc::Material::default()
    }
}

#[derive(Deserialize, Debug)]
struct NodeExtras {
    #[serde(rename = "LOD_distance")]
    lod_distance: Option<f32>,
}

fn find_lod_nodes<'a>(node: &'a gltf::Node) -> Vec<(gltf::Node<'a>, f32)> {
    let mut child_node_dist_vec: Vec<(gltf::Node, f32)> = node.children()
        .filter(|n| n.mesh().is_some())
        .map(|n| {
            let extras_raw: &serde_json::value::RawValue = n.extras().as_ref().unwrap();
            let extras = serde_json::from_str::<NodeExtras>(extras_raw.get()).unwrap();
            (n, extras.lod_distance)
        })
        .filter(|(n, d)| {
            if d.is_none() {
                eprintln!("Warning! Expected LOD_distance in child node {}", n.name().unwrap_or("None"));
            }
            d.is_some()
        })
        .map(|(n, dist_opt)| (n, dist_opt.unwrap()))
        .chain(iter::once((node.clone(), 0f32)))
        .collect();
    child_node_dist_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    child_node_dist_vec
}

fn convert_lod_mesh(node: &gltf::Node, buffers: &[BufferData], is_character: bool) -> Result<v3mc::LodMesh, Box<dyn Error>> {
    let node_transform = node.transform().matrix();
    let mesh = node.mesh().unwrap();
    let name = node.name().unwrap_or("Default").to_string();
    println!("Processing top-level node {} ({})", node.index(), name);

    let parent_name = "None".to_string();
    let version = v3mc::MeshDataBlock::VERSION;
    let child_node_dist_vec = find_lod_nodes(node);
    let distances = child_node_dist_vec.iter().map(|(_, dist)| *dist).collect();
    let (origin, rot_scale_mat) = extract_translation_from_matrix(&node_transform);

    let bbox = compute_mesh_bbox(&mesh, buffers, &rot_scale_mat);
    let (bbox_min, bbox_max) = (bbox.min, bbox.max);

    let offset = origin;
    let radius = compute_mesh_bounding_sphere_radius(&mesh, buffers, &rot_scale_mat);

    let prop_point_nodes: Vec<_> = get_prop_point_nodes(node).collect();

    let mut gltf_materials: Vec<_> = child_node_dist_vec.iter()
        .flat_map(|(n, _)| get_mesh_materials(&n.mesh().unwrap()))
        .collect(); 
    gltf_materials.dedup_by_key(|m| m.index());
    let materials: Vec<_> = gltf_materials.iter().map(convert_material).collect();

    let mut meshes: Vec<_> = Vec::with_capacity(child_node_dist_vec.len());
    for (n, d) in &child_node_dist_vec {
        println!("Processing node {} name {} distance {}", n.index(), n.name().unwrap_or("<unnamed>"), d);
        meshes.push(convert_mesh(n, buffers, &gltf_materials, &prop_point_nodes, &rot_scale_mat, is_character)?);
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

fn convert_csphere(node: &gltf::Node, index: usize) -> v3mc::ColSphere {
    let name = node.name()
        .map(&str::to_owned)
        .unwrap_or_else(|| format!("csphere_{}", index));
    let (translation, _rotation, scale) = node.transform().decomposed();
    let radius = scale[0].max(scale[1]).max(scale[2]);
    v3mc::ColSphere{
        name,
        parent_index: -1,
        pos: translation,
        radius,
    }
}

fn convert_cspheres(nodes: &[gltf::Node]) -> Vec<v3mc::ColSphere> {
    let mut cspheres = Vec::with_capacity(nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        cspheres.push(convert_csphere(n, i));
    }
    cspheres
}

fn get_bone_parent_index(node: &gltf::Node, skin: &gltf::Skin) -> i32 {
    skin.joints()
        .enumerate()
        .find(|(_i, n)| n.children().any(|c| c.index() == node.index()))
        .map(|(i, _n)| i as i32)
        .unwrap_or(-1)
}

fn convert_bones(skin: &gltf::Skin) -> std::io::Result<Vec<v3mc::Bone>> {
    let joints: Vec<_> = skin.joints().collect();
    if joints.len() > v3mc::MAX_BONES {
        let err_msg = format!("too many bones: found {} but only {} are supported", joints.len(), v3mc::MAX_BONES);
        return Err(new_custom_error(err_msg));
    }
    let mut bones = Vec::with_capacity(joints.len());
    for (i, n) in joints.iter().enumerate() {
        let name = n.name().map(&str::to_owned).unwrap_or_else(|| format!("bone_{}", i));
        let (pos, rot, _scale) = n.transform().decomposed();
        let parent = get_bone_parent_index(n, skin);
        let bone = v3mc::Bone { name, pos, rot, parent };
        bones.push(bone);
    }
    Ok(bones)
}

fn make_v3mc_file(doc: &gltf::Document, buffers: &[BufferData], is_character: bool) -> Result<v3mc::File, Box<dyn Error>> {
    let submesh_nodes = get_submesh_nodes(doc);
    println!("Found {} top-level mesh nodes", submesh_nodes.len());
    let mut lod_meshes = Vec::with_capacity(submesh_nodes.len());
    for n in &submesh_nodes {
        lod_meshes.push(convert_lod_mesh(n, buffers, is_character)?);
    }

    let csphere_nodes = get_csphere_nodes(doc);
    let cspheres = convert_cspheres(&csphere_nodes);

    let bones = if let Some(skin) = doc.skins().next() {
        convert_bones(&skin)?
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
        .map(|s| s.to_owned())
        .unwrap_or_else(|| {
            let base_file_name = input_file_name.strip_suffix(".gltf")
            .unwrap_or_else(|| input_file_name.strip_suffix(".glb").unwrap_or(input_file_name));
            let ext = if is_character { "v3c" } else { "v3m" };
            format!("{}.{}", base_file_name, ext)
        })
}

fn convert_gltf_to_v3mc(input_file_name: &str, output_file_name_opt: Option<&str>) -> Result<(), Box<dyn Error>> {
    println!("Importing GLTF file {}...", input_file_name);
    let input_path = Path::new(input_file_name);
    let gltf = gltf::Gltf::open(input_path)?;
    let gltf::Gltf { document, blob } = gltf;

    println!("Importing GLTF buffers...");
    let buffers = import::import_buffer_data(&document, input_path.parent(), blob)?;
    
    println!("Converting...");
    let is_character = document.skins().next().is_some();
    let output_file_name = generate_output_file_name(input_file_name, output_file_name_opt, is_character);
    let v3m = make_v3mc_file(&document, &buffers, is_character)?;
    let file = File::create(output_file_name)?;
    let mut wrt = BufWriter::new(file);
    v3m.write(&mut wrt)?;
    
    println!("Converted successfully.");
    Ok(())
}

fn main() {
    println!("GLTF to V3M converter {} by Rafalh", env!("CARGO_PKG_VERSION"));

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
