use glam::Quat;

use crate::{gltf_to_rf_face, gltf_to_rf_quat, gltf_to_rf_vec, io_utils::new_custom_error, math_utils::{compute_triangle_plane, generate_uv}, rfg::{Brush, Face, FaceVertex, Group, Rfg, Solid}, BoxResult, Context};

pub fn convert_gltf_to_rfg(doc: &gltf::Document, ctx: &Context) -> BoxResult<Rfg> {
    let mut next_uid = 1;
    let mut groups = Vec::new();
    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let group_name = node.name().unwrap_or_default().to_owned();
        let transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        let mut brushes = Vec::new();
        for prim in mesh.primitives() {
            brushes.push(convert_primitive(next_uid, prim, ctx, &transform)?);
            next_uid += 1;
        }
        groups.push(Group { group_name, brushes });
    }
    let rfg = Rfg { groups };
    Ok(rfg)
}

fn convert_primitive(uid: i32, prim: gltf::Primitive, ctx: &Context, transform: &glam::Mat4) -> std::io::Result<Brush> {
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return Err(new_custom_error("only triangle list primitives are supported")); // FIXME
    }
    // TODO: apply scale?
    let (_scale, rotation, translation) = transform.to_scale_rotation_translation();
    let reader = prim.reader(|buffer| ctx.get_buffer_data(buffer));
    let vecs: Vec<_> = reader.read_positions()
        .expect("mesh has no positions")
        .map(|pos| gltf_to_rf_vec(pos))
        .collect();
    let uvs_opt: Option<Vec<_>> = reader.read_tex_coords(0)
        .map(|iter| iter.into_f32().collect());
    let indices: Vec<u32> = reader.read_indices()
        .expect("mesh has no indices")
        .into_u32()
        .collect();

    let faces = indices
        .chunks_exact(3)
        .map(|chunk| gltf_to_rf_face([chunk[0], chunk[1], chunk[2]]))
        .map(|chunk| {
            let plane = compute_triangle_plane(&vecs[chunk[0] as usize], &vecs[chunk[1] as usize], &vecs[chunk[2] as usize]);
            let plane_normal = [plane[0], plane[1], plane[2]];
            Face {
                plane,
                texture: 0,
                vertices: chunk.iter()
                    .copied()
                    .map(|index| FaceVertex { 
                        index, 
                        texture_coords: uvs_opt.as_ref().map_or_else(
                            || generate_uv(&vecs[index as usize], &plane_normal),
                            |uvs| uvs[index as usize]
                        ),
                    })
                    .collect()
            }
        })
        .collect();
    let solid = Solid {
        textures: vec!["Rck_Default.tga".to_owned()], // TODO
        vertices: vecs,
        faces,
    };
    let pos = gltf_to_rf_vec(translation.to_array());
    let orient = gltf_to_rf_quat(rotation.to_array());
    let orient = glam::Mat3::from_quat(Quat::from_array(orient)).to_cols_array();
    let brush = Brush {
        uid,
        pos,
        orient,
        solid,
    };
    Ok(brush)
}
