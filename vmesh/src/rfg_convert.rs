use std::collections::HashMap;

use glam::{Quat, Vec3};

use crate::{
    BoxResult, Context, gltf_to_rf_face, gltf_to_rf_quat, gltf_to_rf_vec,
    io_utils::new_custom_error,
    material::get_material_base_color_texture_name,
    math_utils::{compute_triangle_plane, generate_uv},
    rfg::{Brush, Face, FaceVertex, Group, Rfg, Solid},
};

pub fn convert_gltf_to_rfg(doc: &gltf::Document, ctx: &Context) -> BoxResult<Rfg> {
    let mut next_uid = 1;
    let mut groups = Vec::new();
    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let group_name = node.name().unwrap_or_default().to_owned();
        let transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        let brush = create_brush(mesh, next_uid, ctx, &transform)?;
        next_uid += 1;
        let brushes = vec![brush];
        groups.push(Group {
            group_name,
            brushes,
        });
    }
    let rfg = Rfg { groups };
    Ok(rfg)
}

fn create_brush(
    mesh: gltf::Mesh,
    uid: i32,
    ctx: &Context,
    transform: &glam::Mat4,
) -> std::io::Result<Brush> {
    let (scale, rotation, translation) = transform.to_scale_rotation_translation();

    let mut vertices = Vec::new();
    let mut textures = Vec::new();
    let mut faces = Vec::new();

    for prim in mesh.primitives() {
        if prim.mode() != gltf::mesh::Mode::Triangles {
            return Err(new_custom_error(
                "only triangle list primitives are supported",
            ));
        }

        let texture_name = get_material_base_color_texture_name(&prim.material());
        let texture_index = textures
            .iter()
            .position(|t| t == &texture_name)
            .unwrap_or_else(|| {
                textures.push(texture_name);
                textures.len() - 1
            });

        let reader = prim.reader(|buffer| ctx.get_buffer_data(buffer));

        let prim_v_index_to_brush_v_index: HashMap<usize, usize> = reader
            .read_positions()
            .expect("mesh has no positions")
            .map(gltf_to_rf_vec)
            .map(|v| (Vec3::from_array(v) * scale).to_array())
            .enumerate()
            .map(|(prim_v_index, prim_v)| {
                for (brush_v_index, v) in vertices.iter().enumerate() {
                    if prim_v == *v {
                        return (prim_v_index, brush_v_index);
                    }
                }
                let brush_v_index = vertices.len();
                vertices.push(prim_v);
                (prim_v_index, brush_v_index)
            })
            .collect();

        let uvs_opt: Option<Vec<_>> = reader
            .read_tex_coords(0)
            .map(|iter| iter.into_f32().collect());

        let indices: Vec<u32> = reader
            .read_indices()
            .expect("mesh has no indices")
            .into_u32()
            .collect();

        indices
            .chunks_exact(3)
            .map(|chunk| gltf_to_rf_face([chunk[0], chunk[1], chunk[2]]))
            .map(|chunk| {
                let [v1, v2, v3] = chunk
                    .map(|i| prim_v_index_to_brush_v_index[&(i as usize)])
                    .map(|i| vertices[i]);
                let plane = compute_triangle_plane(&v1, &v2, &v3);
                let plane_normal = [plane[0], plane[1], plane[2]];
                Face {
                    plane,
                    texture: texture_index as i32,
                    vertices: chunk
                        .iter()
                        .copied()
                        .map(|index| {
                            let brush_v_index = prim_v_index_to_brush_v_index[&(index as usize)];
                            let texture_coords = uvs_opt.as_ref().map_or_else(
                                || generate_uv(&vertices[brush_v_index], &plane_normal),
                                |uvs| uvs[index as usize],
                            );
                            FaceVertex {
                                index: brush_v_index as u32,
                                texture_coords,
                            }
                        })
                        .collect(),
                }
            })
            .for_each(|f| faces.push(f));
    }

    if ctx.args.verbose >= 1 {
        println!(
            "Brush {}: {} vertices, {} faces",
            uid,
            vertices.len(),
            faces.len()
        );
    }
    let solid = Solid {
        textures,
        vertices,
        faces,
    };
    let pos = gltf_to_rf_vec(translation.to_array());
    let orient_quat = gltf_to_rf_quat(rotation.to_array());
    let orient = glam::Mat3::from_quat(Quat::from_array(orient_quat)).to_cols_array();
    let brush = Brush {
        uid,
        pos,
        orient,
        solid,
    };
    Ok(brush)
}
