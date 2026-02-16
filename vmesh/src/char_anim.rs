use crate::io_utils::new_custom_error;
use crate::v3mc_convert::get_node_extras;
use crate::{Context, gltf_to_rf_quat, gltf_to_rf_vec, rfa, v3mc};
use gltf::animation::Interpolation;
use gltf::animation::util::{ReadInputs, ReadOutputs};
use serde_derive::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::f32;
use std::fs::File;
use std::io::BufWriter;
use std::vec::Vec;

#[derive(Deserialize, Debug, Default)]
struct JointExtras {
    #[serde(flatten)]
    map: BTreeMap<String, Value>,
}

impl JointExtras {
    fn get_ramp_in_time(&self, anim_name: &str) -> Option<f32> {
        self.get_float(&format!("ramp_in_time.{}", anim_name))
    }

    fn get_ramp_out_time(&self, anim_name: &str) -> Option<f32> {
        self.get_float(&format!("ramp_out_time.{}", anim_name))
    }

    fn get_anim_weight(&self, anim_name: &str) -> Option<f32> {
        self.get_float(&format!("weight.{}", anim_name))
    }

    fn get_float(&self, key: &str) -> Option<f32> {
        self.map.get(key).and_then(|v| v.as_f64()).map(|v| v as f32)
    }
}

fn gltf_time_to_rfa_time(time_sec: f32) -> i32 {
    (time_sec * 30.0_f32 * 160.0_f32) as i32
}

fn make_short_quat(quat: [f32; 4]) -> [i16; 4] {
    quat.map(|x| (x * 16383.0_f32) as i16)
}

fn get_node_anim_channels<'a>(
    n: &gltf::Node,
    anim: &'a gltf::Animation,
) -> impl Iterator<Item = gltf::animation::Channel<'a>> + 'a {
    let node_index = n.index();
    anim.channels()
        .filter(move |c| c.target().node().index() == node_index)
}

fn get_node_anim_data<'a>(
    n: &gltf::Node,
    anim: &'a gltf::Animation,
    ctx: &'a Context,
) -> impl Iterator<Item = (ReadInputs<'a>, ReadOutputs<'a>, Interpolation)> + 'a {
    get_node_anim_channels(n, anim).filter_map(move |channel| {
        let reader = channel.reader(|buffer| ctx.get_buffer_data(buffer));
        let interpolation = channel.sampler().interpolation();
        reader.read_inputs().and_then(|inputs| {
            reader
                .read_outputs()
                .map(|outputs| (inputs, outputs, interpolation))
        })
    })
}

fn convert_rotation_keys(
    n: &gltf::Node,
    anim: &gltf::Animation,
    ctx: &Context,
) -> Vec<rfa::RotationKey> {
    get_node_anim_data(n, anim, ctx)
        .filter_map(|(inputs, outputs, interpolation)| match outputs {
            ReadOutputs::Rotations(rotations) => Some((inputs, rotations, interpolation)),
            _ => None,
        })
        .map(|(inputs, rotations, interpolation)| {
            let rotations_quads = rotations
                .into_f32()
                .map(gltf_to_rf_quat)
                .map(make_short_quat);
            let chunked_rotations = if interpolation == Interpolation::CubicSpline {
                rotations_quads
                    .collect::<Vec<_>>()
                    .chunks(3)
                    .map(|s| (s[0], s[1], s[2]))
                    .collect::<Vec<_>>()
            } else {
                rotations_quads.map(|r| (r, r, r)).collect::<Vec<_>>()
            };
            inputs
                .map(gltf_time_to_rfa_time)
                .zip(chunked_rotations)
                .map(|(time, (_, rotation, _))| rfa::RotationKey {
                    time,
                    rotation,
                    ease_in: 0,
                    ease_out: 0,
                })
                .collect::<Vec<_>>()
        })
        .next()
        .unwrap_or_default()
}

fn convert_translation_keys(
    n: &gltf::Node,
    anim: &gltf::Animation,
    ctx: &Context,
) -> Vec<rfa::TranslationKey> {
    get_node_anim_data(n, anim, ctx)
        .filter_map(|(inputs, outputs, interpolation)| match outputs {
            ReadOutputs::Translations(translations) => Some((inputs, translations, interpolation)),
            _ => None,
        })
        .map(|(inputs, translations, interpolation)| {
            let rf_translations = translations.map(gltf_to_rf_vec);
            let chunked_translations = if interpolation == Interpolation::CubicSpline {
                rf_translations
                    .collect::<Vec<_>>()
                    .chunks(3)
                    .map(|s| (s[0], s[1], s[2]))
                    .collect::<Vec<_>>()
            } else {
                rf_translations.map(|t| (t, t, t)).collect::<Vec<_>>()
            };
            inputs
                .map(gltf_time_to_rfa_time)
                .zip(chunked_translations)
                .map(|(time, (_, translation, _))|
                    // ignore cubic spline tangents for now - RF uses bezier curve and tangents are different
                    rfa::TranslationKey {
                        time,
                        in_tangent: translation,
                        translation,
                        out_tangent: translation,
                    })
                .collect::<Vec<_>>()
        })
        .next()
        .unwrap_or_default()
}

fn determine_anim_time_range(bones: &[rfa::Bone]) -> (i32, i32) {
    bones
        .iter()
        .flat_map(|b| {
            b.rotation_keys
                .iter()
                .map(|k| k.time)
                .chain(b.translation_keys.iter().map(|k| k.time))
        })
        .fold((0_i32, 0_i32), |(min, max), time| {
            (min.min(time), max.max(time))
        })
}

fn check_for_scale_channels(n: &gltf::Node, anim: &gltf::Animation, ctx: &Context) {
    for channel in get_node_anim_channels(n, anim) {
        let reader = channel.reader(|buffer| ctx.get_buffer_data(buffer));
        if let Some(ReadOutputs::Scales(scales)) = reader.read_outputs()
            && scales.flatten().any(|s| (s - 1.0_f32).abs() > 0.01_f32)
        {
            eprintln!(
                "Warning! Animation #{} '{}' is using unsupported scale channel on node #{} '{}'!",
                anim.index(),
                anim.name().unwrap_or_default(),
                n.index(),
                n.name().unwrap_or_default(),
            );
        }
    }
}

fn is_death_anim(anim: &gltf::Animation) -> bool {
    anim.name().unwrap_or_default().contains("death")
}

fn are_all_anim_keys_equal(
    rotation_keys: &[rfa::RotationKey],
    translation_keys: &[rfa::TranslationKey],
) -> bool {
    let initial_rotation = rotation_keys.iter().next().map(|rk| rk.rotation);
    let initial_translation = translation_keys.iter().next().map(|tk| tk.translation);
    let rotation_keys_equal = rotation_keys
        .iter()
        .all(|rk| Some(rk.rotation) == initial_rotation);
    let translation_keys_equal = translation_keys
        .iter()
        .all(|tk| Some(tk.translation) == initial_translation);
    rotation_keys_equal && translation_keys_equal
}

fn determine_anim_weight(
    rotation_keys: &[rfa::RotationKey],
    translation_keys: &[rfa::TranslationKey],
) -> f32 {
    if are_all_anim_keys_equal(rotation_keys, translation_keys) {
        2.0
    } else {
        10.0
    }
}

fn convert_bone_anim(node: &gltf::Node, anim: &gltf::Animation, ctx: &Context) -> rfa::Bone {
    let rotation_keys = convert_rotation_keys(node, anim, ctx);
    let translation_keys = convert_translation_keys(node, anim, ctx);
    check_for_scale_channels(node, anim, ctx);
    let extras = get_node_extras::<JointExtras>(node);
    let weight = extras
        .get_anim_weight(anim.name().unwrap_or_default())
        .or(ctx.args.anim_weight)
        .unwrap_or_else(|| determine_anim_weight(&rotation_keys, &translation_keys));
    rfa::Bone {
        weight,
        rotation_keys,
        translation_keys,
    }
}

fn get_default_ramp_in_time(anim: &gltf::Animation) -> i32 {
    if is_death_anim(anim) { 800 } else { 480 } // 0.1(6) s, 0.1 s
}

fn get_default_ramp_out_time(anim: &gltf::Animation) -> i32 {
    if is_death_anim(anim) { 0 } else { 480 } // 0.0 s, 0.1 s
}

fn determine_ramp_in_time(
    anim: &gltf::Animation,
    root_joint_extras: &JointExtras,
    duration: i32,
    ctx: &Context,
) -> i32 {
    let anim_name = anim.name().unwrap_or_default();
    root_joint_extras
        .get_ramp_in_time(anim_name)
        .or(ctx.args.ramp_in_time)
        .map(gltf_time_to_rfa_time)
        .unwrap_or_else(|| get_default_ramp_in_time(anim).min(duration / 2))
}

fn determine_ramp_out_time(
    anim: &gltf::Animation,
    root_joint_extras: &JointExtras,
    duration: i32,
    ctx: &Context,
) -> i32 {
    let anim_name = anim.name().unwrap_or_default();
    root_joint_extras
        .get_ramp_out_time(anim_name)
        .or(ctx.args.ramp_out_time)
        .map(gltf_time_to_rfa_time)
        .unwrap_or_else(|| get_default_ramp_out_time(anim).min(duration / 2))
}

fn is_root_joint(node: &gltf::Node) -> bool {
    node.name()
        .unwrap_or_default()
        .to_lowercase()
        .ends_with("root")
}

fn find_root_joint<'a>(skin: &'a gltf::Skin) -> Option<gltf::Node<'a>> {
    skin.joints().find(is_root_joint)
}

fn get_root_joint_extras(skin: &gltf::Skin) -> JointExtras {
    find_root_joint(skin)
        .map(|n| get_node_extras::<JointExtras>(&n))
        .unwrap_or_default()
}

fn make_rfa(anim: &gltf::Animation, skin: &gltf::Skin, ctx: &Context) -> rfa::File {
    let mut bones = Vec::with_capacity(skin.joints().count());
    for joint in skin.joints() {
        bones.push(convert_bone_anim(&joint, anim, ctx));
    }
    let (start_time, end_time) = determine_anim_time_range(&bones);
    let duration = end_time - start_time;
    let root_joint_extras = get_root_joint_extras(skin);
    let ramp_in_time = determine_ramp_in_time(anim, &root_joint_extras, duration, ctx);
    let ramp_out_time = determine_ramp_out_time(anim, &root_joint_extras, duration, ctx);
    if ctx.args.verbose >= 2 {
        println!("Start time: {}", start_time);
        println!("End time: {}", end_time);
        println!("Ramp In time: {}", ramp_in_time);
        println!("Ramp Out time: {}", ramp_out_time);
    }
    let header = rfa::FileHeader {
        num_bones: bones.len() as i32,
        start_time,
        end_time,
        ramp_in_time,
        ramp_out_time,
        total_rotation: [0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32],
        total_translation: [0.0_f32; 3],
        ..rfa::FileHeader::default()
    };
    rfa::File { header, bones }
}

pub(crate) fn convert_animation_to_rfa(
    anim: &gltf::Animation,
    index: usize,
    skin: &gltf::Skin,
    ctx: &Context,
) -> std::io::Result<()> {
    let name = anim
        .name()
        .map_or_else(|| format!("anim_{}", index), str::to_owned);
    let file_name = ctx.output_dir.join(format!("{}.rfa", name));
    if ctx.args.verbose >= 1 {
        println!("Exporting animation: {} -> {}", name, file_name.display());
    }
    let mut wrt = BufWriter::new(File::create(&file_name)?);
    let rfa = make_rfa(anim, skin, ctx);
    rfa.write(&mut wrt)?;
    Ok(())
}

fn get_joint_index(node: &gltf::Node, skin: &gltf::Skin) -> usize {
    skin.joints()
        .enumerate()
        .filter(|(_i, n)| node.index() == n.index())
        .map(|(i, _n)| i)
        .next()
        .expect("joint not found")
}

fn get_joint_parent<'a>(node: &gltf::Node, skin: &gltf::Skin<'a>) -> Option<gltf::Node<'a>> {
    skin.joints()
        .find(|n| n.children().any(|c| c.index() == node.index()))
}

fn convert_bone(
    n: &gltf::Node,
    inverse_bind_matrix: &[[f32; 4]; 4],
    index: usize,
    skin: &gltf::Skin,
) -> v3mc::Bone {
    let name = n
        .name()
        .map_or_else(|| format!("bone_{}", index), str::to_owned);
    let parent_node_opt = get_joint_parent(n, skin);
    let parent_index = parent_node_opt.map_or(-1, |pn| get_joint_index(&pn, skin) as i32);
    let inv_transform = glam::Mat4::from_cols_array_2d(inverse_bind_matrix);
    let (gltf_scale, gltf_rotation, gltf_translation) =
        inv_transform.to_scale_rotation_translation();
    assert!(
        (gltf_scale - glam::Vec3::ONE).max_element() < 0.01_f32,
        "scale is not supported: {}",
        gltf_scale
    );
    let base_rotation = gltf_to_rf_quat(gltf_rotation.into());
    let base_translation = gltf_to_rf_vec(gltf_translation.into());
    v3mc::Bone {
        name,
        base_rotation,
        base_translation,
        parent_index,
    }
}

pub(crate) fn convert_bones(skin: &gltf::Skin, ctx: &Context) -> std::io::Result<Vec<v3mc::Bone>> {
    let num_joints = skin.joints().count();
    if ctx.args.verbose >= 2 {
        println!("Bones (joints): {}/{}", num_joints, v3mc::MAX_BONES);
    }
    if num_joints > v3mc::MAX_BONES {
        let err_msg = format!(
            "too many bones: found {} but only {} are supported",
            num_joints,
            v3mc::MAX_BONES
        );
        return Err(new_custom_error(err_msg));
    }

    let inverse_bind_matrices: Vec<_> = skin
        .reader(|buffer| ctx.get_buffer_data(buffer))
        .read_inverse_bind_matrices()
        .expect("expected inverse bind matrices")
        .collect();

    if inverse_bind_matrices.len() != num_joints {
        let err_msg = format!(
            "invalid number of inverse bind matrices: expected {}, got {}",
            num_joints,
            inverse_bind_matrices.len()
        );
        return Err(new_custom_error(err_msg));
    }

    let mut bones = Vec::with_capacity(num_joints);
    for (i, n) in skin.joints().enumerate() {
        let bone = convert_bone(&n, &inverse_bind_matrices[i], i, skin);
        bones.push(bone);
    }
    Ok(bones)
}

fn is_joint(node: &gltf::Node, skin: &gltf::Skin) -> bool {
    skin.joints().any(|joint| node.index() == joint.index())
}

pub(crate) fn get_nodes_parented_to_bones<'a>(
    skin: &'a gltf::Skin,
) -> impl Iterator<Item = (gltf::Node<'a>, i32)> {
    skin.joints()
        .flat_map(move |joint| {
            let joint_index = get_joint_index(&joint, skin) as i32;
            joint.children().map(move |n| (n, joint_index))
        })
        .filter(move |(node, _)| !is_joint(node, skin))
}
