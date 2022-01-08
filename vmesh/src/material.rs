use std::convert::TryInto;
use std::f32;
use std::path::Path;
use crate::v3mc;

pub(crate) fn compute_render_mode_for_material(material: &gltf::material::Material) -> u32 {
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

fn change_texture_ext_to_tga(name: &str) -> String {
    String::from(Path::new(name).with_extension("tga").file_name().unwrap().to_string_lossy())
}

fn get_material_base_color_texture_name(material: &gltf::material::Material) -> String {
    const DEFAULT_TEXTURE: &str = "Rck_Default.tga";
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
    eprintln!("Cannot obtain texture name for material {} (materials without base color texture are not supported)",
        material.index().unwrap_or(0));
    DEFAULT_TEXTURE.into()
}

fn get_material_self_illumination(mat: &gltf::Material) -> f32 {
    mat.emissive_factor().iter().copied().fold(0_f32, f32::max)
}

pub(crate) fn convert_material(mat: &gltf::Material) -> v3mc::Material {
    let tex_name = get_material_base_color_texture_name(mat);
    let self_illumination = get_material_self_illumination(mat);
    let specular_level = mat.pbr_specular_glossiness()
        .map_or_else(
            || mat.pbr_metallic_roughness().metallic_factor(),
            |spec_glos| spec_glos.specular_factor().iter().copied().fold(0_f32, f32::max),
        );
    let glossiness = mat.pbr_specular_glossiness()
        .map_or_else(
            || 1.0 - mat.pbr_metallic_roughness().roughness_factor(), 
            |spec_glos| spec_glos.glossiness_factor(),
        );

    v3mc::Material{
        tex_name,
        self_illumination,
        specular_level,
        glossiness,
        flags: 0x11,
        ..v3mc::Material::default()
    }
}

pub(crate) fn create_mesh_material_ref(material: &gltf::Material, lod_mesh_materials: &[gltf::Material]) -> v3mc::MeshTextureRef {
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
