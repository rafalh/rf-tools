mod char_anim;
mod io_utils;
mod material;
mod math_utils;
mod rfa;
mod rfg;
mod rfg_convert;
mod v3mc;
mod v3mc_convert;

use clap::ArgAction;
use clap::Parser;
use clap::ValueEnum;
use gltf::Buffer;
use math_utils::{Matrix3, Matrix4, Vector3};
use std::env;
use std::error::Error;
use std::f32;
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;
use std::vec::Vec;

type BoxResult<T> = Result<T, Box<dyn Error>>;

// glTF defines -X as right, RF defines +X as right
// Both glTF and RF defines +Y as up, +Z as forward

fn gltf_to_rf_vec(vec: [f32; 3]) -> [f32; 3] {
    // in GLTF negative X is right, in RF positive X is right
    [-vec[0], vec[1], vec[2]]
}

fn gltf_to_rf_quat(quat: [f32; 4]) -> [f32; 4] {
    // convert to RF coordinate system
    // it seems RF expects inverted quaternions...
    [-quat[0], quat[1], quat[2], quat[3]]
}

fn gltf_to_rf_face<T: Copy>(vindices: [T; 3]) -> [T; 3] {
    // because we convert from right-handed to left-handed order of vertices must be flipped to
    // fix backface culling
    [vindices[0], vindices[2], vindices[1]]
}

fn build_child_nodes_indices(doc: &gltf::Document) -> Vec<usize> {
    let mut child_indices: Vec<usize> = doc
        .nodes()
        .flat_map(|n| n.children().map(|n| n.index()))
        .collect();
    child_indices.dedup();
    child_indices
}

fn get_submesh_nodes(doc: &gltf::Document) -> Vec<gltf::Node<'_>> {
    let child_indices = build_child_nodes_indices(doc);
    doc.nodes()
        .filter(|n| n.mesh().is_some() && !child_indices.contains(&n.index()))
        .collect()
}

fn get_mesh_materials<'a>(mesh: &gltf::Mesh<'a>) -> Vec<gltf::Material<'a>> {
    let mut materials = mesh
        .primitives()
        .map(|prim| prim.material())
        .collect::<Vec<_>>();
    materials.dedup_by_key(|m| m.index());
    materials
}

fn get_primitive_vertex_count(prim: &gltf::Primitive) -> usize {
    prim.attributes()
        .find(|p| p.0 == gltf::mesh::Semantic::Positions)
        .map_or(0, |a| a.1.count())
}

fn count_mesh_vertices(mesh: &gltf::Mesh) -> usize {
    mesh.primitives()
        .map(|p| get_primitive_vertex_count(&p))
        .sum()
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

fn get_node_local_transform(node: &gltf::Node) -> glam::Mat4 {
    glam::Mat4::from_cols_array_2d(&node.transform().matrix())
}

struct Context {
    buffers: Vec<gltf::buffer::Data>,
    is_character: bool,
    args: Args,
    output_dir: PathBuf,
}

impl Context {
    fn get_buffer_data(&self, buffer: Buffer) -> Option<&[u8]> {
        Some(&*self.buffers[buffer.index()])
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Format {
    V3m,
    V3c,
    Rfg,
}

fn determine_output_format(args: &Args, is_character: bool) -> Format {
    args.format.unwrap_or_else(|| {
        let ext = args
            .output_file
            .as_ref()
            .and_then(|path| path.extension())
            .and_then(OsStr::to_str);
        match ext {
            Some("v3m") => Format::V3m,
            Some("v3c") => Format::V3c,
            Some("rfg") => Format::Rfg,
            _ => {
                if is_character {
                    Format::V3c
                } else {
                    Format::V3m
                }
            }
        }
    })
}

fn determine_output_file_name(args: &Args, output_format: Format) -> PathBuf {
    args.output_file.as_ref().map_or_else(
        || {
            let ext = match output_format {
                Format::V3m => "v3m",
                Format::V3c => "v3c",
                Format::Rfg => "rfg",
            };
            args.input_file.with_extension(ext)
        },
        |p| p.clone(),
    )
}

fn do_convert(args: Args) -> Result<(), Box<dyn Error>> {
    if args.verbose >= 1 {
        println!("Importing GLTF file: {}", args.input_file.display());
    }
    let input_path = Path::new(&args.input_file);
    let gltf = gltf::Gltf::open(input_path)?;
    let gltf::Gltf { document, blob } = gltf;

    if args.verbose >= 2 {
        println!("Importing GLTF buffers");
    }
    let buffers = gltf::import_buffers(&document, input_path.parent(), blob)?;
    let skin_opt = document.skins().next();
    let is_character = skin_opt.is_some();

    let output_format = determine_output_format(&args, is_character);
    let output_file_name = determine_output_file_name(&args, output_format);
    let output_dir = output_file_name.parent().unwrap().to_owned();

    if args.verbose >= 1 {
        println!("Exporting mesh: {}", output_file_name.display());
    }
    let ctx = Context {
        buffers,
        is_character,
        args,
        output_dir,
    };
    if output_format == Format::Rfg {
        let rfg = rfg_convert::convert_gltf_to_rfg(&document, &ctx)?;
        let file = File::create(output_file_name)?;
        let mut wrt = BufWriter::new(file);
        rfg.write(&mut wrt)?;
    } else {
        let v3m = v3mc_convert::convert_gltf_to_v3mc(&document, &ctx)?;
        let file = File::create(output_file_name)?;
        let mut wrt = BufWriter::new(file);
        v3m.write(&mut wrt)?;

        if let Some(skin) = skin_opt {
            for (i, anim) in document.animations().enumerate() {
                char_anim::convert_animation_to_rfa(&anim, i, &skin, &ctx)?;
            }
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[clap(author, version, about, about = "GLTF to V3M/V3C/RFG converter")]
pub struct Args {
    /// Input GLTF filename
    input_file: PathBuf,

    /// Output filename
    output_file: Option<PathBuf>,

    /// Output file format. If not specified format is detected from output file extension and input file content.
    #[clap(short, long)]
    format: Option<Format>,

    /// Default animation weight to be used when it is not defined in bone extras.
    /// Default is 10 if bone is animated, 2 otherwise
    #[clap(long)]
    anim_weight: Option<f32>,

    /// Default ramp in time in seconds to be used when it is not defined in bone extras.
    /// Default is 0.1(6) for death animation, 0.1 fot other animations
    #[clap(long)]
    ramp_in_time: Option<f32>,

    /// Default ramp in time in seconds to be used when it is not defined in bone extras.
    /// Default is 0 for death animation, 0.1 fot other animations
    #[clap(long)]
    ramp_out_time: Option<f32>,

    /// Enable verbose output. Can be used 2 times to increase verbosity
    #[clap(short, long, action = ArgAction::Count)]
    verbose: u8,
}

fn main() {
    let args = Args::parse();

    if args.verbose >= 1 {
        println!("vmesh {}", env!("CARGO_PKG_VERSION"));
    }

    if let Err(e) = do_convert(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
