use std::env;
use std::io;
use std::fs;
use std::path;

use image::GenericImageView;
use byteorder::{WriteBytesExt, LittleEndian};

#[derive(Clone, Copy)]
enum VbmColorMode {
    _1555 = 0,
    _4444 = 1,
    _565 = 2,
}

fn parse_color_mode(name: &str) -> VbmColorMode {
    match name {
        "1555" => VbmColorMode::_1555,
        "4444" => VbmColorMode::_4444,
        "565" => VbmColorMode::_565,
        _ => panic!("invalid color mode"),
    }
}

fn split_file_name(file_name: &str) -> (&str, &str) {
    if let Some(n) = file_name.rfind('.') {
        (&file_name[..n], &file_name[n..])
    } else {
        (file_name, "")
    }
}

fn build_frame_file_name(prefix: &str, frame_index: u32, dot_ext: &str) -> String {
    format!("{}-{:04}{}", prefix, frame_index, dot_ext)
}

fn count_frames(prefix: &str, dot_ext: &str) -> u32 {
    let mut i = 0;
    while path::Path::new(&build_frame_file_name(prefix, i, dot_ext)).exists() {
        i += 1;
    }
    i
}

fn write_frame<W: io::Write>(wrt: &mut W, clr_mode: VbmColorMode, img: &image::DynamicImage) -> Result<(), Box<dyn std::error::Error>> {
    for (_, _, pixel) in img.pixels() {
        let (r8, g8, b8, a8) = (pixel[0], pixel[1], pixel[2], pixel[3]);
        match clr_mode {
            VbmColorMode::_1555 => {
                let r5 = u16::from(r8) >> 3;
                let g5 = u16::from(g8) >> 3;
                let b5 = u16::from(b8) >> 3;
                let a1 = u16::from(a8) >> 7;
                let data = ((1 - a1) << 15) | (r5 << 10) | (g5 << 5) | b5;
                wrt.write_u16::<LittleEndian>(data)?;
            },
            VbmColorMode::_4444 => {
                let r4 = u16::from(r8) >> 4;
                let g4 = u16::from(g8) >> 4;
                let b4 = u16::from(b8) >> 4;
                let a4 = u16::from(a8) >> 4;
                let data = (a4 << 12) | (r4 << 8) | (g4 << 4) | b4;
                wrt.write_u16::<LittleEndian>(data)?;
            },
            VbmColorMode::_565 => {
                let r5 = u16::from(r8) >> 3;
                let g6 = u16::from(g8) >> 2;
                let b5 = u16::from(b8) >> 3;
                let data = (r5 << 11) | (g6 << 5) | b5;
                wrt.write_u16::<LittleEndian>(data)?;
            },
        }
    }
    Ok(())
}

fn make_vbm(color_mode_name: &str, framerate_str: &str, input_file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    const VBM_SIGNATURE: u32 = 0x6D62762E;
    const VBM_VERSION: u32 = 1;

    let clr_mode = parse_color_mode(color_mode_name);
    let framerate = framerate_str.parse::<u32>()?;

    let (prefix, dot_ext) = split_file_name(input_file_name);
    let num_frames = count_frames(prefix, dot_ext);

    let zero_frame_name = build_frame_file_name(prefix, 0, dot_ext);
    let zero_frame = image::open(zero_frame_name)?;
    let (width, height) = zero_frame.dimensions();

    let output_file_name = prefix.to_owned() + ".vbm";
    let mut wrt = io::BufWriter::new(fs::File::create(output_file_name)?);
    wrt.write_u32::<LittleEndian>(VBM_SIGNATURE)?;
    wrt.write_u32::<LittleEndian>(VBM_VERSION)?;
    wrt.write_u32::<LittleEndian>(width)?;
    wrt.write_u32::<LittleEndian>(height)?;
    wrt.write_u32::<LittleEndian>(clr_mode as u32)?;
    wrt.write_u32::<LittleEndian>(framerate)?;
    wrt.write_u32::<LittleEndian>(num_frames)?;
    wrt.write_u32::<LittleEndian>(0)?; // TODO: num_mipmaps

    for i in 0..num_frames {
        println!("Processing frame {}/{}...", i, num_frames);
        let frame_file_name = build_frame_file_name(prefix, i, dot_ext);
        let frame_img = image::open(frame_file_name)?;
        write_frame(&mut wrt, clr_mode, &frame_img)?;
    }
    Ok(())
}

fn main() {
    println!("makevbm 0.1 created by rafalh");
    let args = env::args().collect::<Vec<_>>();
    if env::args().len() != 4 {
        println!("Usage: {} colormode framerate input_file", args[0]);
        println!();
        println!("Available color modes:");
        println!("  1555 - 5 bits for each RGB channel and 1 bit for alpha channel");
        println!("  4444 - 4 bits for each RGB channel and 4 bit for alpha channel");
        println!("  565  - 5 bits for red and blue channels, 6 bits for green channel, no alpha channel");
        println!();
        println!("Framerate is only important for animated VBMs (those with more than one frame).");
        println!("Input file name is going to be suffixed with a zero-based frame number. For example when invoked:");
        println!("makevbm 565 10 miner.png");
        println!("The tool will attempt to read frames from miner-0000.png, miner-0001.png, miner-0002.png, ...");
        println!("Number of frames is determined automatically from the file system.");

        std::process::exit(1);
    }

    if let Err(e) = make_vbm(&args[1], &args[2], &args[3]) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
