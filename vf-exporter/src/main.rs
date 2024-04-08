mod tga;
mod vf;

use binrw::{BinReaderExt, BinWriterExt};
use clap::Parser;
use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use tga::TgaHeader;
use vf::{VfCharDesc, VfFormat, VfHeader, VfPalette};

use crate::vf::VfKernPair;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn print_vf_metadata(hdr: &VfHeader) {
    println!("VF Metadata:");
    println!("Version: {}", hdr.version);
    println!("Format: {:?}", hdr.format);
    println!("First ASCII: {}", hdr.first_ascii);
    println!("Character size: {}", hdr.default_spacing);
    println!("Number of chars: {}", hdr.num_chars);
    println!("Number of kerning pairs: {}", hdr.num_kern_pairs);
    println!();
}

fn determine_output_image_size(num_pixels: u32) -> (u32, u32) {
    let s = match num_pixels {
        0..=0x3FF => 64,
        0x400..=0xFFF => 128,
        0x1000..=0x3FFF => 256,
        _ => 512,
    };
    (s, s)
}

fn write_font_tga(
    filename: &Path,
    vf_hdr: &VfHeader,
    char_desc: &[VfCharDesc],
    src_data: &[u8],
    palette: Option<&VfPalette>,
) -> Result<()> {
    let vf_bytes_per_pixel: usize = if vf_hdr.format == VfFormat::Rgba4444 {
        2
    } else {
        1
    };
    let num_pixels = vf_hdr.pixel_data_size / vf_bytes_per_pixel as u32;
    let (w, h) = determine_output_image_size(num_pixels);
    let tga_hdr = TgaHeader::new(w.try_into()?, h.try_into()?, 32);

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writer.write_le(&tga_hdr)?;

    let mut bitmap_data = vec![[0u8; 4]; (w * h) as usize]; // RGBA

    let mut dst_x = 0;
    let mut dst_y = 1;
    for char_idx in 0..vf_hdr.num_chars as usize {
        if dst_x + char_desc[char_idx].width + 3 > w {
            dst_x = 0;
            dst_y += vf_hdr.height + 3;
        }
        if dst_y + vf_hdr.height + 3 > h {
            return Err("Font is too big".into());
        }

        // Top and buttom border
        let border_clr = [0, 255, 0, 255];
        for off_x in 0..char_desc[char_idx].width + 2 {
            bitmap_data[(dst_y * w + dst_x + off_x) as usize] = border_clr;
            bitmap_data[((dst_y + 1 + vf_hdr.height) * w + dst_x + off_x) as usize] = border_clr;
        }
        // Left and right border
        for off_y in 0..vf_hdr.height + 2 {
            bitmap_data[((dst_y + off_y) * w + dst_x) as usize] = border_clr;
            bitmap_data[((dst_y + off_y) * w + dst_x + 1 + char_desc[char_idx].width) as usize] =
                border_clr;
        }

        // Copy character pixels
        let mut src_offset = char_desc[char_idx].pixel_data_offset as usize;
        for off_y in 1..vf_hdr.height + 1 {
            for off_x in 1..char_desc[char_idx].width + 1 {
                let dst_offset = ((dst_y + off_y) * w + dst_x + off_x) as usize;
                let dst_pixel_bytes = &mut bitmap_data[dst_offset];
                match vf_hdr.format {
                    VfFormat::Mono4 => {
                        let src_pixel = src_data[src_offset];
                        let value = (14.min(src_pixel) as u32 * 255 / 14) as u8;
                        dst_pixel_bytes[0] = value;
                        dst_pixel_bytes[1] = value;
                        dst_pixel_bytes[2] = value;
                        dst_pixel_bytes[3] = value;
                    }
                    VfFormat::Rgba4444 => {
                        let src_pixel =
                            u16::from_le_bytes(src_data[src_offset..src_offset + 1].try_into()?);
                        let r = (src_pixel) & 0xF;
                        let g = (src_pixel >> 4) & 0xF;
                        let b = (src_pixel >> 8) & 0xF;
                        let a = (src_pixel >> 12) & 0xF;
                        dst_pixel_bytes[0] = (r * 0xFF / 0xF) as u8;
                        dst_pixel_bytes[1] = (g * 0xFF / 0xF) as u8;
                        dst_pixel_bytes[2] = (b * 0xFF / 0xF) as u8;
                        dst_pixel_bytes[3] = (a * 0xFF / 0xF) as u8;
                    }
                    VfFormat::Indexed => {
                        let index = src_data[src_offset];
                        *dst_pixel_bytes = palette.unwrap().entries[usize::from(index)];
                    }
                }
                src_offset += vf_bytes_per_pixel;
            }
        }
        dst_x += char_desc[char_idx].width + 3;
    }

    for pixel in bitmap_data {
        writer.write_all(&pixel)?;
    }

    Ok(())
}

fn export_font(vf_filename: &Path, output_dir: &Path) -> Result<()> {
    println!("Processing {}...", vf_filename.display());
    let file = File::open(vf_filename)?;
    let mut reader = BufReader::new(file);

    let hdr: VfHeader = reader.read_le()?;
    print_vf_metadata(&hdr);

    let mut kern_pairs: Vec<VfKernPair> = Vec::new();
    for _ in 0..hdr.num_kern_pairs {
        kern_pairs.push(reader.read_le()?);
    }

    let mut char_desc: Vec<VfCharDesc> = Vec::new();
    for _ in 0..hdr.num_chars {
        char_desc.push(reader.read_le()?);
    }

    let mut pixel_data = vec![0u8; hdr.pixel_data_size as usize];
    reader.read_exact(&mut pixel_data)?;

    let palette: Option<VfPalette> = if hdr.format == VfFormat::Indexed {
        Some(reader.read_le()?)
    } else {
        None
    };

    let output_path = output_dir
        .join(vf_filename.file_stem().unwrap())
        .with_extension("tga");
    write_font_tga(
        &output_path,
        &hdr,
        &char_desc,
        &pixel_data,
        palette.as_ref(),
    )?;
    Ok(())
}

#[derive(Parser, Debug)]
#[clap(author, version, about, about = "VF exporter")]
pub struct Args {
    /// Input VF files
    vf_files: Vec<PathBuf>,

    /// Output directory
    #[clap(short = 'O', default_value = ".")]
    output_dir: PathBuf,

    /// Verbose output
    #[clap(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("vf-exporter {}", env!("CARGO_PKG_VERSION"));
    }

    for input_file in &args.vf_files {
        export_font(input_file, &args.output_dir)?;
    }

    Ok(())
}
