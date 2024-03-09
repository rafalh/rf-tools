use std::{error::Error, fmt::Display, fs::File, io::{BufReader, BufWriter, Read, Write}, path::{Path, PathBuf}};

use binrw::{BinRead, BinReaderExt, BinWrite, BinWriterExt};
use clap::Parser;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(BinRead)]
#[br(magic = b".vbm")]
struct VbmHeader {
    version: u32,      // RF uses 1 and 2, makeVBM tool always creates files with version 1 */
    width: u32,        // nominal image width
    height: u32,       // nominal image height
    format: VbmColorFormat, // pixel data format
    fps: u32,          // frames per second, ignored if num_frames == 1
    num_frames: u32,   // number of frames, always 1 for not animated VBM
    num_mipmaps: u32,  // number of mipmap levels except for the full size (level 0)
}

#[derive(Clone, Copy, Eq, PartialEq, BinRead)]
#[br(repr = u32)]
enum VbmColorFormat
{
    _1555 = 0,
    _4444 = 1,
    _565  = 2,
}

impl Display for VbmColorFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            VbmColorFormat::_1555 => "1555",
            VbmColorFormat::_4444 => "4444",
            VbmColorFormat::_565 => "565",
        };
        f.write_str(s)
    }
}

fn print_vbm_metadata(hdr: &VbmHeader) {
    println!("VBM Metadata:");
    println!("Version: {}", hdr.version);
    println!("Size: {}x{}", hdr.width, hdr.height);
    println!("Format: {}", hdr.format);
    println!("FPS: {}", hdr.fps);
    println!("Number of frames: {}", hdr.num_frames);
    println!("Number of mipmaps: {}", hdr.num_mipmaps);
    println!();
}

#[derive(BinWrite)]
struct TgaHeader {
   idlength: i8,
   colourmaptype: i8,
   datatypecode: i8,
   colourmaporigin: u16,
   colourmaplength: u16,
   colourmapdepth: i8,
   x_origin: u16,
   y_origin: u16,
   width: i16,
   height: i16,
   bitsperpixel: i8,
   imagedescriptor: i8,
}

impl TgaHeader {
    fn new(w: i16, h: i16, bits_per_pixel: i8) -> Self {
        Self { 
            idlength: 0,
            colourmaptype: 0,
            datatypecode: 2,
            colourmaporigin: 0,
            colourmaplength: 0,
            colourmapdepth: 0,
            x_origin: 0,
            y_origin: 0,
            width: w,
            height: h,
            bitsperpixel: bits_per_pixel,
            imagedescriptor: 1 << 5, // Origin in upper left-hand corner
        }
    }
}

fn write_tga_frame(filename: &Path, pixel_data: &[u8], w: u32, h: u32, vbm_hdr: &VbmHeader) -> Result<()> {
    let bits_per_pixel = if vbm_hdr.format == VbmColorFormat::_565 { 24 } else { 32 };
    let tga_hdr = TgaHeader::new(w.try_into()?, h.try_into()?, bits_per_pixel);

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writer.write_le(&tga_hdr)?;

    for i in 0..pixel_data.len() / 2 {
        let pixel_bytes = <[u8; 2]>::try_from(&pixel_data[i * 2..(i + 1) * 2])?;
        let input_pixel = u16::from_le_bytes(pixel_bytes);
        match vbm_hdr.format {
            VbmColorFormat::_1555 => {
                let output_pixel = [
                    (u32::from((input_pixel >> 0)  & 0x1F) * 255 / 0x1F) as u8, // B
                    (u32::from((input_pixel >> 5)  & 0x1F) * 255 / 0x1F) as u8, // G
                    (u32::from((input_pixel >> 10) & 0x1F) * 255 / 0x1F) as u8, // R
                    (u32::from(!(input_pixel >> 15)) * 255) as u8, // A
                ];
                writer.write_all(&output_pixel)?;
            }
            VbmColorFormat::_4444 => {
                let output_pixel = [
                    (u32::from((input_pixel >> 0)  & 0xF) * 255 / 0xF) as u8, // B
                    (u32::from((input_pixel >> 4)  & 0xF) * 255 / 0xF) as u8, // G
                    (u32::from((input_pixel >> 8)  & 0xF) * 255 / 0xF) as u8, // R
                    (u32::from((input_pixel >> 12) & 0xF) * 255 / 0xF) as u8, // A
                ];
                writer.write_all(&output_pixel)?;
            }
            VbmColorFormat::_565 => {
                let output_pixel = [
                    (u32::from((input_pixel >> 0)  & 0xF) * 255 / 0x1F) as u8, // B
                    (u32::from((input_pixel >> 5)  & 0xF) * 255 / 0x3F) as u8, // G
                    (u32::from((input_pixel >> 11) & 0xF) * 255 / 0x1F) as u8, // R
                ];
                writer.write_all(&output_pixel)?;
            }
        }
    }

    Ok(())
}

fn build_output_filename(prefix: &Path, mipmap_idx: u32, frame_idx: u32) -> PathBuf {
    let mut file_name = prefix.file_name().unwrap().to_owned();
    file_name.push("-");
    if mipmap_idx > 0 {
        let c: char = (u32::from('a') + mipmap_idx - 1).try_into().unwrap();
        file_name.push(c.to_string());
    }
    file_name.push(format!("{:04}.tga", frame_idx));
    prefix.with_file_name(file_name)
}

fn export_vbm(vbm_filename: &Path, output_dir: &Path, verbose: bool) -> Result<()> {
    let file = File::open(vbm_filename)?;
    let mut vbm_reader = BufReader::new(file);
    let hdr: VbmHeader = vbm_reader.read_le()?;
    if verbose {
        println!("Processing {}...", vbm_filename.display());
        print_vbm_metadata(&hdr);
    }
    let prefix = output_dir.join(vbm_filename.file_stem().unwrap());
    let mut pixel_data = vec![0u8; (hdr.width * hdr.height * 2) as usize]; // 16 bit
    
    for frame_idx in 0..hdr.num_frames {
        let mut w = hdr.width;
        let mut h = hdr.height;
        
        for mipmap_idx in 0..hdr.num_mipmaps + 1 {
            let data_len = (w * h * 2) as usize;
            vbm_reader.read_exact(&mut pixel_data[..data_len])?;

            let output_filename = build_output_filename(&prefix, mipmap_idx, frame_idx);
            write_tga_frame(&output_filename, &pixel_data[..data_len], w, h, &hdr)?;

            w = 1.max(w / 2);
            h = 1.max(h / 2);
        }
    }

    println!("makevbm {} {} {}.tga", hdr.format, hdr.fps, prefix.display());
    Ok(())
}

#[derive(Parser, Debug)]
#[clap(author, version, about, about = "VBM exporter")]
pub struct Args {
    /// Input VBM files
    #[clap(required = true)]
    vbm_files: Vec<PathBuf>,

    /// Output directory
    #[clap(short = 'O', default_value = ".")]
    output_dir: PathBuf,

    /// Verbose output
    #[clap(short, long)]
    verbose: bool
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("vbm-exporter {}", env!("CARGO_PKG_VERSION"));
    }

    for input_file in &args.vbm_files {
        export_vbm(input_file, &args.output_dir, args.verbose)?;
    }

    Ok(())
}
