// Fragments of code by natalie was used: https://git.agiri.ninja/natalie/rf2tools/

mod targa;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::cmp;
use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Result, Seek, SeekFrom, Write};

enum PegBmType {
    Mpeg2_16 = 1,
    Mpeg2_32 = 2,
    Rgba5551 = 3,
    Indexed8 = 4,
    Indexed4 = 5,
    Rgba8888 = 7,
}

enum PegPalType {
    Rgba5551 = 1,
    Rgba8888 = 2,
}

struct PegFileHeader {
    identifier: i32,
    version: i32,
    dir_block_size: i32,
    data_block_size: i32,
    num_bitmaps: i32,
    flags: i32,
    total_entries: i32,
    align_value: i32,
}

impl PegFileHeader {
    const PEG_ID: i32 = 0x564b4547;

    fn check(&self) {
        if self.identifier != Self::PEG_ID {
            panic!("Wrong file identifier: {:x}!", self.identifier);
        }
        if self.version < 6 {
            panic!("Unsupported version: {}", self.version);
        }
        if self.version > 6 {
            eprintln!("Warning: version {} is too new", self.version);
        }
    }

    fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        Ok(Self {
            identifier: rdr.read_i32::<LittleEndian>()?,
            version: rdr.read_i32::<LittleEndian>()?,
            dir_block_size: rdr.read_i32::<LittleEndian>()?,
            data_block_size: rdr.read_i32::<LittleEndian>()?,
            num_bitmaps: rdr.read_i32::<LittleEndian>()?,
            flags: rdr.read_i32::<LittleEndian>()?,
            total_entries: rdr.read_i32::<LittleEndian>()?,
            align_value: rdr.read_i32::<LittleEndian>()?,
        })
    }

    #[allow(unused)]
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_i32::<LittleEndian>(self.identifier)?;
        wrt.write_i32::<LittleEndian>(self.version)?;
        wrt.write_i32::<LittleEndian>(self.dir_block_size)?;
        wrt.write_i32::<LittleEndian>(self.data_block_size)?;
        wrt.write_i32::<LittleEndian>(self.num_bitmaps)?;
        wrt.write_i32::<LittleEndian>(self.flags)?;
        wrt.write_i32::<LittleEndian>(self.total_entries)?;
        wrt.write_i32::<LittleEndian>(self.align_value)?;
        Ok(())
    }
}

struct PegFileEntry {
    width: u16,
    height: u16,
    keg_bm_type: u8,
    keg_pal_type: u8,
    flags: u8,
    num_frames: u8,
    fps: u8,
    mip_levels: u8,
    mip_filter_value: i16,
    filename: String,
    data_offset: i32,
}

impl PegFileEntry {
    fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        let read_filename = |rdr: &mut R| -> Result<String> {
            let mut filename = [0u8; 48];
            rdr.read_exact(&mut filename)?;
            let filename_len = filename
                .iter()
                .position(|&c| c == b'\0')
                .unwrap_or(filename.len());
            Ok(String::from_utf8_lossy(&filename[..filename_len]).to_string())
        };
        Ok(Self {
            width: rdr.read_u16::<LittleEndian>()?,
            height: rdr.read_u16::<LittleEndian>()?,
            keg_bm_type: rdr.read_u8()?,
            keg_pal_type: rdr.read_u8()?,
            flags: rdr.read_u8()?,
            num_frames: rdr.read_u8()?,
            fps: rdr.read_u8()?,
            mip_levels: rdr.read_u8()?,
            mip_filter_value: rdr.read_i16::<LittleEndian>()?,
            filename: read_filename(rdr)?,
            data_offset: rdr.read_i32::<LittleEndian>()?,
        })
    }
}

fn print_peg_file_header(hdr: &PegFileHeader) {
    println!("PEG File Header:");
    println!("  identifier       {}", hdr.identifier);
    println!("  version          {}", hdr.version);
    println!("  dir_block_size   {}", hdr.dir_block_size);
    println!("  data_block_size  {}", hdr.data_block_size);
    println!("  num_bitmaps      {}", hdr.num_bitmaps);
    println!("  flags            {}", hdr.flags);
    println!("  total_entries    {}", hdr.total_entries);
    println!("  align_value      {}", hdr.align_value);
}

fn print_peg_entry(e: &PegFileEntry, i: i32) {
    println!("Bitmap {}:", i);
    println!("  filename:          {}", e.filename);
    println!("  width:             {}", e.width);
    println!("  height:            {}", e.height);
    println!("  keg_bm_type:       {}", e.keg_bm_type);
    println!("  keg_pal_type:      {}", e.keg_pal_type);
    println!("  flags:             {}", e.flags);
    println!("  num_frames:        {}", e.num_frames);
    println!("  fps:               {}", e.fps);
    println!("  mip_levels:        {}", e.mip_levels);
    println!("  mip_filter_value:  {}", e.mip_filter_value);
    println!("  data_offset:       {}", e.data_offset);
}

fn print_peg_file_info(pathname: &str) -> Result<()> {
    let mut rdr = BufReader::new(File::open(pathname)?);
    let hdr = PegFileHeader::read(&mut rdr)?;
    hdr.check();

    print_peg_file_header(&hdr);
    for i in 0..hdr.num_bitmaps {
        let entry = PegFileEntry::read(&mut rdr)?;
        print_peg_entry(&entry, i);
    }
    Ok(())
}

#[derive(Default, Copy, Clone)]
struct PalEntry {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

fn extract_peg_mipmap<R: Read>(
    rdr: &mut R,
    ent: &PegFileEntry,
    level: u8,
    frame: u8,
    pal: &[PalEntry; 256],
    output_dir: Option<&str>,
) -> Result<()> {
    let width = ent.width >> level;
    let height = ent.height >> level;
    let output_filename = format!(
        "{}/{}_{:04}_mip{}.tga",
        output_dir.unwrap_or("."),
        ent.filename.trim_end_matches(".tga"),
        frame,
        level
    );
    println!(
        "Writing mip level {} ({}x{}) to {}.",
        level, width, height, output_filename
    );
    let mut wrt = BufWriter::new(File::create(&output_filename)?);
    let (bpp, indexed) = if ent.keg_bm_type == PegBmType::Rgba5551 as u8 {
        (16, false)
    } else if ent.keg_bm_type == PegBmType::Indexed8 as u8
        || ent.keg_bm_type == PegBmType::Indexed4 as u8
    {
        (32, true)
    } else if ent.keg_bm_type == PegBmType::Rgba8888 as u8 {
        (32, false)
    } else {
        panic!("Unsupported keg_bm_type: {}", ent.keg_bm_type)
    };
    targa::TgaFileHeader::new(width as i16, height as i16, bpp, indexed).write(&mut wrt)?;

    if indexed {
        for pal_e in pal {
            wrt.write_u8(pal_e.b)?;
            wrt.write_u8(pal_e.g)?;
            wrt.write_u8(pal_e.r)?;
            wrt.write_u8(pal_e.a)?;
        }
    }

    for _ in 0..height {
        for x in 0..width {
            if ent.keg_bm_type == PegBmType::Rgba5551 as u8 {
                let pixel = rdr.read_u16::<LittleEndian>()?;
                wrt.write_u16::<LittleEndian>(pixel)?;
            } else if ent.keg_bm_type == PegBmType::Indexed8 as u8 {
                let pal_index = rdr.read_u8()?;
                wrt.write_u8(pal_index)?;
            } else if ent.keg_bm_type == PegBmType::Indexed4 as u8 {
                if x % 2 == 0 {
                    let pal_index = rdr.read_u8()?;
                    wrt.write_u8(pal_index & 0x0F)?;
                    if x + 1 < width {
                        wrt.write_u8((pal_index >> 4) & 0x0F)?;
                    }
                }
            } else if ent.keg_bm_type == PegBmType::Rgba8888 as u8 {
                let [red, green, blue, alpha] = [
                    rdr.read_u8()?,
                    rdr.read_u8()?,
                    rdr.read_u8()?,
                    rdr.read_u8()?,
                ];
                wrt.write_u8(red)?;
                wrt.write_u8(green)?;
                wrt.write_u8(blue)?;
                wrt.write_u8(alpha)?;
            } else {
                panic!("Unsupported keg_bm_type: {}", ent.keg_bm_type);
            }
        }
    }
    Ok(())
}

fn extract_mpeg_video<R: Read>(
    rdr: &mut R,
    e: &PegFileEntry,
    output_dir: Option<&str>,
) -> Result<()> {
    let total_size = rdr.read_u32::<LittleEndian>()?;
    let unk0 = rdr.read_u32::<LittleEndian>()?;
    let unk1 = rdr.read_u32::<LittleEndian>()?;
    let unk2 = rdr.read_u32::<LittleEndian>()?;
    println!(
        "MPEG2 video header: {:x} {:x} {} {}",
        total_size, unk0, unk1, unk2
    );

    let output_filename = format!("{}/{}.mpg", output_dir.unwrap_or("."), e.filename);
    let mut wrt = BufWriter::new(File::create(output_filename)?);

    let mut buf = [0u8; 4096];
    let mut bytes_left = (total_size - 16) as usize;
    while bytes_left > 0 {
        let bytes_to_read = cmp::min(bytes_left, buf.len());
        let bytes_read = rdr.read(&mut buf[..bytes_to_read])?;
        if bytes_read == 0 {
            break;
        }
        wrt.write_all(&buf[..bytes_read])?;
        bytes_left -= bytes_read;
    }
    Ok(())
}

fn extract_peg_bitmap<R: Read>(
    rdr: &mut R,
    e: &PegFileEntry,
    output_dir: Option<&str>,
) -> Result<()> {
    if e.keg_bm_type == PegBmType::Mpeg2_16 as u8 || e.keg_bm_type == PegBmType::Mpeg2_32 as u8 {
        extract_mpeg_video(rdr, e, output_dir)?;
        return Ok(());
    }

    for frame in 0..e.num_frames {
        let mut pal = [PalEntry::default(); 256];
        if e.keg_bm_type == PegBmType::Indexed8 as u8 || e.keg_bm_type == PegBmType::Indexed4 as u8
        {
            let num_pal_entries = if e.keg_bm_type == PegBmType::Indexed8 as u8 {
                256
            } else {
                16
            };
            for pal_ent in pal.iter_mut().take(num_pal_entries) {
                *pal_ent = if e.keg_pal_type == PegPalType::Rgba5551 as u8 {
                    let w = rdr.read_u16::<LittleEndian>()?;
                    PalEntry {
                        // TODO: rounding or remove conversion
                        r: (((w) & 0x1f) * 255 / 31) as u8,
                        g: (((w >> 5) & 0x1f) * 255 / 31) as u8,
                        b: (((w >> 10) & 0x1f) * 255 / 31) as u8,
                        a: (((w >> 15) & 0x1) * 255) as u8,
                    }
                } else if e.keg_pal_type == PegPalType::Rgba8888 as u8 {
                    PalEntry {
                        r: rdr.read_u8()?,
                        g: rdr.read_u8()?,
                        b: rdr.read_u8()?,
                        a: rdr.read_u8()?,
                    }
                } else {
                    panic!("Unknown keg_pal_type: {}", e.keg_pal_type)
                };
            }
        }
        for level in 0..e.mip_levels {
            extract_peg_mipmap(rdr, e, level, frame, &pal, output_dir)?;
        }
    }
    Ok(())
}

fn extract_peg_file(pathname: &str, output_dir: Option<&str>) -> Result<()> {
    let mut rdr = BufReader::new(File::open(pathname)?);
    let hdr = PegFileHeader::read(&mut rdr)?;
    hdr.check();

    let mut entries = Vec::new();
    for _ in 0..hdr.num_bitmaps {
        entries.push(PegFileEntry::read(&mut rdr)?)
    }

    for e in &entries {
        println!("Extracting {} (offset {:x})...", e.filename, e.data_offset);
        rdr.seek(SeekFrom::Start(e.data_offset.try_into().unwrap()))?;
        extract_peg_bitmap(&mut rdr, e, output_dir)?;
    }

    Ok(())
}

enum Operation {
    Info,
    Extract,
    Help,
    Version,
}

struct ParsedArgs {
    op: Operation,
    positional: Vec<String>,
    #[allow(dead_code)]
    verbose: bool,
    output_dir: Option<String>,
}

fn parse_args() -> ParsedArgs {
    let mut op_opt = None;
    let mut positional = Vec::<String>::new();
    let mut verbose = false;
    let mut output_dir = None;

    let mut args_it = env::args().skip(1);
    while let Some(arg) = args_it.next() {
        match arg.as_str() {
            "-h" => op_opt = Some(Operation::Help),
            "-v" => op_opt = Some(Operation::Version),
            "-x" => op_opt = Some(Operation::Extract),
            "-O" => output_dir = Some(args_it.next().expect("expected output dir")),
            "--verbose" => verbose = true,
            _ => positional.push(arg),
        }
    }

    let op = op_opt.unwrap_or({
        if !positional.is_empty() {
            Operation::Info
        } else {
            Operation::Help
        }
    });

    ParsedArgs {
        op,
        positional,
        verbose,
        output_dir,
    }
}

fn print_help() {
    println!("Usage:");
    println!("  peg file.peg");
    println!("    shows information about PEG file");
    println!("  peg -x file.peg");
    println!("    extract PEG file");
}

fn print_version() {
    println!("PEG Tool {} by Rafalh", env!("CARGO_PKG_VERSION"));
}

fn main() -> Result<()> {
    let args = parse_args();
    match args.op {
        Operation::Info => {
            for pathname in &args.positional {
                print_peg_file_info(pathname)?;
            }
        }
        Operation::Extract => {
            for pathname in &args.positional {
                extract_peg_file(pathname, args.output_dir.as_deref())?;
            }
        }
        Operation::Help => print_help(),
        Operation::Version => print_version(),
    }
    Ok(())
}
