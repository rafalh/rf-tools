// Fragments of code by natalie was used: https://git.agiri.ninja/natalie/rf2tools/

mod targa;

use std::io::{Read, Write, Result, BufReader, BufWriter, Seek, SeekFrom};
use std::env;
use std::fs::File;
use std::convert::TryInto;
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};

enum PegBmType {
    Rgba5551 = 3,
    Indexed  = 4,
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
            let filename_len = filename.iter().position(|&c| c == b'\0').unwrap_or(filename.len());
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

fn extract_peg_mipmap<R: Read>(rdr: &mut R, e: &PegFileEntry, level: u8, frame: u8, pal: &[PalEntry; 256], output_dir: &Option<String>) -> Result<()> {
    let w = e.width >> level;
    let h = e.height >> level;
    println!("level {} w {} h {}", level, w, h);
    let output_filename = format!(
        "{}/{}_{:04}_mip{}.tga",
        output_dir.clone().unwrap_or(".".to_string()),
        e.filename.trim_end_matches(".tga"),
        frame,
        level
    );
    let mut wrt = BufWriter::new(File::create(&output_filename)?);
    let (bpp, indexed) = if e.keg_bm_type == PegBmType::Rgba5551 as u8 {
        (16, false)
    } else if e.keg_bm_type == PegBmType::Indexed as u8 {
        (32, true)
    } else if e.keg_bm_type == PegBmType::Rgba8888 as u8 {
        (32, false)
    } else {
        panic!("Unsupported keg_bm_type: {}", e.keg_bm_type)
    };
    targa::TgaFileHeader::new(w as i16, h as i16, bpp, indexed).write(&mut wrt)?;

    if indexed {
        for pal_e in pal {
            wrt.write_u8(pal_e.b)?;
            wrt.write_u8(pal_e.g)?;
            wrt.write_u8(pal_e.r)?;
            wrt.write_u8(pal_e.a)?;
        }
    }

    for _ in 0..h {
        for _ in 0..w {
            if e.keg_bm_type == PegBmType::Rgba5551 as u8 {
                let w = rdr.read_u16::<LittleEndian>()?;
                wrt.write_u16::<LittleEndian>(w)?;
            } else if e.keg_bm_type == PegBmType::Indexed as u8 {
                let index = rdr.read_u8()?;
                wrt.write_u8(index)?;
            } else if e.keg_bm_type == PegBmType::Rgba8888 as u8 {
                let [r, g, b, a] = [
                    rdr.read_u8()?,
                    rdr.read_u8()?,
                    rdr.read_u8()?,
                    rdr.read_u8()?,
                ];
                wrt.write_u8(r)?;
                wrt.write_u8(g)?;
                wrt.write_u8(b)?;
                wrt.write_u8(a)?;
            } else {
                panic!("Unsupported keg_bm_type: {}", e.keg_bm_type);
            }
        }
    }
    Ok(())
}

fn extract_peg_bitmap<R: Read>(rdr: &mut R, e: &PegFileEntry, output_dir: &Option<String>) -> Result<()> {
    for frame in 0..e.num_frames {
        let mut pal = [PalEntry::default(); 256];
        if e.keg_bm_type == PegBmType::Indexed as u8 {
            println!("Extracting palette");
            for i in 0..256 {
                pal[i] = if e.keg_pal_type == PegPalType::Rgba5551 as u8 {
                    let w = rdr.read_u16::<LittleEndian>()?;
                    PalEntry { // TODO: rounding or remove conversion
                        r: (((w >> 0) & 0x1f) * 255 / 31) as u8,
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

fn extract_peg_file(pathname: &str, output_dir: &Option<String>) -> Result<()> {
    let mut rdr = BufReader::new(File::open(pathname)?);
    let hdr = PegFileHeader::read(&mut rdr)?;
    hdr.check();

    let mut entries = Vec::new();
    for _ in 0..hdr.num_bitmaps {
        entries.push(PegFileEntry::read(&mut rdr)?)
    }


    for e in &entries {
        println!("Extracting {}...", e.filename);
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

    let op = op_opt.unwrap_or_else(|| {
        match positional.len() {
            1 => Operation::Info,
            _ => Operation::Help,
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
    println!("PEG Tool v0.1 by rafalh");
}

fn main() -> Result<()> {
    let args = parse_args();
    match args.op {
        Operation::Info => print_peg_file_info(&args.positional[0])?,
        Operation::Extract => extract_peg_file(&args.positional[0], &args.output_dir)?,
        Operation::Help => print_help(),
        Operation::Version => print_version()
    }
    Ok(())
}
