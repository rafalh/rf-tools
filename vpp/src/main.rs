use std::env;
use std::fs::File;
use std::u32;
use std::io::{Read, Write, Seek, SeekFrom, BufReader, BufRead, BufWriter, Error, ErrorKind, Result};
use std::cmp;
use std::convert::TryInto;
use std::path::Path;

#[macro_use]
extern crate log;

const VPP_BLOCK_SIZE: usize = 0x800;
const VPP_VERSION: u32 = 1;
const VPP_SIGNATURE: u32 = 0x51890ACE;

trait ReadLe : Read {
    fn read_u32_le(&mut self) -> Result<u32> {
        let mut temp = [0u8; 4];
        self.read_exact(&mut temp)?;
        Ok(u32::from_le_bytes(temp))
    }
}

trait WriteLe : Write {
    fn write_u32_le(&mut self, val: u32) -> Result<()> {
        self.write_all(&val.to_le_bytes())
    }
}

impl<T> ReadLe for T where T: Read {}
impl<T> WriteLe for T where T: Write {}

struct VppHeader {
    signature: u32,
    version: u32,
    num_files: u32,
    size: u32,
}

struct VppEntry {
    name: Vec<u8>,
    size: u32,
}

impl VppHeader {
    fn read<R: Read>(rdr: &mut R) -> Result<VppHeader> {
        let signature = rdr.read_u32_le()?;
        if signature != VPP_SIGNATURE {
            return Err(Error::new(ErrorKind::Other, format!("invalid file signature {}", signature)));
        }
        let version = rdr.read_u32_le()?;
        if version != VPP_VERSION {
            return Err(Error::new(ErrorKind::Other, format!("unsupported version {}", version)));
        }
        let num_files = rdr.read_u32_le()?;
        let size = rdr.read_u32_le()?;
        Ok(VppHeader{
            signature,
            version,
            num_files,
            size,
        })
    }

    fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32_le(self.signature)?;
        wrt.write_u32_le(self.version)?;
        wrt.write_u32_le(self.num_files)?;
        wrt.write_u32_le(self.size)?;
        Ok(())
    }
}

impl VppEntry {

    const NAME_MAX_LEN: usize = 60;

    fn read<R: Read>(rdr: &mut R) -> Result<VppEntry> {
        let mut name_buf = [0u8; Self::NAME_MAX_LEN];
        rdr.read_exact(&mut name_buf)?;
        let size = rdr.read_u32_le()?;
        let name = name_buf.iter().cloned().take_while(|b| b != &0u8).collect();
        Ok(VppEntry{ name, size })
    }

    fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        let mut name_buf = [0u8; Self::NAME_MAX_LEN];
        name_buf[..self.name.len()].copy_from_slice(&self.name);
        wrt.write_all(&mut name_buf)?;
        wrt.write_u32_le(self.size)?;
        Ok(())
    }
}

fn process_file_list(file_list: Vec<String>) -> Result<Vec<String>> {
    debug!("Processing file list");
    let mut result = Vec::new();
    for filename in file_list {
        if filename.starts_with("@") {
            let file = File::open(&filename[1..])?;
            for line_result in BufReader::new(file).lines() {
                let line = line_result?;
                let trimmed_line = line.trim();
                if !trimmed_line.is_empty() {
                    result.push(trimmed_line.to_string());
                }
            }
        } else {
            result.push(filename);
        }
    }
    Ok(result)
}

fn transform_filename_for_dep_file(filename: &str) -> Result<String> {
    let abs_path = std::fs::canonicalize(filename)?;
    Ok(abs_path.display().to_string().replace(" ", "\\ "))
}

fn create_dep_file(packfile_path: &str, file_list: &Vec<String>) -> Result<()> {
    debug!("Creating dep file");
    let mut file = BufWriter::new(File::create(packfile_path.to_string() + ".d")?);
    write!(file, "{}:", transform_filename_for_dep_file(packfile_path)?)?;
    for fname in file_list {
        write!(file, " {}", transform_filename_for_dep_file(fname)?)?;
    }
    Ok(())
}

fn create_vpp(packfile_path: &str, file_list: &Vec<String>, verbose: bool) -> Result<()> {
    debug!("Opening output file {}", packfile_path);
    let mut file = File::create(packfile_path)?;

    debug!("Writing file header");
    let mut hdr = VppHeader {
        signature: VPP_SIGNATURE,
        version: VPP_VERSION,
        num_files: file_list.len() as u32,
        size: 0,
    };
    let mut block = [0u8; VPP_BLOCK_SIZE];
    hdr.write(&mut block.as_mut())?;
    file.write_all(&block)?;

    debug!("Writing entries");
    let mut block_wrt: &mut [u8] = &mut block;
    for fname in file_list {
        if block_wrt.is_empty() {
            file.write_all(&block)?;
            block_wrt = &mut block;
        }

        let size = std::fs::metadata(fname)?.len();
        let basename = Path::new(fname).file_name().unwrap().to_string_lossy();
        let entry = VppEntry {
            name: basename.as_bytes().to_vec(),
            size: size.try_into().unwrap(),
        };
        entry.write(&mut block_wrt)?;
    }
    if block_wrt.len() < block.len() {
        file.write_all(&block)?;
    }

    debug!("Writing data");
    for fname in file_list {
        if verbose {
            println!("Packing {}", fname);
        }
        let mut input_file = File::open(fname)?;
        block_wrt = &mut block;
        loop {
            if block_wrt.is_empty() {
                debug!("Writing data block");
                file.write_all(&block)?;
                block_wrt = &mut block;
            }
            let num_read_bytes = input_file.read(block_wrt)?;
            if num_read_bytes == 0 {
                break;
            }
            block_wrt = &mut block_wrt[num_read_bytes..];
        }
        if block_wrt.len() < block.len() {
            debug!("Writing data block");
            file.write_all(&block)?;
        }
    }

    let pos = file.seek(SeekFrom::Current(0))?;
    file.seek(SeekFrom::Start(0))?;
    hdr.size = pos as u32;
    hdr.write(&mut file)?;

    Ok(())
}

fn extract_vpp(packfile_path: &str, output_dir: Option<&str>, verbose: bool) -> Result<()> {
    debug!("Opening input packfile {}", packfile_path);
    let mut file = File::open(packfile_path)?;

    debug!("Reading file header");
    let mut hdr_block = [0u8; VPP_BLOCK_SIZE];
    file.read_exact(&mut hdr_block)?;
    let hdr = VppHeader::read(&mut hdr_block.as_ref())?;

    debug!("Reading entries");
    let mut block = [0u8; VPP_BLOCK_SIZE];
    let mut block_rdr: &[u8] = &[];
    let mut entries = Vec::<VppEntry>::new();

    for _ in 0..hdr.num_files {
        if block_rdr.len() == 0 {
            file.read_exact(&mut block)?;
            block_rdr = &block;
        }
        let entry = VppEntry::read(&mut block_rdr)?;
        entries.push(entry);
    }

    debug!("Reading data");
    for entry in entries {
        let num_blocks = (entry.size as usize + VPP_BLOCK_SIZE - 1) / VPP_BLOCK_SIZE;
        let name_str = String::from_utf8_lossy(&entry.name);
        let output_path = output_dir
            .map(|dir| dir.to_owned() + "/" + &name_str)
            .unwrap_or_else(|| name_str.to_string());
        if verbose {
            println!("Extracting {}", output_path);
        }
        let mut output_file = File::create(output_path)?;
        let mut bytes_left = entry.size as usize;
        for _ in 0..num_blocks {
            file.read_exact(&mut block)?;
            let bytes_to_write = cmp::min( VPP_BLOCK_SIZE, bytes_left);
            output_file.write_all(&block[..bytes_to_write])?;
            bytes_left -= bytes_to_write;
        }
    }

    Ok(())
}

fn format_size(bytes: u32) -> String {
    if bytes < 1024 {
        return format!("{} B", bytes);
    }
    let kb = bytes / 1024;
    if kb < 1024 {
        return format!("{} KB", kb);
    }
    let mb = kb / 1024;
    return format!("{} MB", mb);

}

fn list_vpp_content(packfile_path: &str) -> Result<()> {
    let mut file = File::open(packfile_path)?;
    let mut hdr_block = [0u8; VPP_BLOCK_SIZE];
    file.read_exact(&mut hdr_block)?;
    let hdr = VppHeader::read(&mut hdr_block.as_ref())?;
    for _ in 0..hdr.num_files {
        let entry = VppEntry::read(&mut file)?;
        let name_str = String::from_utf8_lossy(&entry.name);
        println!("{:60} {}", name_str, format_size(entry.size));
    }
    Ok(())
}

fn help() {
    println!("Usage:");
    println!("  vpp -c vpp_path files...    - create packfile");
    println!("  vpp -x vpp_path             - extract packfile");
    println!("  vpp -l vpp_path             - list packfile content");
    println!("Additional options:");
    println!("  --dep-info  - write vpp dependencies into .d file using Makefile syntax");
}

fn version() {
    println!("VPP tool v0.1 created by Rafalh");
}

enum Mode
{
    Create,
    Extract,
    List,
    Help,
    Version,
}

struct ParsedArgs
{
    mode: Mode,
    positional_args: Vec<String>,
    dep_info: bool,
    verbose: bool,
}

fn parse_args() -> ParsedArgs {
    let mut mode = Mode::Help;
    let mut positional_args = Vec::<String>::new();
    let mut dep_info = false;
    let mut verbose = false;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "-c" => mode = Mode::Create,
            "-x" => mode = Mode::Extract,
            "-l" => mode = Mode::List,
            "-h" => mode = Mode::Help,
            "-v" => mode = Mode::Version,
            "--dep-info" => dep_info = true,
            "--verbose" => verbose = true,
            _ => positional_args.push(arg),
        }
    }

    ParsedArgs { 
        mode, 
        positional_args, 
        dep_info,
        verbose,
    }
}

fn main() -> Result<()> {
    let args = parse_args();
    match args.mode {
        Mode::Create => {
            let vpp_path = args.positional_args.first().unwrap();
            let file_list = process_file_list(args.positional_args.iter().cloned().skip(1).collect::<Vec<_>>())?;
            create_vpp(vpp_path, &file_list, args.verbose)?;
            if args.dep_info {
                create_dep_file(vpp_path, &file_list)?;
            }
        },
        Mode::List => {
            let vpp_path = args.positional_args.first().unwrap();
            list_vpp_content(vpp_path)?;
        },
        Mode::Extract => {
            let vpp_path = args.positional_args.first().unwrap();
            extract_vpp(vpp_path, None, args.verbose)?;
        },
        Mode::Help => help(),
        Mode::Version => version(),
    };
    Ok(())
}
