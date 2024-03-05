#![allow(unused_imports)]

mod wave;
mod adpcm;

use std::env;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter, Result};
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};
use adpcm::Ps2AdpcmDecoder;

enum Operation
{
    Info,
    Convert,
    Help,
    Version,
}

struct ParsedArgs
{
    op: Operation,
    positional: Vec<String>,
    #[allow(dead_code)]
    verbose: bool,
    old: bool,
}

fn parse_args() -> ParsedArgs {
    let mut op_opt = None;
    let mut positional = Vec::<String>::new();
    let mut verbose = false;
    let mut old = false;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "-h" => op_opt = Some(Operation::Help),
            "-v" => op_opt = Some(Operation::Version),
            "-i" => op_opt = Some(Operation::Info),
            "--old" => old = true,
            "--verbose" => verbose = true,
            _ => positional.push(arg),
        }
    }

    let op = op_opt.unwrap_or({
        match positional.len() {
            1 => Operation::Info,
            2 => Operation::Convert,
            _ => Operation::Help,
        }
    });

    ParsedArgs { 
        op,
        positional,
        verbose,
        old
    }
}

struct VSoundHeader {
    sound_time_ms: i32,
    keyoff_ms: i32,
    envelope: i32,
    sample_length: i32,
    loop_start: i32,
    pitch: i16,
    ambient: bool,
    looping: bool,
    preload: bool,
    incidental: bool,
    unused: i16,
}

impl VSoundHeader {
    fn read<R: Read>(rdr: &mut R, is_old_version: bool) -> Result<Self> {
        let sound_time_ms = rdr.read_i32::<LittleEndian>()?;
        let keyoff_ms = if is_old_version { 0 } else { rdr.read_i32::<LittleEndian>()? };
        let envelope = if is_old_version { 0 } else { rdr.read_i32::<LittleEndian>()? };
        let sample_length = rdr.read_i32::<LittleEndian>()?;
        let loop_start = if is_old_version { 0 } else { rdr.read_i32::<LittleEndian>()? };
        let bitfield = rdr.read_u32::<LittleEndian>()?;
        let pitch = (bitfield & 0xFFFF) as i16;
        let ambient = (bitfield >> 16) & 1 != 0;
        let looping = (bitfield >> 17) & 1 != 0;
        let preload = (bitfield >> 18) & 1 != 0;
        let incidental = (bitfield >> 19) & 1 != 0;
        let unused = (bitfield >> 20) as i16;
        Ok(Self {
            sound_time_ms,
            keyoff_ms,
            envelope,
            sample_length,
            loop_start,
            pitch,
            ambient,
            looping,
            preload,
            incidental,
            unused,
        })
    }
}

struct VMusicHeader {
    block_count: i16,
    pitch: i16,
    looping: bool,
    block_sound_time: i32,
    loop_start_block: i16,
    loop_start_offset: i16,
}

impl VMusicHeader {
    pub const BLOCK_SIZE: i32 = 0x8000;

    fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        let block_count = rdr.read_i16::<LittleEndian>()?;
        let pitch_looping_bitfield = rdr.read_u16::<LittleEndian>()?;
        let pitch = (pitch_looping_bitfield & 0x7FFF) as i16;
        let looping = pitch_looping_bitfield & 0x8000 != 0;
        let block_sound_time = rdr.read_i32::<LittleEndian>()?;
        let loop_start_block = rdr.read_i16::<LittleEndian>()?;
        let loop_start_offset = rdr.read_i16::<LittleEndian>()?;
        Ok(Self {
            block_count,
            pitch,
            looping,
            block_sound_time,
            loop_start_block,
            loop_start_offset,
        })
    }
}

fn print_help() {
    println!("Usage:");
    println!("  vsound file.vse");
    println!("    shows information about VSE/VMU file");
    println!("  vsound input_file.vse output_file.wav");
    println!("    converts VSE/VMU file to WAV file");
}

fn print_version() {
    println!("VSound Tool {} by Rafalh", env!("CARGO_PKG_VERSION"));
}

fn print_vsound_header(hdr: &VSoundHeader) {
    println!("Volition Sound Effect File Header:");
    println!("  sound_time_ms  {}", hdr.sound_time_ms);
    println!("  keyoff_ms      {}", hdr.keyoff_ms);
    println!("  envelope       0x{:X}", hdr.envelope);
    println!("  sample_length  {}", hdr.sample_length);
    println!("  loop_start     {}", hdr.loop_start);
    println!("  pitch          {}", hdr.pitch);
    println!("  ambient        {}", hdr.ambient);
    println!("  looping        {}", hdr.looping);
    println!("  preload        {}", hdr.preload);
    println!("  incidental     {}", hdr.incidental);
    println!("  unused         {}", hdr.unused);
}

fn print_vmusic_header(hdr: &VMusicHeader) {
    println!("Volition Music File Header:");
    println!("  block_count        {}", hdr.block_count);
    println!("  pitch              {}", hdr.pitch);
    println!("  looping            {}", hdr.looping);
    println!("  block_sound_time   {}", hdr.block_sound_time);
    println!("  loop_start_block   {}", hdr.loop_start_block);
    println!("  loop_start_offset  {}", hdr.loop_start_offset);
}

fn print_vsound_info(pathname: &str, is_old_version: bool) -> Result<()> {
    let mut rdr = BufReader::new(File::open(pathname)?);
    let hdr = VSoundHeader::read(&mut rdr, is_old_version)?;
    print_vsound_header(&hdr);
    Ok(())
}

fn print_vmusic_info(pathname: &str) -> Result<()> {
    let mut rdr = BufReader::new(File::open(pathname)?);
    let hdr = VMusicHeader::read(&mut rdr)?;
    print_vmusic_header(&hdr);
    Ok(())
}

fn print_file_info(pathname: &str, is_old_version: bool) -> Result<()> {
    if pathname.ends_with(".vse") {
        print_vsound_info(pathname, is_old_version)
    } else if pathname.ends_with(".vmu") {
        print_vmusic_info(pathname)
    } else {
        eprintln!("Unknown input file extension! Supported extensions: vse, vmu.");
        Ok(())
    }
}

fn print_pcm_info(pcm_wf: &wave::PcmWaveFormat) {
    println!(
        "Converting to WAV (PCM, {}, {} Hz)", 
        if pcm_wf.wf.nChannels == 1 { "mono"} else { "stereo" },
        pcm_wf.wf.nSamplesPerSec
    );
}

fn convert_vsound(input_pathname: &str, output_pathname: &str, is_old_version: bool) -> Result<()> {
    let mut rdr = BufReader::new(File::open(input_pathname)?);
    let mut wrt = BufWriter::new(File::create(output_pathname)?);

    let hdr = VSoundHeader::read(&mut rdr, is_old_version)?;
    print_vsound_header(&hdr);

    let samples_per_sec = match hdr.pitch {
        940 => 11025,
        1365 => 16000,
        1881 => 22050,
        3763 => 44100,
        pitch => (f32::from(pitch) * 11.722) as u32,
    };

    let pcm_wf = wave::PcmWaveFormat::new(1, samples_per_sec, 16);
    print_pcm_info(&pcm_wf);
    wave::write_wave_file(&mut wrt, &pcm_wf, |wrt| {
        let mut bytes_left = hdr.sample_length as usize;
        let mut buffer = [0u8; Ps2AdpcmDecoder::BLOCK_SIZE];
        let mut pcm_buf = [0i16; Ps2AdpcmDecoder::SAMPLES_PER_BLOCK];
        let mut decoder = Ps2AdpcmDecoder::new();

        while bytes_left > 0 {
            rdr.read_exact(&mut buffer)?;
            let num_samples = decoder.decode(&mut pcm_buf, &buffer);
            for sample in &pcm_buf[..num_samples] {
                wrt.write_i16::<LittleEndian>(*sample)?;
            }
            bytes_left -= buffer.len();
        }
        Ok(())
    })
}

fn read_into_buffer<R: Read>(rdr: &mut R, buf: &mut [u8]) -> Result<usize> {
    let mut total_bytes_read = 0;
    while total_bytes_read < buf.len() {
        let bytes_read = rdr.read(&mut buf[total_bytes_read..])?;
        if bytes_read == 0 {
            break;
        }
        total_bytes_read += bytes_read;
    }
    Ok(total_bytes_read)
}

fn convert_vmusic(input_pathname: &str, output_pathname: &str) -> Result<()> {
    let mut rdr = BufReader::new(File::open(input_pathname)?);
    let mut wrt = BufWriter::new(File::create(output_pathname)?);

    let hdr = VMusicHeader::read(&mut rdr)?;
    print_vmusic_header(&hdr);

    let pcm_wf = wave::PcmWaveFormat::new(2, 44100, 16);
    print_pcm_info(&pcm_wf);
    wave::write_wave_file(&mut wrt, &pcm_wf, |wrt| {
        let mut block = [0u8; VMusicHeader::BLOCK_SIZE as usize];
        for _ in 0..hdr.block_count {
            let bytes_read = read_into_buffer(&mut rdr, &mut block)?;
            const MAX_ADPCM_BLOCKS: usize = VMusicHeader::BLOCK_SIZE as usize / Ps2AdpcmDecoder::BLOCK_SIZE;
            let mut decoder = Ps2AdpcmDecoder::new();
            let mut pcm_buf = [0i16; Ps2AdpcmDecoder::SAMPLES_PER_BLOCK * MAX_ADPCM_BLOCKS];
            let num_samples = decoder.decode(&mut pcm_buf, &block[..bytes_read]);
            let num_samples_per_channel = num_samples / 2;
            let pcm_left = &pcm_buf[..num_samples_per_channel];
            let pcm_right = &pcm_buf[num_samples_per_channel..];
            for j in 0..num_samples / 2 {
                wrt.write_i16::<LittleEndian>(pcm_left[j])?;
                wrt.write_i16::<LittleEndian>(pcm_right[j])?;
            }
        }
        Ok(())
    })
}

fn convert_file(input_pathname: &str, output_pathname: &str, is_old_version: bool) -> Result<()> {
    if input_pathname.ends_with(".vse") {
        convert_vsound(input_pathname, output_pathname, is_old_version)
    } else if input_pathname.ends_with(".vmu") {
        convert_vmusic(input_pathname, output_pathname)
    } else {
        eprintln!("Unknown input file extension! Supported extensions: vse, vmu.");
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = parse_args();
    match args.op {
        Operation::Info => print_file_info(&args.positional[0], args.old)?,
        Operation::Convert => convert_file(&args.positional[0], &args.positional[1], args.old)?,
        Operation::Help => print_help(),
        Operation::Version => print_version()
    }
    Ok(())
}
