use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Result, Seek, SeekFrom, Write};

pub struct RiffChunkHeader {
    pub chunk_id: u32,
    pub chunk_size: u32,
}

impl RiffChunkHeader {
    #[allow(dead_code)]
    pub fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        Ok(Self {
            chunk_id: rdr.read_u32::<BigEndian>()?,
            chunk_size: rdr.read_u32::<LittleEndian>()?,
        })
    }

    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32::<BigEndian>(self.chunk_id)?;
        wrt.write_u32::<LittleEndian>(self.chunk_size)?;
        Ok(())
    }
}

// Big Endian Four CC values
pub const RIFF_CHUNK_ID: u32 = 0x52494646;
pub const FMT_CHUNK_ID: u32 = 0x666d7420;
pub const DATA_CHUNK_ID: u32 = 0x64617461;
pub const WAVE_FORMAT: u32 = 0x57415645;
pub const WAVE_FORMAT_PCM: u16 = 1;

#[allow(non_snake_case)]
pub struct WaveFormat {
    pub wFormatTag: u16,
    pub nChannels: u16,
    pub nSamplesPerSec: u32,
    pub nAvgBytesPerSec: u32,
    pub nBlockAlign: u16,
}

impl WaveFormat {
    #[allow(dead_code)]
    pub fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        Ok(Self {
            wFormatTag: rdr.read_u16::<LittleEndian>()?,
            nChannels: rdr.read_u16::<LittleEndian>()?,
            nSamplesPerSec: rdr.read_u32::<LittleEndian>()?,
            nAvgBytesPerSec: rdr.read_u32::<LittleEndian>()?,
            nBlockAlign: rdr.read_u16::<LittleEndian>()?,
        })
    }

    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u16::<LittleEndian>(self.wFormatTag)?;
        wrt.write_u16::<LittleEndian>(self.nChannels)?;
        wrt.write_u32::<LittleEndian>(self.nSamplesPerSec)?;
        wrt.write_u32::<LittleEndian>(self.nAvgBytesPerSec)?;
        wrt.write_u16::<LittleEndian>(self.nBlockAlign)?;
        Ok(())
    }
}

#[allow(non_snake_case)]
pub struct PcmWaveFormat {
    pub wf: WaveFormat,
    pub wBitsPerSample: u16,
}

impl PcmWaveFormat {
    pub fn new(channels: u16, samples_per_sec: u32, bits_per_sample: u16) -> PcmWaveFormat {
        PcmWaveFormat {
            wf: WaveFormat {
                wFormatTag: WAVE_FORMAT_PCM,
                nChannels: channels,
                nSamplesPerSec: samples_per_sec,
                nAvgBytesPerSec: u32::from(bits_per_sample / 8 * channels) * samples_per_sec,
                nBlockAlign: bits_per_sample / 8 * channels,
            },
            wBitsPerSample: bits_per_sample,
        }
    }

    #[allow(dead_code)]
    pub fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        Ok(Self {
            wf: WaveFormat::read(rdr)?,
            wBitsPerSample: rdr.read_u16::<LittleEndian>()?,
        })
    }

    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        self.wf.write(wrt)?;
        wrt.write_u16::<LittleEndian>(self.wBitsPerSample)?;
        Ok(())
    }
}

pub fn write_riff_chunk<W: Write + Seek, F: FnMut(&mut W) -> Result<()>>(
    wrt: &mut W,
    chunk_id: u32,
    mut fun: F,
) -> Result<()> {
    let header_pos = wrt.stream_position()?;
    let mut chunk_hdr = RiffChunkHeader {
        chunk_id,
        chunk_size: 0,
    };
    chunk_hdr.write(wrt)?;
    let pos_before = wrt.stream_position()?;
    fun(wrt)?;
    let pos_after = wrt.stream_position()?;
    // Update chunk size
    chunk_hdr.chunk_size = (pos_after - pos_before).try_into().unwrap();
    wrt.seek(SeekFrom::Start(header_pos))?;
    chunk_hdr.write(wrt)?;
    // Seek the stream to previous position
    wrt.seek(SeekFrom::Start(pos_after))?;
    Ok(())
}

pub fn write_wave_file<W: Write + Seek, F: FnMut(&mut W) -> Result<()>>(
    wrt: &mut W,
    pcm_wf: &PcmWaveFormat,
    mut data_fun: F,
) -> Result<()> {
    write_riff_chunk(wrt, RIFF_CHUNK_ID, |wrt| {
        wrt.write_u32::<BigEndian>(WAVE_FORMAT)?;
        write_riff_chunk(wrt, FMT_CHUNK_ID, |wrt| pcm_wf.write(wrt))?;
        write_riff_chunk(wrt, DATA_CHUNK_ID, |wrt| data_fun(wrt))?;
        Ok(())
    })
}
