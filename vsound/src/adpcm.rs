// Algorithm based PCSX2 emulator code:
// https://github.com/PCSX2/pcsx2/blob/ef6b65afb40ebd855b0cc113ea04b5afe2bf9607/pcsx2/SPU2/Mixer.cpp

const TBL_XA_FACTOR: [[i32; 2]; 5] = [
    [0, 0],
    [60, 0],
    [115, -52],
    [98, -55],
    [122, -60],
];

fn clamp(val: i32, min: i32, max: i32) -> i32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

pub struct Ps2AdpcmDecoder {
    prev1: i32,
    prev2: i32,
}

impl Ps2AdpcmDecoder {
    pub const BLOCK_SIZE: usize = 16;
    pub const SAMPLES_PER_BLOCK: usize = 2 * 14;

    pub fn new() -> Self {
        Self {
            prev1: 0,
            prev2: 0
        }
    }

    pub fn decode(&mut self, pcm_buf: &mut [i16], adpcm_data: &[u8]) -> usize {
        if adpcm_data.len() % Self::BLOCK_SIZE != 0 {
            panic!("Invalid ADPCM data size");
        }
        let num_blocks = adpcm_data.len() / Self::BLOCK_SIZE;
        let max_samples = num_blocks * Self::SAMPLES_PER_BLOCK;
        if pcm_buf.len() < max_samples {
            panic!("PCM buffer too small");
        }
        let mut num_samples = 0;
        for i in 0..num_blocks {
            let pcm = &mut pcm_buf[i * Self::SAMPLES_PER_BLOCK..(i + 1) * Self::SAMPLES_PER_BLOCK];
            let adpcm_block = &adpcm_data[i * Self::BLOCK_SIZE..(i + 1) * Self::BLOCK_SIZE];
            if adpcm_block[1] != 7 && adpcm_block[0..2] != [0xC, 0]  {
                self.decode_block(pcm, adpcm_block);
                num_samples += Self::SAMPLES_PER_BLOCK;
            }
        }
        num_samples
    }

    fn decode_block(&mut self, buffer: &mut [i16], block: &[u8]) {
        let header: i32 = block[0].into();
        let shift: i32 = (header & 0xF) + 16;
        let id: i32 = header >> 4 & 0xF;
        if id > 4 {
            panic!("Unknown ADPCM coefficients table id {}", id);
        }
        let mut prev1 = self.prev1;
        let mut prev2 = self.prev2;
        let pred1: i32 = TBL_XA_FACTOR[id as usize][0];
        let pred2: i32 = TBL_XA_FACTOR[id as usize][1];
    
        let blockbytes: &[u8] = &block[2..];
    
        for i in 0..14 {
            let byte: i32 = blockbytes[i].into();
            let mut data: i32 = (((byte as u32) << 28) & 0xF000_0000u32) as i32;
            let mut pcm: i32 = (data >> shift) + (((pred1 * prev1) + (pred2 * prev2) + 32) >> 6);
    
            pcm = clamp(pcm, -0x8000, 0x7fff);
            buffer[i * 2] = pcm as i16;
    
            data = (((byte as u32) << 24) & 0xF000_0000u32) as i32;
            let mut pcm2: i32 = (data >> shift) + (((pred1 * pcm) + (pred2 * prev1) + 32) >> 6);
    
            pcm2 = clamp(pcm2, -0x8000, 0x7fff);
            buffer[i * 2 + 1] = pcm2 as i16;
    
            prev2 = pcm;
            prev1 = pcm2;
        }
    
        self.prev1 = prev1;
        self.prev2 = prev2;
    }
}
