use std::io::{Write, Seek, SeekFrom, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use crate::io_utils::WriteExt;

pub const RFA_SIGNATURE: u32 = 0x46564D56; // 'VMVF'
pub const RFA_VERSION: i32 = 8; // 'VMVF'

pub struct File {
    pub header: FileHeader,
    pub bones: Vec<Bone>,
    // TODO: morphing
}

impl File {
    pub fn write<W: Write + Seek>(&self, wrt: &mut W) -> Result<()> {
        self.header.write(wrt)?;
        let mut offsets = FileOffsets {
            morph_vert_mappings_offset: 0,
            morph_vert_data_offset: 0,
            bone_offsets: vec![0; self.bones.len()],
        };
        let offsets_struct_offset = wrt.seek(SeekFrom::Current(0))?;
        offsets.write(wrt)?;
        
        for (i, b) in self.bones.iter().enumerate() {
            let bone_offset = wrt.seek(SeekFrom::Current(0))?;
            offsets.bone_offsets[i] = bone_offset as i32;
            b.write(wrt)?;
        }
        offsets.morph_vert_mappings_offset = wrt.seek(SeekFrom::Current(0))? as i32;
        offsets.morph_vert_data_offset = wrt.seek(SeekFrom::Current(0))? as i32;

        wrt.seek(SeekFrom::Start(offsets_struct_offset))?;
        offsets.write(wrt)?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct FileHeader {
    pub magic: u32,
    pub version: i32,
    pub pos_reduction: f32,
    pub rot_reduction: f32,
    pub start_time: i32,
    pub end_time: i32,
    pub num_bones: i32,
    pub num_morph_vertices: i32,
    pub num_morph_keyframes: i32,
    pub ramp_in_time: i32,
    pub ramp_out_time: i32,
    pub total_rotation: [f32; 4],
    pub total_translation: [f32; 3],
}

impl FileHeader {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32::<LittleEndian>(RFA_SIGNATURE)?;
        wrt.write_i32::<LittleEndian>(RFA_VERSION)?;
        wrt.write_f32::<LittleEndian>(self.pos_reduction)?;
        wrt.write_f32::<LittleEndian>(self.rot_reduction)?; 
        wrt.write_i32::<LittleEndian>(self.start_time)?;
        wrt.write_i32::<LittleEndian>(self.end_time)?;
        wrt.write_i32::<LittleEndian>(self.num_bones)?;
        wrt.write_i32::<LittleEndian>(self.num_morph_vertices)?;
        wrt.write_i32::<LittleEndian>(self.num_morph_keyframes)?;
        wrt.write_i32::<LittleEndian>(self.ramp_in_time)?;
        wrt.write_i32::<LittleEndian>(self.ramp_out_time)?;
        wrt.write_f32_slice::<LittleEndian>(&self.total_rotation)?;
        wrt.write_f32_slice::<LittleEndian>(&self.total_translation)?;
        Ok(())
    }
}

pub struct FileOffsets {
    pub morph_vert_mappings_offset: i32,
    pub morph_vert_data_offset: i32,
    pub bone_offsets: Vec<i32>,
}

impl FileOffsets {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_i32::<LittleEndian>(self.morph_vert_mappings_offset)?;
        wrt.write_i32::<LittleEndian>(self.morph_vert_data_offset)?;
        for o in &self.bone_offsets {
            wrt.write_i32::<LittleEndian>(*o)?;
        }
        Ok(())
    }
}

pub struct Bone {
    pub weight: f32,
    pub rotation_keys: Vec<RotationKey>,
    pub translation_keys: Vec<TranslationKey>,
}

impl Bone {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_f32::<LittleEndian>(self.weight)?;
        wrt.write_i16::<LittleEndian>(self.rotation_keys.len() as i16)?;
        wrt.write_i16::<LittleEndian>(self.translation_keys.len() as i16)?;
        for k in &self.rotation_keys {
            k.write(wrt)?;
        }
        for k in &self.translation_keys {
            k.write(wrt)?;
        }
        Ok(())
    }
}

pub struct RotationKey {
    pub time: i32,
    pub rotation: [i16; 4],
    pub ease_in: i8,
    pub ease_out: i8,
}

impl RotationKey {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_i32::<LittleEndian>(self.time)?;
        wrt.write_i16_slice::<LittleEndian>(&self.rotation)?;
        wrt.write_i8(self.ease_in)?;
        wrt.write_i8(self.ease_out)?;
        wrt.write_i16::<LittleEndian>(0)?; // pad
        Ok(())
    }
}

pub struct TranslationKey {
    pub time: i32,
    pub translation: [f32; 3],
    pub in_tangent: [f32; 3],
    pub out_tangent: [f32; 3],
}

impl TranslationKey {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_i32::<LittleEndian>(self.time)?;
        wrt.write_f32_slice::<LittleEndian>(&self.translation)?;
        wrt.write_f32_slice::<LittleEndian>(&self.in_tangent)?;
        wrt.write_f32_slice::<LittleEndian>(&self.out_tangent)?;
        Ok(())
    }
}
