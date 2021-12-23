use std::io::{Write, Seek, SeekFrom, Result};
use std::convert::TryInto;
use byteorder::{LittleEndian, WriteBytesExt};
use crate::io_utils::WriteExt;

// File signatures
pub const V3M_SIGNATURE: u32 = 0x52463344; // RF3D
pub const V3C_SIGNATURE: u32 = 0x5246434D; // RFCM

// Supported format version
pub const VERSION: u32 = 0x40000;

// File chunk types
pub const END_CHUNK: u32       = 0x00000000; // terminating section
pub const SUBMESH_CHUNK: u32   = 0x5355424D; // 'SUBM'
pub const CSPHERE_CHUNK: u32   = 0x43535048; // 'CSPH'
pub const BONE_CHUNK: u32      = 0x424F4E45; // 'BONE'

#[allow(unused)]
pub const MAX_BONES: usize = 50;

pub struct File {
    pub header: FileHeader,
    pub lod_meshes: Vec<LodMesh>,
    pub cspheres: Vec<ColSphere>,
    pub bones: Vec<Bone>,
}

impl File {
    pub fn write<W: Write + Seek>(&self, wrt: &mut W) -> Result<()> {
        self.header.write(wrt)?;
        for lod_mesh in &self.lod_meshes {
            FileChunk{
                chunk_type: SUBMESH_CHUNK,
                chunk_size: 0, // ccrunch sets it to 0
            }.write(wrt)?;
            lod_mesh.write(wrt)?;
        }
        for csphere in &self.cspheres {
            FileChunk::write_new(wrt, CSPHERE_CHUNK, |wrt| {
                csphere.write(wrt)
            })?;
        }
        if !self.bones.is_empty() {
            FileChunk::write_new(wrt, BONE_CHUNK, |wrt| {
                wrt.write_i32::<LittleEndian>(self.bones.len().try_into().expect("number of bones fits in i32"))?;
                for bone in &self.bones {
                    bone.write(wrt)?
                }
                Ok(())
            })?;
        }
        FileChunk{
            chunk_type: END_CHUNK,
            chunk_size: 0,
        }.write(wrt)
    }
}

#[derive(Default)]
pub struct FileHeader {
    pub signature: u32,
    pub version: u32,
    pub num_lod_meshes: i32,
    pub num_all_vertices: i32, // ccrunch resets value to 0
    pub num_all_faces: i32, // ccrunch resets value to 0
    pub num_all_vertex_normals: i32, // ccrunch resets value to 0
    pub num_all_materials: i32,
    pub num_all_meshes: i32, // ccrunch resets value to 0
    pub num_dumbs: i32, // ccrunch resets value to 0 and discards dumb sections
    pub num_cspheres: i32,
}

impl FileHeader {

    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32::<LittleEndian>(self.signature)?;
        wrt.write_u32::<LittleEndian>(self.version)?;
        wrt.write_i32::<LittleEndian>(self.num_lod_meshes)?;
        wrt.write_i32::<LittleEndian>(self.num_all_vertices)?; 
        wrt.write_i32::<LittleEndian>(self.num_all_faces)?;
        wrt.write_i32::<LittleEndian>(self.num_all_vertex_normals)?;
        wrt.write_i32::<LittleEndian>(self.num_all_materials)?;
        wrt.write_i32::<LittleEndian>(self.num_all_meshes)?;
        wrt.write_i32::<LittleEndian>(self.num_dumbs)?;
        wrt.write_i32::<LittleEndian>(self.num_cspheres)?;
        Ok(())
    }
}

pub struct FileChunk {
    pub chunk_type: u32,
    pub chunk_size: u32,
}

impl FileChunk {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32::<LittleEndian>(self.chunk_type)?;
        wrt.write_u32::<LittleEndian>(self.chunk_size)?;
        Ok(())
    }

    pub fn write_new<W: Write + Seek, F: FnMut(&mut W) -> Result<()>>(wrt: &mut W, chunk_type: u32, mut fun: F) -> Result<()> {
        let header_pos = wrt.stream_position()?;
        let mut chunk_hdr = FileChunk {
            chunk_type,
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
}

pub struct LodMesh {
    pub name: String,
    pub parent_name: String,
    pub version: i32,
    pub distances: Vec<f32>,
    pub offset: [f32; 3],
    pub radius: f32,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl LodMesh {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        assert!(self.meshes.len() == self.distances.len());

        wrt.write_char_array(&self.name, 24)?;
        wrt.write_char_array(&self.parent_name, 24)?;
        wrt.write_i32::<LittleEndian>(self.version)?;
        wrt.write_i32::<LittleEndian>(self.meshes.len() as i32)?;
        for dist in &self.distances {
            wrt.write_f32::<LittleEndian>(*dist)?;
        }

        wrt.write_f32_slice_le(&self.offset)?;
        wrt.write_f32::<LittleEndian>(self.radius)?;
        wrt.write_f32_slice_le(&self.bbox_min)?;
        wrt.write_f32_slice_le(&self.bbox_max)?;

        for lod_mesh in &self.meshes {
            lod_mesh.write(wrt)?;
        }

        wrt.write_i32::<LittleEndian>(self.materials.len() as i32)?;
        for material in &self.materials {
            material.write(wrt)?;
        }

        wrt.write_u32::<LittleEndian>(1)?; // num_unknown1
        wrt.write_char_array(&self.name, 24)?; // unknown1[0].unknown0
        wrt.write_f32::<LittleEndian>(0.0)?; // unknown1[0].unknown1

        Ok(())
    }
}

pub struct Mesh {
    pub flags: u32,
    pub num_vecs: i32,
    pub chunks: Vec<MeshChunk>,
    pub data_block: Vec<u8>,
    pub num_prop_points: i32,
    pub textures: Vec<MeshTextureRef>
}

impl Mesh {
    pub const MAX_TEXTURES: usize = 7;

    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32::<LittleEndian>(self.flags)?;
        wrt.write_i32::<LittleEndian>(self.num_vecs)?;
        wrt.write_u16::<LittleEndian>(self.chunks.len().try_into().unwrap())?;

        wrt.write_i32::<LittleEndian>(self.data_block.len() as i32)?;
        wrt.write_all(&self.data_block)?;

        wrt.write_i32::<LittleEndian>(-1)?; // unknown1
        for chunk in &self.chunks {
            chunk.write(wrt)?;
        }

        wrt.write_i32::<LittleEndian>(self.num_prop_points)?;

        assert!(self.textures.len() <= Self::MAX_TEXTURES);
        wrt.write_i32::<LittleEndian>(self.textures.len() as i32)?;
        for texture in &self.textures {
            texture.write(wrt)?;
        }

        Ok(())
    }
}

pub struct MeshChunk {
    pub num_vecs: u16,
    pub num_faces: u16,
    pub vecs_alloc: u16,
    pub faces_alloc: u16,
    pub same_pos_vertex_offsets_alloc: u16,
    pub wi_alloc: u16,
    pub uvs_alloc: u16,
    pub render_mode: u32,
}

impl MeshChunk {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u16::<LittleEndian>(self.num_vecs)?;
        wrt.write_u16::<LittleEndian>(self.num_faces)?;
        wrt.write_u16::<LittleEndian>(self.vecs_alloc)?;
        wrt.write_u16::<LittleEndian>(self.faces_alloc)?;
        wrt.write_u16::<LittleEndian>(self.same_pos_vertex_offsets_alloc)?;
        wrt.write_u16::<LittleEndian>(self.wi_alloc)?;
        wrt.write_u16::<LittleEndian>(self.uvs_alloc)?;
        wrt.write_u32::<LittleEndian>(self.render_mode)?;
        Ok(())
    }
}

pub struct MeshTextureRef {
    pub material_index: u8,
    pub tex_name: String,
}

impl MeshTextureRef {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u8(self.material_index)?;
        wrt.write_all(self.tex_name.as_bytes())?;
        wrt.write_u8(0)?;
        Ok(())
    }
}

pub struct MeshDataBlock {
    pub chunks: Vec<MeshDataBlockChunkInfo>,
    pub chunks_data: Vec<MeshChunkData>,
    pub prop_points: Vec<PropPoint>,
}

impl MeshDataBlock {
    pub const VERSION: i32 = 7;
    
    pub fn write<W: Write + Seek>(&self, wrt: &mut W) -> Result<()> {
        for chunk in &self.chunks {
            chunk.write(wrt)?;
        }
        // padding to 0x10 (to data section begin)
        write_v3mc_data_block_padding(wrt)?;
        for chunk_data in &self.chunks_data {
            chunk_data.write(wrt)?;
        }
        // padding to 0x10 (to data section begin)
        write_v3mc_data_block_padding(wrt)?;
        for prop in &self.prop_points {
            prop.write(wrt)?;
        }
        Ok(())
    }
}

fn write_v3mc_data_block_padding<W: Write + Seek>(wrt: &mut W) -> std::io::Result<()> {
    let mut pos = wrt.seek(SeekFrom::Current(0))?;
    while pos & 0xF != 0 {
        wrt.write_u8(0)?;
        pos += 1;
    }
    Ok(())
}

pub struct MeshDataBlockChunkInfo {
    pub texture_index: i32,
}

impl MeshDataBlockChunkInfo {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        // unused data before texture index (game overrides it with data from MeshChunk)
        let unused_0 = [0u8; 0x20];
        wrt.write_all(&unused_0)?;
        // write texture index in LOD model textures array
        wrt.write_i32::<LittleEndian>(self.texture_index)?;
        // unused data after texture index (game overrides it with data from MeshChunk)
        let unused_24 = [0u8; 0x38 - 0x24];
        wrt.write_all(&unused_24)?;
        Ok(())
    }
}

pub struct MeshChunkData {
    pub vecs: Vec<[f32; 3]>,
    pub norms: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub faces: Vec<MeshFace>,
    pub face_planes: Vec<[f32; 4]>,
    pub same_pos_vertex_offsets: Vec<i16>,
    pub wi: Vec<WeightIndexArray>,
}

impl MeshChunkData {
    pub fn write<W: Write + Seek>(&self, wrt: &mut W) -> Result<()> {
        for pos in &self.vecs {
            wrt.write_f32_slice_le(pos)?;
        }
        write_v3mc_data_block_padding(wrt)?;
    
        for norm in &self.norms {
            wrt.write_f32_slice_le(norm)?;
        }
        write_v3mc_data_block_padding(wrt)?;
    
        for uv in &self.uvs {
            wrt.write_f32_slice_le(uv)?;
        }
        write_v3mc_data_block_padding(wrt)?;
    
        for face in &self.faces {
            face.write(wrt)?;
        }
        write_v3mc_data_block_padding(wrt)?;

        // write triangle planes (used for backface culling)
        for p in &self.face_planes {
            wrt.write_f32_slice_le(p)?;
        }
        write_v3mc_data_block_padding(wrt)?;
    
        // same_pos_vertex_offsets
        for off in &self.same_pos_vertex_offsets {
            wrt.write_i16::<LittleEndian>(*off)?;
        }
        write_v3mc_data_block_padding(wrt)?;
    
        for wi in &self.wi {
            wi.write(wrt)?;
        }
        write_v3mc_data_block_padding(wrt)?;
    
        // if (Mesh::flags & 0x1) { // morph_vertices_map
        //     orig_vert_map: [u16; Mesh::num_vertices];
        //     // padding to 0x10 (to data section begin)
        // }
        Ok(())
    }
}

pub struct MeshFace {
    pub vindices: [u16; 3],
    pub flags: u16,
}

impl MeshFace {
    pub const DOUBLE_SIDED: u16 = 0x20;

    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        for i in &self.vindices {
            wrt.write_u16::<LittleEndian>(*i)?;
        }
        wrt.write_u16::<LittleEndian>(self.flags)?;
        Ok(())
    }
}

#[derive(Default)]
pub struct WeightIndexArray {
    pub weights: [u8; 4],
    pub indices: [u8; 4],
}

impl WeightIndexArray {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        for w in &self.weights {
            wrt.write_u8(*w)?;
        }
        for i in &self.indices {
            wrt.write_u8(*i)?;
        }
        Ok(())
    }
}

pub struct PropPoint {
    pub name: String,
    pub orient: [f32; 4],
    pub pos: [f32; 3],
    pub parent_index: i32,
}

impl PropPoint {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_char_array(&self.name, 0x44)?;
        wrt.write_f32_slice_le(&self.orient)?;
        wrt.write_f32_slice_le(&self.pos)?;
        wrt.write_i32::<LittleEndian>(self.parent_index)?;
        Ok(())
    }
}

pub struct ColSphere {
    pub name: String,
    pub parent_index: i32,
    pub pos: [f32; 3],
    pub radius: f32,
}

impl ColSphere {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_char_array(&self.name, 24)?;
        wrt.write_i32::<LittleEndian>(self.parent_index)?;
        wrt.write_f32_slice_le(&self.pos)?;
        wrt.write_f32::<LittleEndian>(self.radius)?;
        Ok(())
    }
}

pub struct Bone {
    pub name: String,
    pub rot: [f32; 4],
    pub pos: [f32; 3],
    pub parent: i32,
}

impl Bone {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_char_array(&self.name, 24)?;
        wrt.write_f32_slice_le(&self.rot)?;
        wrt.write_f32_slice_le(&self.pos)?;
        wrt.write_i32::<LittleEndian>(self.parent)?;
        Ok(())
    }
}

#[derive(Default, PartialEq)]
pub struct Material {
    pub tex_name: String,  // not used by RF PC
    pub self_illumination: f32,  // used by static lighting code that is not working in RF PC (it does work in DF)
    pub specular_level: f32,  // not used by RF PC
    pub glossiness: f32,  // not used by RF PC
    pub reflection_amount: f32,  // not used by RF PC
    pub refl_tex_name: String,  // not used by RF PC
    pub flags: u32,  // not used by RF PC
}

impl Material {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_char_array(&self.tex_name, 32)?;
        wrt.write_f32::<LittleEndian>(self.self_illumination)?;
        wrt.write_f32::<LittleEndian>(self.specular_level)?;
        wrt.write_f32::<LittleEndian>(self.glossiness)?;
        wrt.write_f32::<LittleEndian>(self.reflection_amount)?;
        wrt.write_char_array(&self.refl_tex_name, 32)?;
        wrt.write_u32::<LittleEndian>(self.flags)?;
        Ok(())
    }
}

#[allow(dead_code)]
pub enum TextureSource {
    None = 0,
    Wrap = 1,
    Clamp = 2,
    ClampNoFiltering = 3,
    // Other values are used with multi-texturing so they do not make sense here because
    // V3M and V3C support only a single diffuse texture
}

#[allow(dead_code)]
pub enum ColorOp {
    SelectArg0IgnoreCurrentColor = 0x0,
    SelectArg0 = 0x1,
    Mul = 0x2,
    Add = 0x3,
    Mul2x = 0x4,
}

#[allow(dead_code)]
pub enum AlphaOp {
    SelArg2 = 0x0,
    SelArg1 = 0x1,
    SelArg1IgnoreCurrentColor = 0x2,
    Mul = 0x3,
}

#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
pub enum AlphaBlend {
    None = 0x0,
    AlphaAdditive = 0x1,
    SrcAlpha2 = 0x2,
    AlphaBlendAlpha = 0x3,
    SrcAlpha4 = 0x4,
    DestColor = 0x5,
    InvDestColor = 0x6,
    SwappedSrcDestColor = 0x7,
}

#[allow(dead_code)]
pub enum ZbufferType {
    None = 0x0,
    Read = 0x1,
    ReadEqFunc = 0x2,
    Write = 0x3,
    Full = 0x4,
    FullAlphaTest = 0x5,
}

#[allow(dead_code)]
pub enum FogType {
    Type0 = 0x0,
    Type1 = 0x1,
    Type2 = 0x2,
    ForceOff = 0x3,
}

pub fn encode_render_mode(
    tex_src: TextureSource, 
    color_op: ColorOp, 
    alpha_op: AlphaOp, 
    alpha_blend: AlphaBlend, 
    zbuffer_type: ZbufferType, 
    fog: FogType
) -> u32 {
    (tex_src as u32)
        | ((color_op as u32) << 5)
        | ((alpha_op as u32) << 10)
        | ((alpha_blend as u32) << 15)
        | ((zbuffer_type as u32) << 20)
        | ((fog as u32) << 25)
}
