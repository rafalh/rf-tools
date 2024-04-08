use crate::io_utils::WriteExt;
use byteorder::{LittleEndian, WriteBytesExt};
use std::convert::TryInto;
use std::io::{Result, Write};

pub struct Rfg {
    pub groups: Vec<Group>,
}

impl Rfg {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_all(&[0x0D, 0xD0, 0x3D, 0xD4])?; // signature
        wrt.write_u32::<LittleEndian>(0xC8)?; // version
        wrt.write_i32::<LittleEndian>(self.groups.len().try_into().unwrap())?;
        for group in &self.groups {
            group.write(wrt)?;
        }

        Ok(())
    }
}

pub struct Group {
    pub group_name: String,
    pub brushes: Vec<Brush>,
}

impl Group {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_vstr(&self.group_name)?;
        wrt.write_u8(0)?; // is_moving
        wrt.write_u32::<LittleEndian>(self.brushes.len().try_into().unwrap())?; // brushes
        for brush in &self.brushes {
            brush.write(wrt)?;
        }
        wrt.write_u32::<LittleEndian>(0)?; // geo_regions
        wrt.write_u32::<LittleEndian>(0)?; // lights
        wrt.write_u32::<LittleEndian>(0)?; // cutscene_cameras
        wrt.write_u32::<LittleEndian>(0)?; // cutscene_path_nodes
        wrt.write_u32::<LittleEndian>(0)?; // ambient_sounds
        wrt.write_u32::<LittleEndian>(0)?; // events
        wrt.write_u32::<LittleEndian>(0)?; // mp_respawn_points
        wrt.write_u32::<LittleEndian>(0)?; // nav_points
        wrt.write_u32::<LittleEndian>(0)?; // entities
        wrt.write_u32::<LittleEndian>(0)?; // items
        wrt.write_u32::<LittleEndian>(0)?; // clutters
        wrt.write_u32::<LittleEndian>(0)?; // triggers
        wrt.write_u32::<LittleEndian>(0)?; // particle_emitters
        wrt.write_u32::<LittleEndian>(0)?; // gas_regions
        wrt.write_u32::<LittleEndian>(0)?; // decals
        wrt.write_u32::<LittleEndian>(0)?; // climbing_regions
        wrt.write_u32::<LittleEndian>(0)?; // room_effects
        wrt.write_u32::<LittleEndian>(0)?; // eax_effects
        wrt.write_u32::<LittleEndian>(0)?; // bolt_emitters
        wrt.write_u32::<LittleEndian>(0)?; // targets
        wrt.write_u32::<LittleEndian>(0)?; // push_regions
        Ok(())
    }
}

pub struct Brush {
    pub uid: i32,
    pub pos: [f32; 3],
    pub orient: [f32; 9],
    pub solid: Solid,
}

impl Brush {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_i32::<LittleEndian>(self.uid)?;
        wrt.write_f32_slice::<LittleEndian>(&self.pos)?;
        wrt.write_f32_slice::<LittleEndian>(&reorder_matrix_rows(self.orient))?;
        self.solid.write(wrt)?;
        wrt.write_u32::<LittleEndian>(0)?; // flags
        wrt.write_i32::<LittleEndian>(-1)?; // life
        wrt.write_u32::<LittleEndian>(0)?; // state
        Ok(())
    }
}

fn reorder_matrix_rows(mat: [f32; 9]) -> [f32; 9] {
    // rfl/rfg uses non-standard row order: forward, right, up
    [
        mat[6], mat[7], mat[8], mat[0], mat[1], mat[2], mat[3], mat[4], mat[5],
    ]
}

pub struct Solid {
    pub textures: Vec<String>,
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<Face>,
}

impl Solid {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32::<LittleEndian>(0)?; // unknown1
        wrt.write_u32::<LittleEndian>(0)?; // modifiability
        wrt.write_vstr("")?; // modifiability
        wrt.write_u32::<LittleEndian>(self.textures.len().try_into().unwrap())?; // num_textures
        for texture in &self.textures {
            wrt.write_vstr(texture)?;
        }
        wrt.write_u32::<LittleEndian>(0)?; // num_face_scroll_data
        wrt.write_u32::<LittleEndian>(0)?; // num_rooms
        wrt.write_u32::<LittleEndian>(0)?; // num_subroom_lists
        wrt.write_u32::<LittleEndian>(0)?; // num_portals
        wrt.write_u32::<LittleEndian>(self.vertices.len().try_into().unwrap())?; // num_vertices
        for vertex in &self.vertices {
            wrt.write_f32_slice::<LittleEndian>(vertex)?;
        }
        wrt.write_u32::<LittleEndian>(self.faces.len().try_into().unwrap())?; // num_faces
        for face in &self.faces {
            face.write(wrt)?;
        }
        wrt.write_u32::<LittleEndian>(0)?; // num_surfaces
        Ok(())
    }
}

pub struct FaceVertex {
    pub index: u32,
    pub texture_coords: [f32; 2],
}

impl FaceVertex {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_u32::<LittleEndian>(self.index)?;
        wrt.write_f32_slice::<LittleEndian>(&self.texture_coords)?;
        Ok(())
    }
}

pub struct Face {
    pub plane: [f32; 4],
    pub texture: i32,
    pub vertices: Vec<FaceVertex>,
}

impl Face {
    pub fn write<W: Write>(&self, wrt: &mut W) -> Result<()> {
        wrt.write_f32_slice::<LittleEndian>(&self.plane)?;
        wrt.write_i32::<LittleEndian>(self.texture)?;
        wrt.write_i32::<LittleEndian>(-1)?; // surface_index
        wrt.write_i32::<LittleEndian>(-1)?; // face_id
        wrt.write_i32::<LittleEndian>(-1)?; // reserved1
        wrt.write_i32::<LittleEndian>(-1)?; // reserved1
        wrt.write_i32::<LittleEndian>(0)?; // portal_index_plus_2
        wrt.write_u16::<LittleEndian>(0)?; // flags
        wrt.write_u16::<LittleEndian>(0)?; // reserved2
        wrt.write_u32::<LittleEndian>(0)?; // smoothing_groups
        wrt.write_i32::<LittleEndian>(-1)?; // room_index
        wrt.write_i32::<LittleEndian>(self.vertices.len().try_into().unwrap())?; // num_vertices
        for fvert in &self.vertices {
            fvert.write(wrt)?;
        }
        Ok(())
    }
}
