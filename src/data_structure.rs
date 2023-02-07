use wgpu::util::DeviceExt;

const F32_MEM_SIZE: wgpu::BufferAddress = std::mem::size_of::<f32>() as wgpu::BufferAddress;

//derive from bytemuck
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Particle {
    pub(crate) pos: [f32; 2],
    pub(crate) vel: [f32; 2],
    pub(crate) charge: f32,
    padding: f32,
}

impl Particle {
    pub(crate) fn new(pos: [f32; 2], vel: [f32; 2], charge: f32) -> Self {
        Self {
            pos,
            vel,
            charge,
            padding: 0.0,
        }
    }

    pub(crate) fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 2 * F32_MEM_SIZE,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub(crate) struct GPU_data<T> {
    pub(crate) data: Vec<T>,
}

impl<T: bytemuck::Pod> GPU_data<T> {
    pub(crate) fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub(crate) fn mem_size(&self) -> wgpu::BufferAddress {
        (self.data.len() * std::mem::size_of::<T>()) as wgpu::BufferAddress
    }

    pub(crate) fn bind_group_layout(
        &self,
        binding: u32,
        read_only: bool,
        is_data: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match is_data {
                    true => wgpu::BufferBindingType::Storage { read_only },
                    false => wgpu::BufferBindingType::Uniform,
                },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(self.mem_size()),
            },
            count: None,
        }
    }

    pub(crate) fn buffer(&self, device: &wgpu::Device, name: &str) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(&self.data),
            // usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, //for constant params
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        })
    }
}
