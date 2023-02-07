mod data_structure;

use data_structure::{GPU_data, Particle};
use nanorand::{Rng, WyRand};
use std::{borrow::Cow, future::Future};
use wgpu::util::DeviceExt;
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

const NUM_PARTICLES: u32 = 5000;

// number of single-particle calculations (invocations) in each gpu work group
const PARTICLES_PER_GROUP: u32 = 256;

/// Example struct holds references to wgpu resources and frame persistent data
struct Example {
    particle_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    vertices_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    work_group_count: u32,
    frame_num: usize,
}

impl Example {
    /// constructs initial instance of Example struct
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        //import the shaders
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
        });

        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw.wgsl"))),
        });

        let sim_param_data = vec![
            0.005f32, // deltaT
            0.1,      // rule1Distance
            0.025,    // rule2Distance
            0.025,    // rule3Distance
            0.02,     // rule1Scale
            0.05,     // rule2Scale
            0.005,    // rule3Scale
        ];

        let sim_param: GPU_data<f32> = GPU_data::new(sim_param_data);

        let particles = GPU_data::new(
            (0..NUM_PARTICLES)
                .map(|i| {
                    let x = (i % ((NUM_PARTICLES as f32).sqrt() as u32)) as f32
                        / (NUM_PARTICLES as f32).sqrt()
                        * 2.0
                        - 1.0;
                    let y = (i / ((NUM_PARTICLES as f32).sqrt() as u32)) as f32
                        / (NUM_PARTICLES as f32).sqrt()
                        * 2.0
                        - 1.0;
                    Particle::new([x, y], [0.0, 0.0], ((i % 2) as f32) * 2.0 - 1.0)
                })
                .collect::<Vec<_>>(),
        );

        // create compute bind layout group and compute pipeline layout
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    sim_param.bind_group_layout(0, false, false),
                    particles.bind_group_layout(1, true, true),
                    particles.bind_group_layout(2, false, true),
                ],
                label: None,
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        // create render pipeline with empty bind group layout

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: "main_vs",
                buffers: &[
                    Particle::desc(),
                    wgpu::VertexBufferLayout {
                        array_stride: (std::mem::size_of::<f32>() * 2) as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![3 => Float32x2],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: "main_fs",
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // create compute pipeline

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        // buffer for the three 2d triangle vertices of each instance

        let vertex_buffer_data = [-0.01f32, -0.02, 0.01, -0.02, 0.00, 0.02];
        let vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // creates two buffers of particle data each of size NUM_PARTICLES
        // the two buffers alternate as dst and src for each frame

        let particle_buffers = vec![
            particles.buffer(device, "particle_buffer_0"),
            particles.buffer(device, "particle_buffer_1"),
        ];

        let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();

        // create two bind groups, one for each buffer as the src
        // where the alternate buffer is used as the dst

        for i in 0..2 {
            particle_bind_groups.push(
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: sim_param
                                .buffer(device, "sim_param_buffer")
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: particle_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: particle_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                        },
                    ],
                    label: None,
                }),
            );
        }

        // calculates number of work groups from PARTICLES_PER_GROUP constant
        let work_group_count =
            ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        // returns Example struct and No encoder commands

        Example {
            particle_bind_groups,
            particle_buffers,
            vertices_buffer,
            compute_pipeline,
            render_pipeline,
            work_group_count,
            frame_num: 0,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}
    fn resize(
        &mut self,
        _sc_desc: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
    }

    /// render is called each frame, dispatching compute groups proportional
    ///   a TriangleList draw call for all NUM_PARTICLES at 3 vertices each
    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // create render pass descriptor and its color attachments
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                // Not clearing here in order to test wgpu's zero texture initialization on a surface texture.
                // Users should avoid loading uninitialized memory since this can cause additional overhead.
                load: wgpu::LoadOp::Load,
                store: true,
            },
        })];
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
        };

        // get command encoder
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.push_debug_group("compute boid movement");
        {
            // compute pass
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.particle_bind_groups[self.frame_num % 2], &[]);
            cpass.dispatch_workgroups(self.work_group_count, 1, 1);
        }
        command_encoder.pop_debug_group();

        command_encoder.push_debug_group("render boids");
        {
            // render pass
            let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
            rpass.set_pipeline(&self.render_pipeline);
            // render dst particles
            rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
            // the three instance-local vertices
            rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
            rpass.draw(0..3, 0..NUM_PARTICLES);
        }
        command_encoder.pop_debug_group();

        // update frame count
        self.frame_num += 1;

        // done
        queue.submit(Some(command_encoder.finish()));
    }
}

/// run example
fn main() {
    let setup = pollster::block_on(setup());

    start(setup);
}
struct Setup {
    window: winit::window::Window,
    event_loop: EventLoop<()>,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn setup() -> Setup {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title("my title");
    let window = builder.build(&event_loop).unwrap();
    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        dx12_shader_compiler,
    });

    let (size, surface) = unsafe {
        let size = window.inner_size();
        let surface = instance.create_surface(&window).unwrap();
        (size, surface)
    };

    let adapter =
        wgpu::util::initialize_adapter_from_env_or_default(&instance, backends, Some(&surface))
            .await
            .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }
}

fn start(
    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }: Setup,
) {
    let spawner = Spawner::new();

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .expect("Surface isn't supported by the adapter.");
    surface.configure(&device, &config);

    let mut example = Example::init(&config, &adapter, &device, &queue);

    log::info!("Entering render loop...");
    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::RedrawEventsCleared => {
                #[cfg(not(target_arch = "wasm32"))]
                spawner.run_until_stalled();

                window.request_redraw();
            }
            event::Event::WindowEvent {
                event:
                    WindowEvent::Resized(size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut size,
                        ..
                    },
                ..
            } => {
                log::info!("Resizing to {:?}", size);
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                example.resize(&config, &device, &queue);
                surface.configure(&device, &config);
            }
            event::Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                #[cfg(not(target_arch = "wasm32"))]
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::R),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    println!("{:#?}", instance.generate_report());
                }
                _ => {
                    example.update(event);
                }
            },
            event::Event::RedrawRequested(_) => {
                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        surface.configure(&device, &config);
                        surface
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                example.render(&view, &device, &queue);

                frame.present();
            }
            _ => {}
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Spawner<'a> {
    executor: async_executor::LocalExecutor<'a>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> Spawner<'a> {
    fn new() -> Self {
        Self {
            executor: async_executor::LocalExecutor::new(),
        }
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'a) {
        self.executor.spawn(future).detach();
    }

    fn run_until_stalled(&self) {
        while self.executor.try_tick() {}
    }
}
