//! wgpu GPU compositor (feature `gpu`).
//!
//! Fullscreen-triangle pipeline: foreground and mask textures are uploaded,
//! blur mode runs two separable blur passes on the foreground, and the
//! composite shader blends `mask * fg + (1 - mask) * bg` where `bg` is the
//! blurred foreground, a 1x1 solid color, or the user image (sampler scales
//! it to the frame). Passthrough avoids the GPU entirely.
//!
//! Frame readback is synchronous (`map_async` + `device.poll` loop) because
//! the [`Compositor`](frameiru_core::Compositor) trait is sync; this adds one
//! frame of latency on the readback, acceptable for the fallback role of
//! this path.

use std::sync::Arc;

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::format::PixelFormat;
use frameiru_core::traits::Compositor;
use frameiru_core::{BackgroundMode, FrameBuffer, OverlayMode, Resolution};

use crate::mode::Background;
use crate::shaders::{BLOOM_BLUR_WGSL, BLOOM_WGSL, BLUR_WGSL, COMPOSITE_WGSL};

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// GPU compositor; created via [`GpuCompositor::try_new`].
pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    composite: wgpu::RenderPipeline,
    blur_h: wgpu::RenderPipeline,
    blur_v: wgpu::RenderPipeline,
    blur_uniform: wgpu::Buffer,
    background: Background,
    /// Full-frame overlay applied after the composite shader pass.
    overlay: OverlayMode,
    /// Subject fill light in [0, 1]; 0 disables (see
    /// [`Compositor::set_subject_light`]).
    subject_light: f32,

    // Cyberpunk halation chain: bright-pass 4x4 downsample + two separable
    // box blurs, all at 1/4 resolution.
    bloom_down: wgpu::RenderPipeline,
    bloom_blur_h: wgpu::RenderPipeline,
    bloom_blur_v: wgpu::RenderPipeline,
    bloom_blur_uniform: wgpu::Buffer,

    // Cyberpunk 1/4-res bloom buffers and their current resolution.
    bloom_tex: wgpu::Texture,
    bloom_tmp: wgpu::Texture,
    quarter_res: Resolution,

    // Frame-sized resources, recreated when the resolution changes.
    frame_res: Resolution,
    fg_tex: wgpu::Texture,
    mask_tex: wgpu::Texture,
    bg_tex: wgpu::Texture,
    blur_tex: wgpu::Texture,
    output_tex: wgpu::Texture,
    readback: wgpu::Buffer,
    rgba: Vec<u8>,
    mask_u8: Vec<u8>,
    image_rgba: Vec<u8>,
}

impl GpuCompositor {
    /// Initializes wgpu; fails cleanly when no usable adapter exists.
    pub fn try_new() -> Result<Self, FrameiruError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| FrameiruError::Composition(format!("no wgpu adapter: {e}")))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("frameiru"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| FrameiruError::Composition(format!("device request failed: {e}")))?;

        let shader_composite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let shader_blur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur"),
            source: wgpu::ShaderSource::Wgsl(BLUR_WGSL.into()),
        });
        let shader_bloom = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_WGSL.into()),
        });
        let shader_bloom_blur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom blur"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_BLUR_WGSL.into()),
        });

        let composite = Self::composite_pipeline(&device, &shader_composite);
        let (blur_h, blur_v, blur_uniform) = Self::blur_pipelines(&device, &shader_blur);
        let bloom_down = Self::bloom_pipeline(&device, &shader_bloom);
        let (bloom_blur_h, bloom_blur_v, bloom_blur_uniform) =
            Self::bloom_blur_pipelines(&device, &shader_bloom_blur);

        // Placeholder 1x1 resources; recreated on the first composite.
        let res = Resolution::new(1, 1).expect("1x1 is valid");
        let fg_tex = Self::create_texture(&device, res, TEXTURE_FORMAT, "fg");
        let mask_tex = Self::create_texture(&device, res, MASK_FORMAT, "mask");
        let bg_tex = Self::create_texture(&device, res, TEXTURE_FORMAT, "bg");
        let blur_tex = Self::create_texture(&device, res, TEXTURE_FORMAT, "blur");
        let output_tex = Self::create_texture(&device, res, TEXTURE_FORMAT, "output");
        let bloom_tex = Self::create_texture(&device, res, TEXTURE_FORMAT, "bloom");
        let bloom_tmp = Self::create_texture(&device, res, TEXTURE_FORMAT, "bloom tmp");
        let readback = Self::create_readback(&device, res);

        Ok(Self {
            device,
            queue,
            composite,
            blur_h,
            blur_v,
            blur_uniform,
            bloom_down,
            bloom_blur_h,
            bloom_blur_v,
            bloom_blur_uniform,
            background: Background::Passthrough,
            overlay: OverlayMode::None,
            subject_light: 0.0,
            frame_res: res,
            quarter_res: res,
            bloom_tex,
            bloom_tmp,
            fg_tex,
            mask_tex,
            bg_tex,
            blur_tex,
            output_tex,
            readback,
            rgba: Vec::new(),
            mask_u8: Vec::new(),
            image_rgba: Vec::new(),
        })
    }

    /// Shared scene-pipeline bind group: fg, sampler, mask, bg, sampler,
    /// globals uniform, and the cyberpunk bloom texture (unused unless the
    /// `ov == 9` branch samples it).
    fn composite_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bgl"),
            entries: &[
                tex_binding(0, TEXTURE_FORMAT),
                sampler_binding(1),
                tex_binding(2, MASK_FORMAT),
                tex_binding(3, TEXTURE_FORMAT),
                sampler_binding(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
                tex_binding(6, TEXTURE_FORMAT),
            ],
        })
    }

    fn composite_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
    ) -> wgpu::RenderPipeline {
        Self::scene_pipeline(device, shader, "composite")
    }

    /// Cyberpunk bright pass: same scene inputs as the composite (fg, mask,
    /// bg, globals) but no bloom texture binding — it renders the halo.
    fn bloom_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> wgpu::RenderPipeline {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom bgl"),
            entries: &[
                tex_binding(0, TEXTURE_FORMAT),
                sampler_binding(1),
                tex_binding(2, MASK_FORMAT),
                tex_binding(3, TEXTURE_FORMAT),
                sampler_binding(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom down"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TEXTURE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    }

    fn scene_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let bgl = Self::composite_bgl(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TEXTURE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    }

    fn blur_pipelines(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
    ) -> (wgpu::RenderPipeline, wgpu::RenderPipeline, wgpu::Buffer) {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur bgl"),
            entries: &[
                tex_binding(0, TEXTURE_FORMAT),
                sampler_binding(1),
                tex_binding(2, MASK_FORMAT),
                sampler_binding(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make = |label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: TEXTURE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        (make("blur h"), make("blur v"), uniform)
    }

    /// Plain box-blur pipelines for the 1/4-res bloom buffer (no mask):
    /// 3-entry bind group — texture, sampler, 16-byte params uniform.
    fn bloom_blur_pipelines(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
    ) -> (wgpu::RenderPipeline, wgpu::RenderPipeline, wgpu::Buffer) {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom blur bgl"),
            entries: &[
                tex_binding(0, TEXTURE_FORMAT),
                sampler_binding(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom blur layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom blur uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make = |label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: TEXTURE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        (make("bloom blur h"), make("bloom blur v"), uniform)
    }

    fn create_texture(
        device: &wgpu::Device,
        res: Resolution,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: res.width,
                height: res.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn create_readback(device: &wgpu::Device, res: Resolution) -> wgpu::Buffer {
        let bytes_per_row = (res.width as usize * 4).next_multiple_of(256);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * res.height as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Ensures frame-sized resources exist for `res`.
    fn ensure_resources(&mut self, res: Resolution) {
        if res == self.frame_res {
            return;
        }
        self.frame_res = res;
        self.fg_tex = Self::create_texture(&self.device, res, TEXTURE_FORMAT, "fg");
        self.mask_tex = Self::create_texture(&self.device, res, MASK_FORMAT, "mask");
        self.bg_tex = Self::create_texture(&self.device, res, TEXTURE_FORMAT, "bg");
        self.blur_tex = Self::create_texture(&self.device, res, TEXTURE_FORMAT, "blur");
        self.output_tex = Self::create_texture(&self.device, res, TEXTURE_FORMAT, "output");
        self.readback = Self::create_readback(&self.device, res);
        self.rgba.clear();
        self.rgba
            .resize(res.width as usize * res.height as usize * 4, 0);

        // Cyberpunk halation buffers track their own 1/4 resolution, so the
        // bloom passes stay cheap when the camera stream shrinks.
        let quarter = Resolution {
            width: (res.width / 4).max(1),
            height: (res.height / 4).max(1),
        };
        if quarter != self.quarter_res {
            self.quarter_res = quarter;
            self.bloom_tex = Self::create_texture(&self.device, quarter, TEXTURE_FORMAT, "bloom");
            self.bloom_tmp =
                Self::create_texture(&self.device, quarter, TEXTURE_FORMAT, "bloom tmp");
        }
    }

    fn upload_rgba(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        res: Resolution,
        bytes_per_pixel: u32,
        data: &[u8],
    ) {
        let bytes_per_row = res.width * bytes_per_pixel;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(res.height),
            },
            wgpu::Extent3d {
                width: res.width,
                height: res.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Runs the cyberpunk halation chain: bright-pass 4x4 downsample into
    /// `bloom_tex`, then a separable box blur (h, then v) at 1/4 res. Three
    /// small submits, mirroring the background-blur pass structure.
    fn run_bloom(&mut self, sampler: &wgpu::Sampler, globals: &wgpu::Buffer) {
        let quarter = self.quarter_res;
        // 1) Bright pass: graded over-threshold scene -> 1/4-res bloom_tex.
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bloom down"),
                });
            let bgl = self.bloom_down.get_bind_group_layout(0);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom down bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.fg_tex.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.mask_tex.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bg_tex.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(globals.as_entire_buffer_binding()),
                    },
                ],
            });
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom down pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_tex.create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rp.set_pipeline(&self.bloom_down);
                rp.set_bind_group(0, &bg, &[]);
                rp.draw(0..4, 0..1);
            }
            self.queue.submit([encoder.finish()]);
        }
        // 2) Separable box blur at 1/4 res; each axis gets its own encoder so
        // the uniform write lands before the next pass reads it.
        let step_h = 1.0 / quarter.width as f32;
        self.queue.write_buffer(
            &self.bloom_blur_uniform,
            0,
            bytemuck::cast_slice(&[12.0, step_h, 0.0, 0.0]),
        );
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bloom blur h"),
                });
            let bgl = self.bloom_blur_h.get_bind_group_layout(0);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom blur h bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_tex.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(
                            self.bloom_blur_uniform.as_entire_buffer_binding(),
                        ),
                    },
                ],
            });
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom blur h pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_tmp.create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rp.set_pipeline(&self.bloom_blur_h);
                rp.set_bind_group(0, &bg, &[]);
                rp.draw(0..4, 0..1);
            }
            self.queue.submit([encoder.finish()]);
        }
        let step_v = 1.0 / quarter.height as f32;
        self.queue.write_buffer(
            &self.bloom_blur_uniform,
            0,
            bytemuck::cast_slice(&[12.0, 0.0, step_v, 0.0]),
        );
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bloom blur v"),
                });
            let bgl = self.bloom_blur_v.get_bind_group_layout(0);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom blur v bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.bloom_tmp.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(
                            self.bloom_blur_uniform.as_entire_buffer_binding(),
                        ),
                    },
                ],
            });
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom blur v pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_tex.create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rp.set_pipeline(&self.bloom_blur_v);
                rp.set_bind_group(0, &bg, &[]);
                rp.draw(0..4, 0..1);
            }
            self.queue.submit([encoder.finish()]);
        }
    }

    fn readback_rgb(&mut self, res: Resolution, out: &mut [u8]) -> Result<(), FrameiruError> {
        let bytes_per_row = (res.width as usize * 4).next_multiple_of(256);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row as u32),
                    rows_per_image: Some(res.height),
                },
            },
            wgpu::Extent3d {
                width: res.width,
                height: res.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.readback.slice(..);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done2 = Arc::clone(&done);
        slice.map_async(wgpu::MapMode::Read, move |_| {
            done2.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        // Synchronous readback: spin on poll until the map completes.
        while !done.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = self.device.poll(wgpu::PollType::Wait);
        }

        let (w, h) = (res.width as usize, res.height as usize);
        {
            let data = slice.get_mapped_range();
            for y in 0..h {
                let row = &data[y * bytes_per_row..y * bytes_per_row + w * 4];
                for x in 0..w {
                    let (i, o) = (x * 4, (y * w + x) * 3);
                    out[o] = row[i];
                    out[o + 1] = row[i + 1];
                    out[o + 2] = row[i + 2];
                }
            }
        }
        self.readback.unmap();
        Ok(())
    }
}

fn tex_binding(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    let _ = format;
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

impl Compositor for GpuCompositor {
    fn update_background(&mut self, mode: BackgroundMode) -> Result<(), FrameiruError> {
        self.background = Background::resolve(mode)?;
        Ok(())
    }

    fn update_overlay(&mut self, overlay: OverlayMode) {
        self.overlay = overlay;
    }

    fn set_subject_light(&mut self, light: f32) {
        self.subject_light = light.clamp(0.0, 1.0);
    }

    fn composite(
        &mut self,
        source: &FrameBuffer,
        mask: &Mask,
        output: &mut FrameBuffer,
    ) -> Result<(), FrameiruError> {
        if source.metadata.format != PixelFormat::Rgb8 {
            return Err(FrameiruError::FormatMismatch {
                expected: PixelFormat::Rgb8,
                actual: source.metadata.format,
            });
        }
        let res = source.metadata.resolution;
        if mask.resolution != res || mask.data.len() < res.area() as usize {
            return Err(FrameiruError::InvalidArgument(format!(
                "mask resolution {:?} does not match source resolution {:?}",
                mask.resolution, res
            )));
        }
        output.metadata = source.metadata;
        let area = res.area() as usize;
        output.data.resize(area * 3, 0);

        let is_passthrough = matches!(self.background, Background::Passthrough);
        if is_passthrough && self.overlay == OverlayMode::None {
            output.data.copy_from_slice(&source.data[..area * 3]);
            return Ok(());
        }

        self.ensure_resources(res);

        // Foreground RGB -> RGBA upload.
        self.rgba.clear();
        for px in source.data[..area * 3].chunks_exact(3) {
            self.rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        Self::upload_rgba(&self.queue, &self.fg_tex, res, 4, &self.rgba);
        // Mask is f32 0..1; R8Unorm wants u8 0..255 (1 byte per pixel).
        self.mask_u8.clear();
        self.mask_u8.extend(
            mask.data[..area]
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8),
        );
        Self::upload_rgba(&self.queue, &self.mask_tex, res, 1, &self.mask_u8);

        // Globals uniform: written once per frame and shared by the bloom and
        // composite passes (the cyberpunk shader branch also reads it).
        let globals = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wrap = if matches!(
            self.background,
            Background::Image { .. } | Background::Video { .. }
        ) {
            0.5f32
        } else {
            0.0f32
        };
        let t = ((source.metadata.timestamp_us % (3600 * 1_000_000)) as f64 / 1_000_000.0) as f32;
        let passthrough_flag = if is_passthrough { 1.0f32 } else { 0.0f32 };
        self.queue.write_buffer(
            &globals,
            0,
            bytemuck::cast_slice(&[
                wrap,
                self.subject_light,
                self.overlay as u32 as f32,
                t,
                res.width as f32,
                res.height as f32,
                passthrough_flag,
                0.0,
            ]),
        );

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Background source per mode.
        match &self.background {
            Background::Blur { radius } => {
                let one = 1.0 / res.width as f32;
                let one_v = 1.0 / res.height as f32;
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("blur h"),
                        });
                self.queue.write_buffer(
                    &self.blur_uniform,
                    0,
                    bytemuck::cast_slice(&[*radius as f32, one, 0.0, 0.0]),
                );
                {
                    let bgl = self.blur_h.get_bind_group_layout(0);
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("blur h"),
                        layout: &bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.fg_tex.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.mask_tex.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Buffer(
                                    self.blur_uniform.as_entire_buffer_binding(),
                                ),
                            },
                        ],
                    });
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("blur h pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.blur_tex.create_view(&Default::default()),
                            resolve_target: None,
                            ops: wgpu::Operations::default(),
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rp.set_pipeline(&self.blur_h);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.draw(0..4, 0..1);
                }
                self.queue.submit([encoder.finish()]);

                // Second uniform write must land before the vertical pass is
                // submitted, hence the separate encoder/submit below.
                self.queue.write_buffer(
                    &self.blur_uniform,
                    0,
                    bytemuck::cast_slice(&[*radius as f32, 0.0, one_v, 0.0]),
                );
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("blur v"),
                        });
                {
                    let bgl = self.blur_v.get_bind_group_layout(0);
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("blur v"),
                        layout: &bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.blur_tex.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.mask_tex.create_view(&Default::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Buffer(
                                    self.blur_uniform.as_entire_buffer_binding(),
                                ),
                            },
                        ],
                    });
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("blur v pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.bg_tex.create_view(&Default::default()),
                            resolve_target: None,
                            ops: wgpu::Operations::default(),
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rp.set_pipeline(&self.blur_v);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.draw(0..4, 0..1);
                }
                self.queue.submit([encoder.finish()]);
            }
            Background::Color { r, g, b } => {
                let rgba = [*r, *g, *b, 255];
                Self::upload_rgba(&self.queue, &self.bg_tex, res, 4, &rgba.repeat(area));
            }
            Background::Image { data, resolution } => {
                // The background texture must match the image size so the
                // linear sampler can scale it to the frame in the shader.
                if self.bg_tex.size().width != resolution.width
                    || self.bg_tex.size().height != resolution.height
                {
                    self.bg_tex =
                        Self::create_texture(&self.device, *resolution, TEXTURE_FORMAT, "bg");
                }
                self.image_rgba.clear();
                for px in data.chunks_exact(3) {
                    self.image_rgba
                        .extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
                Self::upload_rgba(&self.queue, &self.bg_tex, *resolution, 4, &self.image_rgba);
            }
            Background::Video { source } => {
                // Same as image: texture sized to the video, re-uploaded with
                // the newest decoded frame on every composite.
                let frame = source.current();
                let res = frame.resolution;
                if self.bg_tex.size().width != res.width || self.bg_tex.size().height != res.height
                {
                    self.bg_tex = Self::create_texture(&self.device, res, TEXTURE_FORMAT, "bg");
                }
                self.image_rgba.clear();
                for px in frame.data.chunks_exact(3) {
                    self.image_rgba
                        .extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
                Self::upload_rgba(&self.queue, &self.bg_tex, res, 4, &self.image_rgba);
            }
            Background::Passthrough => {
                // In passthrough mode, bg_tex is not sampled in the shader;
                // self.bg_tex remains bound as a valid dummy texture.
            }
        }

        // Halation chain for the cyberpunk overlay: bright pass + blur at
        // 1/4 res, so the composite shader can add the halo additively.
        if self.overlay == OverlayMode::Cyberpunk {
            self.run_bloom(&sampler, &globals);
        }

        // Composite pass.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite"),
            });
        let bgl = self.composite.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.fg_tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &self.mask_tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &self.bg_tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(globals.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(
                        &self.bloom_tex.create_view(&Default::default()),
                    ),
                },
            ],
        });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output_tex.create_view(&Default::default()),
                    resolve_target: None,
                    ops: wgpu::Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.composite);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..4, 0..1);
        }
        self.queue.submit([encoder.finish()]);

        self.readback_rgb(res, &mut output.data)
    }
}
