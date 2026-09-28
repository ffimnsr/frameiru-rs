//! WGSL shaders for the GPU compositor (feature `gpu`).

/// Fullscreen-triangle pass: blend foreground with the mask over a background.
///
/// `bg_tex` is either the blurred foreground (blur mode), a 1x1 solid color
/// (color mode), or the user image (image mode); sampling handles scaling.
pub const COMPOSITE_WGSL: &str = r#"
struct Globals {
    // Unused for now; kept for future modes (e.g. feathering).
    _pad: vec4<f32>,
}

@group(0) @binding(0) var fg_tex: texture_2d<f32>;
@group(0) @binding(1) var fg_samp: sampler;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var bg_tex: texture_2d<f32>;
@group(0) @binding(4) var bg_samp: sampler;
@group(0) @binding(5) var<uniform> globals: Globals;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    // NDC is y-up, textures are y-down: bottom-left has uv.y = 1, so the
    // image does not come out vertically flipped.
    var uv = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
    o.uv = uv[vi];
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let fg = textureSample(fg_tex, fg_samp, in.uv);
    let bg = textureSample(bg_tex, bg_samp, in.uv);
    let m = textureSample(mask_tex, fg_samp, in.uv).r;
    let blended = mix(bg, fg, m);
    return vec4<f32>(blended.rgb, 1.0);
}
"#;

/// Separable box-blur pass; direction selected by uniform.
pub const BLUR_WGSL: &str = r#"
struct Params {
    // x: radius, y: step x (uv), z: step y (uv), w: unused
    params: vec4<f32>,
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    // NDC is y-up, textures are y-down: bottom-left has uv.y = 1, so the
    // image does not come out vertically flipped.
    var uv = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
    o.uv = uv[vi];
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let radius = u.params.x;
    let step = vec2<f32>(u.params.y, u.params.z);
    var acc = vec4<f32>(0.0);
    var n = 0.0;
    for (var i = 0.0; i <= radius; i += 1.0) {
        acc += textureSample(tex, samp, in.uv + step * i);
        if (i > 0.0) {
            acc += textureSample(tex, samp, in.uv - step * i);
            n += 2.0;
        } else {
            n += 1.0;
        }
    }
    return acc / n;
}
"#;
