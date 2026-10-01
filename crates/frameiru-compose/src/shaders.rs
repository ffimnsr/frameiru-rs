//! WGSL shaders for the GPU compositor (feature `gpu`).

/// Fullscreen-triangle pass: blend foreground with the mask over a background.
///
/// `bg_tex` is either the blurred foreground (blur mode), a 1x1 solid color
/// (color mode), or the user image (image mode); sampling handles scaling.
pub const COMPOSITE_WGSL: &str = r#"
struct Globals {
    // x: light-wrap strength (0 = off; 0.5 for image backgrounds)
    // y: subject fill light (0 = off; lifts the subject toward white)
    g: vec4<f32>,
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
    // Light wrap (U9.5): spill the background color over the subject edge
    // for image backgrounds, so the fringe takes the room's ambient colors.
    let fgw = mix(fg, bg, globals.g.x * (1.0 - m));
    let blended = mix(bg, fgw, m);
    // Subject fill light: lift the (masked) subject toward white.
    let lit = blended + globals.g.y * m * (vec4<f32>(1.0, 1.0, 1.0, 1.0) - blended);
    return vec4<f32>(lit.rgb, 1.0);
}
"#;

/// Foreground-aware separable blur pass; direction selected by uniform.
///
/// Kernel samples are weighted by `1 - alpha` and foreground pixels
/// (alpha >= 0.5) are excluded entirely, so bright subject colors cannot
/// smear over the matte edge into the background (halo-free, U9.5). With a
/// zero mask this degenerates to the plain box blur.
pub const BLUR_WGSL: &str = r#"
struct Params {
    // x: radius, y: step x (uv), z: step y (uv), w: unused
    params: vec4<f32>,
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var mask_samp: sampler;
@group(0) @binding(4) var<uniform> u: Params;

fn fw(am: f32) -> f32 {
    if (am >= 0.5) {
        return 0.0;
    }
    return 1.0 - am;
}

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
    var wsum = 0.0;
    for (var i = 0.0; i <= radius; i += 1.0) {
        var w = fw(textureSample(mask_tex, mask_samp, in.uv + step * i).r);
        acc += textureSample(tex, samp, in.uv + step * i) * w;
        wsum += w;
        if (i > 0.0) {
            w = fw(textureSample(mask_tex, mask_samp, in.uv - step * i).r);
            acc += textureSample(tex, samp, in.uv - step * i) * w;
            wsum += w;
        }
    }
    if (wsum <= 0.0) {
        // Fully foreground window: keep the center sample (subject stays
        // crisp; the composite masks this out at alpha == 1 anyway).
        return textureSample(tex, samp, in.uv);
    }
    return acc / wsum;
}
"#;
