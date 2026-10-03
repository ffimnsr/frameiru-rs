//! WGSL shaders for the GPU compositor (feature `gpu`).

/// Fullscreen-triangle pass: blend foreground with the mask over a background.
///
/// `bg_tex` is either the blurred foreground (blur mode), a 1x1 solid color
/// (color mode), or the user image (image mode); sampling handles scaling.
pub const COMPOSITE_WGSL: &str = r#"
struct Globals {
    // g.xy: light-wrap strength (0 = off; 0.5 for image/video backgrounds)
    //       and subject fill light (0 = off; lifts the subject toward white)
    // g.z:  overlay mode (0 off, 1 scanlines, 2 light leak, 3 CRT)
    // g.w:  time in seconds (overlay animation phase)
    // g2.xy: frame size in pixels (overlay coordinates)
    // g2.z:  passthrough mode (1.0 = passthrough, 0.0 = composite)
    // g2.w:  unused
    g: vec4<f32>,
    g2: vec4<f32>,
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
    var out_c: vec3<f32>;
    if (globals.g2.z > 0.5) {
        out_c = fg.rgb;
    } else {
        let bg = textureSample(bg_tex, bg_samp, in.uv);
        let m = textureSample(mask_tex, fg_samp, in.uv).r;
        // Light wrap (U9.5): spill the background color over the subject edge
        // for image backgrounds, so the fringe takes the room's ambient colors.
        let fgw = mix(fg, bg, globals.g.x * (1.0 - m));
        let blended = mix(bg, fgw, m);
        // Subject fill light: lift the (masked) subject toward white.
        let lit = blended + globals.g.y * m * (vec4<f32>(1.0, 1.0, 1.0, 1.0) - blended);
        out_c = lit.rgb;
    }

    // Full-frame overlay pass (mirrors the CPU compositor's formulas).
    let ov = globals.g.z;
    let t = globals.g.w;
    let px = vec2<f32>(in.uv.x * globals.g2.x, in.uv.y * globals.g2.y);
    if (ov == 1.0) {
        // Authentic 270-scanline look: 4-pixel period with 2px bright beam, 2px dark trough
        // 2-pixel width ensures scanlines survive bilinear minification in preview windows.
        let line = u32(in.pos.y) % 4u;
        if (line >= 2u) {
            out_c *= 0.50;
        }
    } else if (ov == 2.0) {
        let cx = (0.5 + 0.35 * sin(t * 0.7)) * globals.g2.x;
        let cy = (0.25 + 0.2 * sin(t * 0.9 + 1.3)) * globals.g2.y;
        let d2 = distance(px, vec2<f32>(cx, cy));
        let r = max(globals.g2.x, globals.g2.y) * 0.65;
        if (d2 < r) {
            let blob = 1.0 - d2 / r;
            let k = blob * blob * 0.35;
            out_c += vec3<f32>(1.0, 178.0 / 255.0, 89.0 / 255.0) * k;
        }
    } else if (ov == 3.0) {
        // CRT: Strong scanlines (4px period), rolling cathode beam, RGB phosphor triad, and curved vignette
        let line = u32(in.pos.y) % 4u;
        var scan = 1.0;
        if (line >= 2u) {
            scan = 0.45;
        }
        let roll = 1.0 + 0.10 * sin(in.uv.y * 12.56637 - t * 5.0);
        let rx = in.uv.x - 0.5;
        let ry = in.uv.y - 0.5;
        let vignette = clamp(1.0 - 1.5 * (rx * rx + ry * ry), 0.0, 1.0);
        let flicker = 1.0 + 0.05 * sin(t * 15.0);

        let sub = u32(in.pos.x) % 3u;
        var rgb_triad = vec3<f32>(0.9, 0.9, 0.9);
        if (sub == 0u) {
            rgb_triad = vec3<f32>(1.15, 0.88, 0.88);
        } else if (sub == 1u) {
            rgb_triad = vec3<f32>(0.88, 1.15, 0.88);
        } else {
            rgb_triad = vec3<f32>(0.88, 0.88, 1.15);
        }

        out_c = clamp(out_c * scan * roll * rgb_triad * vignette * flicker, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<f32>(out_c, 1.0);
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
