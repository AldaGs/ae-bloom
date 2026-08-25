/*	Bloom_Kernel_Metal.h

	Metal (MSL) port of Bloom_Kernel.cu, for the macOS GPU path.

	The kernels are a faithful 1:1 translation of the CUDA versions - same math,
	same coordinate discipline (OUTPUT / INPUT / CROPPED), same half-pixel
	alignment. The host orchestrator lives in Bloom_Metal.mm and mirrors
	BL_Bloom_CUDA step for step.

	The MSL source is embedded as a C string and compiled at RUNTIME via
	-[MTLDevice newLibraryWithSource:...], exactly like the SDK's
	SDK_Invert_ProcAmp sample. No .metallib build rule is needed.

	Differences from CUDA that are pure API, not algorithm:
	  - kernels use [[thread_position_in_grid]] instead of blockIdx/threadIdx
	  - scalar kernel args arrive as `constant T& [[buffer(n)]]` (set via setBytes)
	  - atomics use MSL atomic_int + atomic_fetch_min/max_explicit
	AE hands worlds as GPU_BGRA128 (float4, .x=B .y=G .z=R .w=A); internal
	buffers use natural RGBA, swizzled at the edges - identical to the CUDA path.
*/

#pragma once

static const char *kBloomKernelMetalString = R"BLMETAL(
#include <metal_stdlib>
using namespace metal;

#define BL_TONE_NONE_K       1
#define BL_TONE_REINHARD_K   2
#define BL_TONE_ACES_K       3
#define BL_VIEW_COMPOSITE_K  1
#define BL_VIEW_GLOW_K       2
#define BL_VIEW_GLOWBLACK_K  3
#define BL_UNMULT_AFLOOR_K   0.05f

// LAYOUT MUST MATCH BLGpuParams in Bloom.cpp / Bloom_Metal.mm exactly.
struct BLGpuParams {
    int   offX, offY;
    int   inW, inH;
    int   W, H;
    int   srcPitch, dstPitch;
    int   pad;
    int   wx0, wy0, CW, CH;
    float scatter, gain, bias, threshold;
    float huePreserve, saturation;
    float tintR, tintG, tintB, tintAmt;
    float caAmt, caCX, caCY;
    float streakInten, streakAngle, streakLen, streakThr;
    float streakSpread;
    int   streakCount, streakQuality;
    int   tone, unmult, karis, view;
    int   inputStraight;
};

/* ---- tiny buffer ops (replace CUDA cudaMemset / cudaMemcpy D2D, so the whole
   pipeline stays inside one serial compute encoder - no blit encoder needed) - */

kernel void bl_clear(device float4 *b   [[buffer(0)]],
                     constant int  &n   [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if ((int)gid < n) b[gid] = float4(0.0f);
}

kernel void bl_copybuf(const device float4 *s   [[buffer(0)]],
                       device float4       *dst [[buffer(1)]],
                       constant int        &n   [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if ((int)gid < n) dst[gid] = s[gid];
}

/* ------------------------------------------------------------------ helpers */

inline float bl_straight01(float v) {
    return (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
}

inline float bl_srgb_to_linear(float c) {
    if (c <= 0.0f) return 0.0f;
    if (c <= 0.04045f) return c / 12.92f;
    return pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float bl_linear_to_srgb(float c) {
    if (c <= 0.0f) return 0.0f;
    if (c <= 0.0031308f) return c * 12.92f;
    return 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
}

inline float bl_luma(float4 p) { return p.x*0.2126f + p.y*0.7152f + p.z*0.0722f; }

/* Bilinear fetch from a level buffer at fractional index coords, clamped. */
inline float4 bl_fetch(const device float4 *buf, int w, int h, float fx, float fy) {
    fx = fmin(fmax(fx, 0.0f), (float)(w - 1));
    fy = fmin(fmax(fy, 0.0f), (float)(h - 1));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = (x0 + 1 < w) ? x0 + 1 : x0;
    int y1 = (y0 + 1 < h) ? y0 + 1 : y0;
    float tx = fx - x0, ty = fy - y0;
    float4 p00 = buf[y0*w + x0];
    float4 p01 = buf[y0*w + x1];
    float4 p10 = buf[y1*w + x0];
    float4 p11 = buf[y1*w + x1];
    float4 a0 = mix(p00, p01, tx);
    float4 a1 = mix(p10, p11, tx);
    return mix(a0, a1, ty);
}

/* Read the INPUT world at an input pixel, as straight RGBA. Outside = 0. */
inline float4 bl_read_in(const device float4 *src, constant BLGpuParams &p, int u, int v) {
    if (u < 0 || v < 0 || u >= p.inW || v >= p.inH)
        return float4(0.0f);
    float4 c = src[v * p.srcPitch + u];
    return float4(c.z, c.y, c.x, c.w);   // BGRA -> RGBA
}

/* Clamped bilinear fetch from the cropped glow buffer. */
inline float4 bl_glow_bilinear(const device float4 *glow, int CW, int CH, float fx, float fy) {
    fx = fmin(fmax(fx, 0.0f), (float)(CW - 1));
    fy = fmin(fmax(fy, 0.0f), (float)(CH - 1));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = (x0 + 1 < CW) ? x0 + 1 : x0;
    int y1 = (y0 + 1 < CH) ? y0 + 1 : y0;
    float tx = fx - x0, ty = fy - y0;
    float4 p00 = glow[y0*CW + x0], p01 = glow[y0*CW + x1];
    float4 p10 = glow[y1*CW + x0], p11 = glow[y1*CW + x1];
    float4 a0 = mix(p00, p01, tx);
    float4 a1 = mix(p10, p11, tx);
    return mix(a0, a1, ty);
}

inline float bl_tone_scalar(float x, int mode, float white) {
    if (x <= 0.0f) return 0.0f;
    if (mode == BL_TONE_REINHARD_K) return x / (1.0f + x / white);
    if (mode == BL_TONE_ACES_K) {
        float a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
        return fmax((x * (a*x + b)) / (x * (c*x + d) + e), 0.0f);
    }
    return x;
}

/* ------------------------------------------------------------------- bbox */

kernel void bl_bbox(const device float4 *src   [[buffer(0)]],
                    constant BLGpuParams &p     [[buffer(1)]],
                    device atomic_int   *box    [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    int u = gid.x, v = gid.y;
    if (u >= p.inW || v >= p.inH) return;
    float4 s = src[v * p.srcPitch + u];
    if (s.w > 0.0f) {
        atomic_fetch_min_explicit(&box[0], u, memory_order_relaxed);
        atomic_fetch_min_explicit(&box[1], v, memory_order_relaxed);
        atomic_fetch_max_explicit(&box[2], u, memory_order_relaxed);
        atomic_fetch_max_explicit(&box[3], v, memory_order_relaxed);
        if (s.w < 1.0f) {
            float smax = fmax(fmax(s.x, s.y), s.z);
            if (smax > s.w + (1.0f / 255.0f))
                atomic_fetch_max_explicit(&box[4], 1, memory_order_relaxed);
        }
    }
}

/* ---------------------------------------------------------------- extract */

kernel void bl_extract(const device float4 *src [[buffer(0)]],
                       device float4 *lv0        [[buffer(1)]],
                       constant BLGpuParams &p    [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int cx = gid.x, cy = gid.y;
    if (cx >= p.CW || cy >= p.CH) return;

    int u = cx + p.wx0 - p.pad - p.offX;
    int v = cy + p.wy0 - p.pad - p.offY;
    float4 s = bl_read_in(src, p, u, v);
    float  a = s.w;

    float ia = p.inputStraight ? 1.0f
             : (1.0f / ((a > BL_UNMULT_AFLOOR_K) ? a : BL_UNMULT_AFLOOR_K));
    float3 linS = float3(bl_srgb_to_linear(bl_straight01(s.x * ia)),
                         bl_srgb_to_linear(bl_straight01(s.y * ia)),
                         bl_srgb_to_linear(bl_straight01(s.z * ia)));
    float3 emit;
    if (p.unmult) {
        emit = (a > 1e-4f) ? linS : float3(0.0f);
    } else {
        emit = float3(linS.x*a, linS.y*a, linS.z*a);
    }
    float lum = emit.x*0.2126f + emit.y*0.7152f + emit.z*0.0722f;
    if (p.bias > 0.0f) {
        emit.x = emit.x*(1.0f-p.bias) + emit.x*lum*p.bias;
        emit.y = emit.y*(1.0f-p.bias) + emit.y*lum*p.bias;
        emit.z = emit.z*(1.0f-p.bias) + emit.z*lum*p.bias;
    }
    float gate = 1.0f;
    if (p.threshold > 0.0f) {
        float k = fmax(p.threshold * 0.5f, 1e-4f);
        float soft = fmin(fmax(lum - p.threshold + k, 0.0f), 2.0f*k);
        soft = soft * soft / (4.0f * k);
        float g = fmax(fmax(soft, lum - p.threshold), 0.0f);
        gate = (lum > 1e-6f) ? fmin(g / lum, 1.0f) : 0.0f;
    }
    lv0[cy * p.CW + cx] = float4(emit.x*gate, emit.y*gate, emit.z*gate, a*gate);
}

/* -------------------------------------------------------------- downsample */

kernel void bl_down13(const device float4 *src [[buffer(0)]],
                      constant int &sw           [[buffer(1)]],
                      constant int &sh           [[buffer(2)]],
                      device float4 *dst         [[buffer(3)]],
                      constant int &dw           [[buffer(4)]],
                      constant int &dh           [[buffer(5)]],
                      constant int &karis        [[buffer(6)]],
                      uint2 gid [[thread_position_in_grid]]) {
    int dx = gid.x, dy = gid.y;
    if (dx >= dw || dy >= dh) return;

    float cx = 2.0f * dx + 0.5f;
    float cy = 2.0f * dy + 0.5f;

    const float ox[13] = { 0.f, -2.f, 0.f, 2.f, -2.f, 2.f, -2.f, 0.f, 2.f, -1.f, 1.f, -1.f, 1.f };
    const float oy[13] = { 0.f, -2.f,-2.f,-2.f,  0.f, 0.f,  2.f, 2.f, 2.f, -1.f,-1.f,  1.f, 1.f };
    const float ow[13] = { 0.125f, 0.03125f, 0.0625f, 0.03125f,
                           0.0625f, 0.0625f, 0.03125f, 0.0625f, 0.03125f,
                           0.125f, 0.125f, 0.125f, 0.125f };
    float4 acc = float4(0.0f);
    float wsum = 0.0f;
    for (int t = 0; t < 13; t++) {
        float4 s = bl_fetch(src, sw, sh, cx + ox[t], cy + oy[t]);
        float w = karis ? (ow[t] / (1.0f + bl_luma(s))) : ow[t];
        acc += s * w;
        wsum += w;
    }
    float inv = (karis && wsum > 1e-8f) ? (1.0f / wsum) : 1.0f;
    dst[dy * dw + dx] = acc * inv;
}

kernel void bl_scale(const device float4 *src [[buffer(0)]],
                     device float4 *dst         [[buffer(1)]],
                     constant int &w            [[buffer(2)]],
                     constant int &h            [[buffer(3)]],
                     constant float &k          [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= w || y >= h) return;
    dst[y * w + x] = src[y * w + x] * k;
}

/* ---------------------------------------------------------------- upsample */

kernel void bl_up_tent(const device float4 *small [[buffer(0)]],
                       constant int &sw            [[buffer(1)]],
                       constant int &sh            [[buffer(2)]],
                       const device float4 *level  [[buffer(3)]],
                       device float4 *dst          [[buffer(4)]],
                       constant int &dw            [[buffer(5)]],
                       constant int &dh            [[buffer(6)]],
                       constant float &wk          [[buffer(7)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= dw || y >= dh) return;

    const float tw[9] = { 1.f/16, 2.f/16, 1.f/16,
                          2.f/16, 4.f/16, 2.f/16,
                          1.f/16, 2.f/16, 1.f/16 };
    float4 acc = float4(0.0f);
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            float dxc = fmin(fmax((float)(x + kx), 0.0f), (float)(dw - 1));
            float dyc = fmin(fmax((float)(y + ky), 0.0f), (float)(dh - 1));
            float fx = 0.5f * dxc - 0.25f;
            float fy = 0.5f * dyc - 0.25f;
            float4 s = bl_fetch(small, sw, sh, fx, fy);
            float w = tw[(ky + 1) * 3 + (kx + 1)];
            acc += s * w;
        }
    }
    float4 L = level[y * dw + x];
    dst[y * dw + x] = acc + L * wk;
}

/* --------------------------------------------------------------- composite */

kernel void bl_composite(const device float4 *src  [[buffer(0)]],
                         const device float4 *glow  [[buffer(1)]],
                         device float4 *dst          [[buffer(2)]],
                         constant BLGpuParams &p      [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;

    float4 s = bl_read_in(src, p, x - p.offX, y - p.offY);

    int cx = x + p.pad - p.wx0;
    int cy = y + p.pad - p.wy0;
    if (cx < 0 || cy < 0 || cx >= p.CW || cy >= p.CH) {
        if (p.view == BL_VIEW_COMPOSITE_K)
            dst[y * p.dstPitch + x] = float4(s.z, s.y, s.x, s.w);
        else
            dst[y * p.dstPitch + x] = float4(0.0f, 0.0f, 0.0f,
                (p.view == BL_VIEW_GLOWBLACK_K) ? 1.0f : 0.0f);
        return;
    }

    float a = s.w;

    float4 g;
    if (p.caAmt != 0.0f) {
        float vx = (float)x - p.caCX, vy = (float)y - p.caCY;
        float4 gc = glow[cy * p.CW + cx];
        float4 sR = bl_glow_bilinear(glow, p.CW, p.CH, cx + vx*(-p.caAmt), cy + vy*(-p.caAmt));
        float4 sB = bl_glow_bilinear(glow, p.CW, p.CH, cx + vx*( p.caAmt), cy + vy*( p.caAmt));
        g = float4(sR.x, gc.y, sB.z, gc.w);
    } else {
        g = glow[cy * p.CW + cx];
    }
    float k = p.scatter * p.gain;

    float3 gT = float3(g.x, g.y, g.z);
    if (p.tintAmt > 0.0f) {
        float gL = 0.2126f*g.x + 0.7152f*g.y + 0.0722f*g.z;
        gT.x = g.x*(1.0f-p.tintAmt) + (p.tintR*gL)*p.tintAmt;
        gT.y = g.y*(1.0f-p.tintAmt) + (p.tintG*gL)*p.tintAmt;
        gT.z = g.z*(1.0f-p.tintAmt) + (p.tintB*gL)*p.tintAmt;
    }

    float srcW = (p.view == BL_VIEW_COMPOSITE_K) ? (1.0f - p.scatter) : 0.0f;

    float glowA = fmin(fmax(g.w * p.scatter, 0.0f), 1.0f);
    float3 glowDisp = float3(0.0f);
    if (glowA > 1e-4f) {
        float ig = 1.0f / glowA;
        float3 st = float3(gT.x*k*ig, gT.y*k*ig, gT.z*k*ig);
        if (p.tone != BL_TONE_NONE_K) {
            float L  = st.x*0.2126f + st.y*0.7152f + st.z*0.0722f;
            float Lt = bl_tone_scalar(L, p.tone, 4.0f);
            float ratio = (L > 1e-6f) ? (Lt / L) : 0.0f;
            float hp = p.huePreserve;
            float3 pc = float3(bl_tone_scalar(st.x, p.tone, 4.0f),
                               bl_tone_scalar(st.y, p.tone, 4.0f),
                               bl_tone_scalar(st.z, p.tone, 4.0f));
            st.x = fmin(fmax(hp*(st.x*ratio) + (1.0f-hp)*pc.x, 0.0f), 1.0f);
            st.y = fmin(fmax(hp*(st.y*ratio) + (1.0f-hp)*pc.y, 0.0f), 1.0f);
            st.z = fmin(fmax(hp*(st.z*ratio) + (1.0f-hp)*pc.z, 0.0f), 1.0f);
        }
        if (p.saturation != 1.0f) {
            float L = st.x*0.2126f + st.y*0.7152f + st.z*0.0722f;
            st.x = fmax(L + (st.x - L)*p.saturation, 0.0f);
            st.y = fmax(L + (st.y - L)*p.saturation, 0.0f);
            st.z = fmax(L + (st.z - L)*p.saturation, 0.0f);
        }
        glowDisp = float3(bl_linear_to_srgb(st.x*glowA),
                          bl_linear_to_srgb(st.y*glowA),
                          bl_linear_to_srgb(st.z*glowA));
    }

    float ia = p.inputStraight ? 1.0f : ((a > 1e-4f) ? (1.0f / a) : 0.0f);
    float artR = bl_linear_to_srgb(bl_srgb_to_linear(bl_straight01(s.x*ia))*srcW) * a;
    float artG = bl_linear_to_srgb(bl_srgb_to_linear(bl_straight01(s.y*ia))*srcW) * a;
    float artB = bl_linear_to_srgb(bl_srgb_to_linear(bl_straight01(s.z*ia))*srcW) * a;
    float3 outRGB = float3(fmin(artR + glowDisp.x, 1.0f),
                           fmin(artG + glowDisp.y, 1.0f),
                           fmin(artB + glowDisp.z, 1.0f));

    float outA = (p.view == BL_VIEW_COMPOSITE_K) ? a
               : (p.view == BL_VIEW_GLOW_K)       ? glowA
               :                                    1.0f;
    float mx = fmax(fmax(outRGB.x, outRGB.y), outRGB.z);
    float finalA = fmin(fmax(outA, mx), 1.0f);

    dst[y * p.dstPitch + x] = float4(outRGB.z, outRGB.y, outRGB.x, finalA); // RGBA -> BGRA
}

kernel void bl_copythrough(const device float4 *src [[buffer(0)]],
                           device float4 *dst        [[buffer(1)]],
                           constant BLGpuParams &p    [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;
    if (p.view == BL_VIEW_COMPOSITE_K) {
        float4 s = bl_read_in(src, p, x - p.offX, y - p.offY);
        dst[y * p.dstPitch + x] = float4(s.z, s.y, s.x, s.w);
    } else {
        dst[y * p.dstPitch + x] = float4(0.0f, 0.0f, 0.0f,
            (p.view == BL_VIEW_GLOWBLACK_K) ? 1.0f : 0.0f);
    }
}

/* ---- streaks -------------------------------------------------------------- */

kernel void bl_streak_step(const device float4 *A [[buffer(0)]],
                           device float4 *B        [[buffer(1)]],
                           constant int &CW         [[buffer(2)]],
                           constant int &CH         [[buffer(3)]],
                           constant float &ox       [[buffer(4)]],
                           constant float &oy       [[buffer(5)]],
                           constant float &wk       [[buffer(6)]],
                           uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= CW || y >= CH) return;
    float4 a = A[y * CW + x];
    float4 s = bl_glow_bilinear(A, CW, CH, (float)x - ox, (float)y - oy);
    B[y * CW + x] = a + wk * s;
}

kernel void bl_streak_spread(const device float4 *A [[buffer(0)]],
                             device float4 *B        [[buffer(1)]],
                             constant int &CW         [[buffer(2)]],
                             constant int &CH         [[buffer(3)]],
                             constant float &qx       [[buffer(4)]],
                             constant float &qy       [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= CW || y >= CH) return;
    float4 a  = A[y * CW + x];
    float4 sp = bl_glow_bilinear(A, CW, CH, (float)x + qx, (float)y + qy);
    float4 sm = bl_glow_bilinear(A, CW, CH, (float)x - qx, (float)y - qy);
    B[y * CW + x] = 0.5f*a + 0.25f*(sp + sm);
}

kernel void bl_streak_add(device float4 *acc         [[buffer(0)]],
                          const device float4 *full   [[buffer(1)]],
                          const device float4 *core    [[buffer(2)]],
                          constant int &CW             [[buffer(3)]],
                          constant int &CH             [[buffer(4)]],
                          constant float &scale        [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= CW || y >= CH) return;
    int i = y * CW + x;
    acc[i] += scale * (full[i] - core[i]);
}

kernel void bl_streak_gate(const device float4 *src [[buffer(0)]],
                           device float4 *dst        [[buffer(1)]],
                           constant int &CW           [[buffer(2)]],
                           constant int &CH           [[buffer(3)]],
                           constant float &thr        [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= CW || y >= CH) return;
    int i = y * CW + x;
    float4 s = src[i];
    float lum = s.x*0.2126f + s.y*0.7152f + s.z*0.0722f;
    float g = 1.0f;
    if (thr > 0.0f) {
        float k = (thr*0.5f > 1e-4f) ? thr*0.5f : 1e-4f;
        float soft = lum - thr + k;
        soft = fmin(fmax(soft, 0.0f), 2.0f*k);
        soft = soft*soft / (4.0f*k);
        float hard = lum - thr;
        float gg = fmax((soft > hard) ? soft : hard, 0.0f);
        g = (lum > 1e-6f) ? fmin(gg / lum, 1.0f) : 0.0f;
    }
    dst[i] = s * g;
}

kernel void bl_streak_blur(const device float4 *src [[buffer(0)]],
                           device float4 *dst        [[buffer(1)]],
                           constant int &CW           [[buffer(2)]],
                           constant int &CH           [[buffer(3)]],
                           constant int &horiz        [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= CW || y >= CH) return;
    const float kf[5] = { 1.f/16, 4.f/16, 6.f/16, 4.f/16, 1.f/16 };
    float4 a = float4(0.0f);
    for (int t = -2; t <= 2; t++) {
        int u = horiz ? x + t : x;
        int v = horiz ? y : y + t;
        if (u < 0) u = 0; else if (u >= CW) u = CW - 1;
        if (v < 0) v = 0; else if (v >= CH) v = CH - 1;
        float4 pv = src[v * CW + u];
        a += pv * kf[t + 2];
    }
    dst[y * CW + x] = a;
}
)BLMETAL";
