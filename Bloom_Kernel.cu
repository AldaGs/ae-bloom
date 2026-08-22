/*	Bloom_Kernel.cu

	GPU (CUDA) path for Bloom - STAGE 2: the full pyramid on device.

	The pyramid maps onto GPU passes almost perfectly: every octave is one
	independent kernel launch over a smaller grid, exactly the log-many-passes
	shape as the JFA in Buildable Stroke. No sequential dependency anywhere.

	    bl_bbox      device atomic bbox -> work rect (crop for perf/memory)
	    bl_extract   sRGB->linear, unmult, bias, soft-knee gate -> level 0
	    bl_down13    one octave DOWN  (13 bilinear taps, Karis on level 1 only)
	    bl_up_tent   one octave UP    (bilinear + 3x3 tent) + w_k * level k
	    bl_composite (1-scatter)*src + scatter*gain*glow, tone map, saturate

	COORDINATE DISCIPLINE (same three spaces as the CPU path):
	    OUTPUT  (x,y)   the world we write; 0..W-1, 0..H-1
	    INPUT   (u,v)   u = x - offX
	    CROPPED (cx,cy) cx = x + pad - wx0   (all pyramid work lives here)

	HALF-PIXEL ALIGNMENT (the bug that shipped in stage 1): destination pixel dx
	covers source pixels {2dx, 2dx+1}, so its centre is at source index 2dx+0.5,
	NOT 2dx. The 13 taps therefore land on half-integer positions and MUST be
	bilinear fetches. Centring at 2dx instead costs half a pixel per octave -
	0.5*2^k full-res px at level k - which summed over ~7 levels is tens of
	pixels of DOWN-RIGHT drift. The upsample's (dx+0.5)*ratio-0.5 mapping is the
	exact inverse of the CORRECTLY centred downsample.

	AE hands us GPU worlds as GPU_BGRA128: float4, .x=B .y=G .z=R .w=A, rows
	separated by `pitch` float4s. Internal buffers use natural RGBA order
	(.x=R .y=G .z=B .w=A) so the math reads normally; we swizzle at the edges.
*/

#include <cuda_runtime.h>
#include <math.h>

#define BL_TONE_NONE_K		1
#define BL_TONE_REINHARD_K	2
#define BL_TONE_ACES_K		3

#define BL_VIEW_COMPOSITE_K	1
#define BL_VIEW_GLOW_K		2
#define BL_VIEW_GLOWBLACK_K	3

// Must match BL_UNMULT_AFLOOR in Bloom.h: floors the unmult divisor so a
// near-zero-coverage edge pixel can't explode into a firefly.
#define BL_UNMULT_AFLOOR_K	0.05f

// POD params. LAYOUT MUST MATCH the BLGpuParams struct in Bloom.cpp exactly.
struct BLGpuParams {
	int		offX, offY;
	int		inW, inH;
	int		W, H;
	int		srcPitch, dstPitch;
	int		pad;
	int		wx0, wy0, CW, CH;
	float	scatter, gain, bias, threshold;
	float	huePreserve, saturation;
	float	tintR, tintG, tintB, tintAmt;
	float	caAmt, caCX, caCY;
	float	streakInten, streakAngle, streakLen, streakThr;
	float	streakSpread;			// lateral fan (see BL_STREAK_SPREAD_MAX)
	int		streakCount, streakQuality;
	int		tone, unmult, karis, view;
	int		inputStraight;			// 1 = straight rgb, 0 = premultiplied
};

#define BL_STREAK_MAX_AXES_K	8
#define BL_STREAK_EPS_K			0.02f
#define BL_SQ_FULL_K			2

/* ------------------------------------------------------------------ helpers */

/* Mirrors BL_Straight01 in Bloom.cpp: a premultiplied pixel is legal only when
   rgb <= alpha, so the straight colour recovered by dividing out coverage can
   never exceed 1. Without this a preserved-RGB fringe pixel gets multiplied by
   up to 1/BL_UNMULT_AFLOOR_K and clips to white. */
__device__ __forceinline__ float
bl_straight01 (float v)
{
	return (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
}

/* Mirrors BL_UnmultScale in Bloom.cpp - see the full rationale there. Only
   un-premultiply a pixel that actually IS premultiplied (rgb <= alpha). */
/* UNUSED - rejected approach, see the note in bl_extract. */
__device__ __forceinline__ float
bl_unmult_scale (float4 s, float a, float afloor)
{
	float smax = (s.x > s.y) ? s.x : s.y;
	if (s.z > smax) smax = s.z;
	if (smax > a + (1.0f / 255.0f)) return 1.0f;	// already straight
	return 1.0f / ((a > afloor) ? a : afloor);
}

__device__ __forceinline__ float
bl_srgb_to_linear (float c)
{
	if (c <= 0.0f) return 0.0f;
	if (c <= 0.04045f) return c / 12.92f;
	return __powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float
bl_linear_to_srgb (float c)
{
	if (c <= 0.0f) return 0.0f;
	if (c <= 0.0031308f) return c * 12.92f;
	return 1.055f * __powf(c, 1.0f / 2.4f) - 0.055f;
}

__device__ __forceinline__ float
bl_luma (float4 p) { return p.x*0.2126f + p.y*0.7152f + p.z*0.0722f; }

/* Bilinear fetch from a CROPPED-space level buffer at fractional index coords
   (index i == centre of pixel i), clamped at the edges. Clamping is safe: the
   work rect is (shape bbox + pad), so its border is already ~0. */
__device__ __forceinline__ float4
bl_fetch (const float4 *buf, int w, int h, float fx, float fy)
{
	fx = fminf(fmaxf(fx, 0.0f), (float)(w - 1));
	fy = fminf(fmaxf(fy, 0.0f), (float)(h - 1));
	const int x0 = (int)fx, y0 = (int)fy;
	const int x1 = (x0 + 1 < w) ? x0 + 1 : x0;
	const int y1 = (y0 + 1 < h) ? y0 + 1 : y0;
	const float tx = fx - x0, ty = fy - y0;

	const float4 p00 = buf[(size_t)y0*w + x0];
	const float4 p01 = buf[(size_t)y0*w + x1];
	const float4 p10 = buf[(size_t)y1*w + x0];
	const float4 p11 = buf[(size_t)y1*w + x1];

	float4 r;
	r.x = (p00.x + (p01.x-p00.x)*tx) + ((p10.x + (p11.x-p10.x)*tx) - (p00.x + (p01.x-p00.x)*tx))*ty;
	r.y = (p00.y + (p01.y-p00.y)*tx) + ((p10.y + (p11.y-p10.y)*tx) - (p00.y + (p01.y-p00.y)*tx))*ty;
	r.z = (p00.z + (p01.z-p00.z)*tx) + ((p10.z + (p11.z-p10.z)*tx) - (p00.z + (p01.z-p00.z)*tx))*ty;
	r.w = (p00.w + (p01.w-p00.w)*tx) + ((p10.w + (p11.w-p10.w)*tx) - (p00.w + (p01.w-p00.w)*tx))*ty;
	return r;
}

/* Read the INPUT world at an input pixel, as straight RGBA. Outside = 0. */
__device__ __forceinline__ float4
bl_read_in (const float4 *src, const BLGpuParams &p, int u, int v)
{
	if (u < 0 || v < 0 || u >= p.inW || v >= p.inH)
		return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	const float4 c = src[(size_t)v * p.srcPitch + u];
	return make_float4(c.z, c.y, c.x, c.w);		// BGRA -> RGBA
}

/* ------------------------------------------------------------------- bbox */

__global__ void
bl_bbox (const float4 *src, BLGpuParams p, int *box)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= p.inW || v >= p.inH) return;

	const float4 s = src[(size_t)v * p.srcPitch + u];
	if (s.w > 0.0f) {
		atomicMin(&box[0], u);  atomicMin(&box[1], v);
		atomicMax(&box[2], u);  atomicMax(&box[3], v);

		// box[4]: per-FRAME premultiplied-vs-straight decision, folded into the
		// reduction we are already doing. A legal premultiplied pixel has
		// rgb <= alpha, so one partial-coverage pixel above that proves the
		// frame carries straight rgb. Mirrors the CPU scan in Bloom.cpp.
		if (s.w < 1.0f) {
			float smax = (s.x > s.y) ? s.x : s.y;
			if (s.z > smax) smax = s.z;
			if (smax > s.w + (1.0f / 255.0f)) atomicMax(&box[4], 1);
		}
	}
}

/* ---------------------------------------------------------------- extract */

__global__ void
bl_extract (const float4 *src, float4 *lv0, BLGpuParams p)
{
	const int cx = blockIdx.x * blockDim.x + threadIdx.x;
	const int cy = blockIdx.y * blockDim.y + threadIdx.y;
	if (cx >= p.CW || cy >= p.CH) return;

	const int u = cx + p.wx0 - p.pad - p.offX;
	const int v = cy + p.wy0 - p.pad - p.offY;
	const float4 s = bl_read_in(src, p, u, v);
	const float  a = s.w;

	// AE hands us PREMULTIPLIED pixels. Recover the straight colour BEFORE
	// linearizing - the gamma of a premultiplied value is not the premultiplied
	// gamma. The divisor is FLOORED so a near-zero-alpha edge pixel can't blow up
	// into a firefly (which CA would split into coloured specks).
	// Un-premultiply only if bl_bbox decided this frame is premultiplied. See
	// the long note at the matching spot in Bloom.cpp.
	const float ia = p.inputStraight ? 1.0f
					 : (1.0f / ((a > BL_UNMULT_AFLOOR_K) ? a : BL_UNMULT_AFLOOR_K));
	const float3 linS = make_float3(bl_srgb_to_linear(bl_straight01(s.x * ia)),
									bl_srgb_to_linear(bl_straight01(s.y * ia)),
									bl_srgb_to_linear(bl_straight01(s.z * ia)));

	// UNMULT recovers the true surface colour at semi-transparent edges, which
	// otherwise emit too little light (a dark fringe around soft edges). Without
	// it the emitted light is just the premultiplied value - which IS the light.
	float3 emit;
	if (p.unmult) {
		emit = (a > 1e-4f) ? linS : make_float3(0.0f, 0.0f, 0.0f);
	} else {
		emit = make_float3(linS.x*a, linS.y*a, linS.z*a);
	}

	const float lum = emit.x*0.2126f + emit.y*0.7152f + emit.z*0.0722f;

	// artistic: blend toward the luma-weighted (quadratic) extract
	if (p.bias > 0.0f) {
		emit.x = emit.x*(1.0f-p.bias) + emit.x*lum*p.bias;
		emit.y = emit.y*(1.0f-p.bias) + emit.y*lum*p.bias;
		emit.z = emit.z*(1.0f-p.bias) + emit.z*lum*p.bias;
	}

	// soft-knee threshold, normalized so the extract stays LINEAR above the knee
	float gate = 1.0f;
	if (p.threshold > 0.0f) {
		const float k = fmaxf(p.threshold * 0.5f, 1e-4f);
		float soft = fminf(fmaxf(lum - p.threshold + k, 0.0f), 2.0f*k);
		soft = soft * soft / (4.0f * k);
		const float g = fmaxf(fmaxf(soft, lum - p.threshold), 0.0f);
		gate = (lum > 1e-6f) ? fminf(g / lum, 1.0f) : 0.0f;
	}

	lv0[(size_t)cy * p.CW + cx] =
		make_float4(emit.x*gate, emit.y*gate, emit.z*gate, a*gate);
}

/* -------------------------------------------------------------- downsample */

__global__ void
bl_down13 (const float4 *src, int sw, int sh,
		   float4 *dst, int dw, int dh, int karis)
{
	const int dx = blockIdx.x * blockDim.x + threadIdx.x;
	const int dy = blockIdx.y * blockDim.y + threadIdx.y;
	if (dx >= dw || dy >= dh) return;

	// dest pixel centre in SOURCE index space - the half-pixel that matters.
	const float cx = 2.0f * dx + 0.5f;
	const float cy = 2.0f * dy + 0.5f;

	const float ox[13] = { 0.f, -2.f, 0.f, 2.f, -2.f, 2.f, -2.f, 0.f, 2.f,
						  -1.f, 1.f, -1.f, 1.f };
	const float oy[13] = { 0.f, -2.f,-2.f,-2.f,  0.f, 0.f,  2.f, 2.f, 2.f,
						  -1.f,-1.f,  1.f, 1.f };
	const float ow[13] = { 0.125f, 0.03125f, 0.0625f, 0.03125f,
						   0.0625f, 0.0625f, 0.03125f, 0.0625f, 0.03125f,
						   0.125f, 0.125f, 0.125f, 0.125f };

	float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float wsum = 0.0f;

	#pragma unroll
	for (int t = 0; t < 13; t++) {
		const float4 s = bl_fetch(src, sw, sh, cx + ox[t], cy + oy[t]);
		// Karis: one blazing pixel otherwise dominates the box and FLICKERS as
		// it moves sub-pixel between frames. Only worth it on the first octave.
		const float w = karis ? (ow[t] / (1.0f + bl_luma(s))) : ow[t];
		acc.x += s.x*w; acc.y += s.y*w; acc.z += s.z*w; acc.w += s.w*w;
		wsum += w;
	}
	const float inv = (karis && wsum > 1e-8f) ? (1.0f / wsum) : 1.0f;

	dst[(size_t)dy * dw + dx] =
		make_float4(acc.x*inv, acc.y*inv, acc.z*inv, acc.w*inv);
}

/* Seed the accumulator with the coarsest level, scaled by its weight. */
__global__ void
bl_scale (const float4 *src, float4 *dst, int w, int h, float k)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	const float4 s = src[(size_t)y * w + x];
	dst[(size_t)y * w + x] = make_float4(s.x*k, s.y*k, s.z*k, s.w*k);
}

/* ---------------------------------------------------------------- upsample */

/* One octave up: bilinear from `small` + 3x3 tent, plus this level's weighted
   contribution. The tent is evaluated as 9 bilinear fetches inline, so no temp
   buffer is needed and there is no in-place aliasing hazard (the CPU path needs
   three distinct buffers for exactly this reason). */
__global__ void
bl_up_tent (const float4 *small, int sw, int sh,
			const float4 *level, float4 *dst, int dw, int dh, float wk)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dw || y >= dh) return;

	const float tw[9] = { 1.f/16, 2.f/16, 1.f/16,
						  2.f/16, 4.f/16, 2.f/16,
						  1.f/16, 2.f/16, 1.f/16 };

	float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	#pragma unroll
	for (int ky = -1; ky <= 1; ky++) {
		for (int kx = -1; kx <= 1; kx++) {
			const float dxc = fminf(fmaxf((float)(x + kx), 0.0f), (float)(dw - 1));
			const float dyc = fminf(fmaxf((float)(y + ky), 0.0f), (float)(dh - 1));
			// FIXED factor-of-two inverse of bl_down13 ("dest d <- source 2d+0.5").
			// NOT a generic (X+0.5)*sw/dw-0.5 resize: with dw odd, sw = dw/2 floor
			// so sw/dw != 0.5, which is a SCALE error that compounds per octave
			// (measured: 2.6px drift over 4 levels at dw=101). Parity-independent
			// here; edge clamping absorbs the row/column the downsample dropped.
			const float fx = 0.5f * dxc - 0.25f;
			const float fy = 0.5f * dyc - 0.25f;
			const float4 s = bl_fetch(small, sw, sh, fx, fy);
			const float w = tw[(ky + 1) * 3 + (kx + 1)];
			acc.x += s.x*w; acc.y += s.y*w; acc.z += s.z*w; acc.w += s.w*w;
		}
	}

	const float4 L = level[(size_t)y * dw + x];
	dst[(size_t)y * dw + x] = make_float4(acc.x + L.x*wk, acc.y + L.y*wk,
										  acc.z + L.z*wk, acc.w + L.w*wk);
}

/* --------------------------------------------------------------- composite */

__device__ __forceinline__ float
bl_tone_scalar (float x, int mode, float white)
{
	if (x <= 0.0f) return 0.0f;
	if (mode == BL_TONE_REINHARD_K) return x / (1.0f + x / white);
	if (mode == BL_TONE_ACES_K) {
		const float a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
		return fmaxf((x * (a*x + b)) / (x * (c*x + d) + e), 0.0f);
	}
	return x;
}

/* Clamped bilinear fetch from the cropped glow buffer (matches CPU BL_Fetch). */
__device__ __forceinline__ float4
bl_glow_bilinear (const float4 *glow, int CW, int CH, float fx, float fy)
{
	fx = fminf(fmaxf(fx, 0.0f), (float)(CW - 1));
	fy = fminf(fmaxf(fy, 0.0f), (float)(CH - 1));
	const int x0 = (int)fx, y0 = (int)fy;
	const int x1 = (x0 + 1 < CW) ? x0 + 1 : x0;
	const int y1 = (y0 + 1 < CH) ? y0 + 1 : y0;
	const float tx = fx - x0, ty = fy - y0;
	const float4 p00 = glow[(size_t)y0*CW + x0], p01 = glow[(size_t)y0*CW + x1];
	const float4 p10 = glow[(size_t)y1*CW + x0], p11 = glow[(size_t)y1*CW + x1];
	float4 a0 = make_float4(p00.x+(p01.x-p00.x)*tx, p00.y+(p01.y-p00.y)*tx,
							p00.z+(p01.z-p00.z)*tx, p00.w+(p01.w-p00.w)*tx);
	float4 a1 = make_float4(p10.x+(p11.x-p10.x)*tx, p10.y+(p11.y-p10.y)*tx,
							p10.z+(p11.z-p10.z)*tx, p10.w+(p11.w-p10.w)*tx);
	return make_float4(a0.x+(a1.x-a0.x)*ty, a0.y+(a1.y-a0.y)*ty,
					   a0.z+(a1.z-a0.z)*ty, a0.w+(a1.w-a0.w)*ty);
}

__global__ void
bl_composite (const float4 *src, const float4 *glow, float4 *dst, BLGpuParams p)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= p.W || y >= p.H) return;

	const float4 s = bl_read_in(src, p, x - p.offX, y - p.offY);

	const int cx = x + p.pad - p.wx0;
	const int cy = y + p.pad - p.wy0;
	if (cx < 0 || cy < 0 || cx >= p.CW || cy >= p.CH) {
		// Beyond any reach: no glow. Composite shows the source; the isolate
		// views show empty glow (transparent for Glow Only, black for Over Black).
		if (p.view == BL_VIEW_COMPOSITE_K)
			dst[(size_t)y * p.dstPitch + x] = make_float4(s.z, s.y, s.x, s.w);
		else
			dst[(size_t)y * p.dstPitch + x] = make_float4(
				0.0f, 0.0f, 0.0f, (p.view == BL_VIEW_GLOWBLACK_K) ? 1.0f : 0.0f);
		return;
	}

	const float a = s.w;

	// Chromatic aberration: sample the glow's R/B at radially scaled positions
	// about the centre (crop-independent displacement in OUTPUT coords). G stays
	// exact so amount 0 is a no-op; alpha from the centre sample.
	float4 g;
	if (p.caAmt != 0.0f) {
		const float vx = (float)x - p.caCX, vy = (float)y - p.caCY;
		const float4 gc = glow[(size_t)cy * p.CW + cx];
		const float4 sR = bl_glow_bilinear(glow, p.CW, p.CH,
							cx + vx*(-p.caAmt), cy + vy*(-p.caAmt));
		const float4 sB = bl_glow_bilinear(glow, p.CW, p.CH,
							cx + vx*( p.caAmt), cy + vy*( p.caAmt));
		g = make_float4(sR.x, gc.y, sB.z, gc.w);
	} else {
		g = glow[(size_t)cy * p.CW + cx];
	}
	const float  k = p.scatter * p.gain;

	// Tint = colorize the glow toward tintN (luma-normalized on the host),
	// preserving the glow's own luminance so 100% red -> red halo of the same
	// brightness (a subtractive filter would give black). amt 0 = untouched.
	float3 gT = make_float3(g.x, g.y, g.z);
	if (p.tintAmt > 0.0f) {
		const float gL = 0.2126f*g.x + 0.7152f*g.y + 0.0722f*g.z;
		gT.x = g.x*(1.0f-p.tintAmt) + (p.tintR*gL)*p.tintAmt;
		gT.y = g.y*(1.0f-p.tintAmt) + (p.tintG*gL)*p.tintAmt;
		gT.z = g.z*(1.0f-p.tintAmt) + (p.tintB*gL)*p.tintAmt;
	}

	// Direct beam weight: 1-scatter for Composite, 0 for the isolate views.
	const float srcW = (p.view == BL_VIEW_COMPOSITE_K) ? (1.0f - p.scatter) : 0.0f;

	// GLOW (emissive halo): premultiply-in-LINEAR + tone map, so it stays correct
	// over black. Straighten by its own coverage, shape it, re-premultiply linear.
	// glowA is COVERAGE (scatter only, NEVER gain - gain is brightness, not opacity).
	const float glowA = fminf(fmaxf(g.w * p.scatter, 0.0f), 1.0f);
	float3 glowDisp = make_float3(0.0f, 0.0f, 0.0f);
	if (glowA > 1e-4f) {
		const float ig = 1.0f / glowA;
		float3 st = make_float3(gT.x*k*ig, gT.y*k*ig, gT.z*k*ig);

		// HUE-PRESERVING tone map: compress LUMINANCE, keep the chroma ratio.
		if (p.tone != BL_TONE_NONE_K) {
			const float L  = st.x*0.2126f + st.y*0.7152f + st.z*0.0722f;
			const float Lt = bl_tone_scalar(L, p.tone, 4.0f);
			const float ratio = (L > 1e-6f) ? (Lt / L) : 0.0f;
			const float hp = p.huePreserve;
			float3 pc = make_float3(bl_tone_scalar(st.x, p.tone, 4.0f),
									bl_tone_scalar(st.y, p.tone, 4.0f),
									bl_tone_scalar(st.z, p.tone, 4.0f));
			st.x = fminf(fmaxf(hp*(st.x*ratio) + (1.0f-hp)*pc.x, 0.0f), 1.0f);
			st.y = fminf(fmaxf(hp*(st.y*ratio) + (1.0f-hp)*pc.y, 0.0f), 1.0f);
			st.z = fminf(fmaxf(hp*(st.z*ratio) + (1.0f-hp)*pc.z, 0.0f), 1.0f);
		}
		if (p.saturation != 1.0f) {
			const float L = st.x*0.2126f + st.y*0.7152f + st.z*0.0722f;
			st.x = fmaxf(L + (st.x - L)*p.saturation, 0.0f);
			st.y = fmaxf(L + (st.y - L)*p.saturation, 0.0f);
			st.z = fmaxf(L + (st.z - L)*p.saturation, 0.0f);
		}
		glowDisp = make_float3(bl_linear_to_srgb(st.x*glowA),
							   bl_linear_to_srgb(st.y*glowA),
							   bl_linear_to_srgb(st.z*glowA));
	}

	// ART (direct beam): dim the LIGHT by srcW in LINEAR (the beam carries
	// 1-scatter of the light), but premultiply in GAMMA (encode THEN * a) - AE's
	// native convention, so the source AA edge is preserved and legal (rgb <= a).
	// Only the GLOW needs linear premult; the art keeps gamma -> no staircase, and
	// dimming in linear (not gamma) leaves the base brightness unchanged.
	const float ia = p.inputStraight ? 1.0f : ((a > 1e-4f) ? (1.0f / a) : 0.0f);
	const float artR = bl_linear_to_srgb(bl_srgb_to_linear(bl_straight01(s.x*ia))*srcW) * a;
	const float artG = bl_linear_to_srgb(bl_srgb_to_linear(bl_straight01(s.y*ia))*srcW) * a;
	const float artB = bl_linear_to_srgb(bl_srgb_to_linear(bl_straight01(s.z*ia))*srcW) * a;
	float3 outRGB = make_float3(fminf(artR + glowDisp.x, 1.0f),
								fminf(artG + glowDisp.y, 1.0f),
								fminf(artB + glowDisp.z, 1.0f));

	// Coverage = union of art and glow coverage.
	const float outA = (p.view == BL_VIEW_COMPOSITE_K) ? a  // emissive: max(a,mx) below
					 : (p.view == BL_VIEW_GLOW_K)       ? glowA
					 :                                    1.0f;

	// One legality raise (rgb <= alpha). The art is already legal, so this now
	// bites ONLY on glow-dominated pixels: halo stays correct over black, art
	// edge untouched.
	const float mx = fmaxf(fmaxf(outRGB.x, outRGB.y), outRGB.z);
	const float finalA = fminf(fmaxf(outA, mx), 1.0f);

	dst[(size_t)y * p.dstPitch + x] = make_float4(outRGB.z, outRGB.y, outRGB.x, finalA); // RGBA -> BGRA
}

/* Straight copy-through, used when the layer has no coverage at all. */
__global__ void
bl_copythrough (const float4 *src, float4 *dst, BLGpuParams p)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= p.W || y >= p.H) return;
	// No glow anywhere: Composite copies the input; the isolate views show
	// empty glow (transparent for Glow Only, black for Glow Over Black).
	if (p.view == BL_VIEW_COMPOSITE_K) {
		const float4 s = bl_read_in(src, p, x - p.offX, y - p.offY);
		dst[(size_t)y * p.dstPitch + x] = make_float4(s.z, s.y, s.x, s.w);
	} else {
		dst[(size_t)y * p.dstPitch + x] = make_float4(
			0.0f, 0.0f, 0.0f, (p.view == BL_VIEW_GLOWBLACK_K) ? 1.0f : 0.0f);
	}
}

/* ---- streaks: one log-doubling pass B(p) = A(p) + wk*A(p - off*d) --------- */
__global__ void
bl_streak_step (const float4 *A, float4 *B, int CW, int CH,
				float ox, float oy, float wk)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;
	const float4 a = A[(size_t)y * CW + x];
	const float4 s = bl_glow_bilinear(A, CW, CH, (float)x - ox, (float)y - oy);
	B[(size_t)y * CW + x] = make_float4(a.x + wk*s.x, a.y + wk*s.y,
										a.z + wk*s.z, a.w + wk*s.w);
}

/* One PERPENDICULAR 3-tap [0.25,0.5,0.25] pass: the lateral fan of the ray.
   Mirrors the spread block in BL_StreakSweep (Bloom.cpp) - see the long note
   there. Perpendicular only, so tip sharpness is untouched; and it is the only
   thing mixing across the sweep direction, which is what stops an axis-aligned
   sweep stamping per-row structure down the ray's whole length. */
__global__ void
bl_streak_spread (const float4 *A, float4 *B, int CW, int CH, float qx, float qy)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;
	const float4 a  = A[(size_t)y * CW + x];
	const float4 sp = bl_glow_bilinear(A, CW, CH, (float)x + qx, (float)y + qy);
	const float4 sm = bl_glow_bilinear(A, CW, CH, (float)x - qx, (float)y - qy);
	B[(size_t)y * CW + x] = make_float4(0.5f*a.x + 0.25f*(sp.x + sm.x),
										0.5f*a.y + 0.25f*(sp.y + sm.y),
										0.5f*a.z + 0.25f*(sp.z + sm.z),
										0.5f*a.w + 0.25f*(sp.w + sm.w));
}

/* Add scale*(full - core) into the glow accumulator (the streak TAIL). */
__global__ void
bl_streak_add (float4 *acc, const float4 *full, const float4 *core,
			   int CW, int CH, float scale)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;
	const size_t i = (size_t)y * CW + x;
	acc[i].x += scale * (full[i].x - core[i].x);
	acc[i].y += scale * (full[i].y - core[i].y);
	acc[i].z += scale * (full[i].z - core[i].z);
	acc[i].w += scale * (full[i].w - core[i].w);
}

/* Streak source = extract gated by a luma soft-knee (matches the CPU). */
__global__ void
bl_streak_gate (const float4 *src, float4 *dst, int CW, int CH, float thr)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;
	const size_t i = (size_t)y * CW + x;
	const float4 s = src[i];
	const float lum = s.x*0.2126f + s.y*0.7152f + s.z*0.0722f;
	float g = 1.0f;
	if (thr > 0.0f) {
		const float k = (thr*0.5f > 1e-4f) ? thr*0.5f : 1e-4f;
		float soft = lum - thr + k;
		soft = fminf(fmaxf(soft, 0.0f), 2.0f*k);
		soft = soft*soft / (4.0f*k);
		const float hard = lum - thr;
		float gg = fmaxf((soft > hard) ? soft : hard, 0.0f);
		g = (lum > 1e-6f) ? fminf(gg / lum, 1.0f) : 0.0f;
	}
	dst[i] = make_float4(s.x*g, s.y*g, s.z*g, s.w*g);
}

/* One separable [1,4,6,4,1] pass. Runs ONCE on the full-res streak source so an
   axis-aligned integer-exact sweep doesn't print persistent per-column stripes
   (the sweep integrates each column independently, so any column step in the
   source lasts the whole streak). horiz != 0 blurs in x, else in y. */
__global__ void
bl_streak_blur (const float4 *src, float4 *dst, int CW, int CH, int horiz)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;
	const float kf[5] = { 1.f/16, 4.f/16, 6.f/16, 4.f/16, 1.f/16 };
	float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
	#pragma unroll
	for (int t = -2; t <= 2; t++) {
		int u = horiz ? x + t : x;
		int v = horiz ? y : y + t;
		if (u < 0) u = 0; else if (u >= CW) u = CW - 1;
		if (v < 0) v = 0; else if (v >= CH) v = CH - 1;
		const float4 p = src[(size_t)v * CW + u];
		const float w = kf[t + 2];
		a.x += p.x*w; a.y += p.y*w; a.z += p.z*w; a.w += p.w*w;
	}
	dst[(size_t)y * CW + x] = a;
}

/* ======================================================================= host
   Orchestrates the whole pipeline. `wgt` is the host-computed level weight
   array (normalized, so the PSF integrates to 1) and `nLev` its length; the
   host also decides how many levels to build, so the GPU and CPU paths share
   exactly the same radius -> levels mapping.
   ======================================================================= */

#define BL_MAX_LEV_K	14
#define BL_TRY(call)	do { if ((call) != cudaSuccess) goto cleanup; } while (0)

void
BL_Bloom_CUDA (const float *srcIn, float *dstIn, BLGpuParams p,
			   const float *wgt, int nLev)
{
	const float4 *src = (const float4*)srcIn;
	float4       *dst = (float4*)dstIn;

	float4 *lv[BL_MAX_LEV_K];
	int     lw[BL_MAX_LEV_K], lh[BL_MAX_LEV_K];
	float4 *accA = 0, *accB = 0;
	float4 *strkA = 0, *strkB = 0, *strkGated = 0;
	float4 *strkSrcHalf = 0, *strkAccHalf = 0, *strkTmp = 0;
	int    *devBox = 0;
	for (int i = 0; i < BL_MAX_LEV_K; i++) lv[i] = 0;

	// Declare launch configs BEFORE any goto (C++ forbids jumping across a
	// non-trivial initializer - the gotcha from the Buildable Stroke port).
	dim3 blk(16, 16);
	dim3 gridOut((p.W + 15) / 16, (p.H + 15) / 16);
	dim3 gridIn((p.inW + 15) / 16, (p.inH + 15) / 16);
	dim3 gridCrop;
	int levels = 0;

	// --- bbox -> work rect (crop, for perf and memory) --------------------
	{
		int hostBox[5] = { p.inW, p.inH, -1, -1, 0 };
		BL_TRY(cudaMalloc(&devBox, 5 * sizeof(int)));
		BL_TRY(cudaMemcpy(devBox, hostBox, 5 * sizeof(int), cudaMemcpyHostToDevice));
		bl_bbox<<<gridIn, blk>>>(src, p, devBox);
		BL_TRY(cudaMemcpy(hostBox, devBox, 5 * sizeof(int), cudaMemcpyDeviceToHost));
		p.inputStraight = hostBox[4];		// resolved before any kernel uses it

		if (hostBox[2] < 0) {			// nothing with coverage -> copy through
			bl_copythrough<<<gridOut, blk>>>(src, dst, p);
			goto cleanup;
		}

		const int PW = p.W + 2 * p.pad;
		const int PH = p.H + 2 * p.pad;
		int wx0 = hostBox[0] + p.offX;					// = minPadded - pad
		int wy0 = hostBox[1] + p.offY;
		int wx1 = hostBox[2] + p.offX + 2 * p.pad;		// = maxPadded + pad
		int wy1 = hostBox[3] + p.offY + 2 * p.pad;
		if (wx0 < 0) wx0 = 0;
		if (wy0 < 0) wy0 = 0;
		if (wx1 > PW - 1) wx1 = PW - 1;
		if (wy1 > PH - 1) wy1 = PH - 1;
		if (wx1 < wx0 || wy1 < wy0) {
			bl_copythrough<<<gridOut, blk>>>(src, dst, p);
			goto cleanup;
		}
		p.wx0 = wx0;  p.wy0 = wy0;
		p.CW  = wx1 - wx0 + 1;
		p.CH  = wy1 - wy0 + 1;
	}

	gridCrop = dim3((p.CW + 15) / 16, (p.CH + 15) / 16);

	// --- level 0 = the extract --------------------------------------------
	lw[0] = p.CW; lh[0] = p.CH;
	BL_TRY(cudaMalloc(&lv[0], (size_t)lw[0] * lh[0] * sizeof(float4)));
	bl_extract<<<gridCrop, blk>>>(src, lv[0], p);
	levels = 1;

	// --- downsample chain --------------------------------------------------
	for (int k = 1; k < nLev && k < BL_MAX_LEV_K; k++) {
		const int nw = lw[k-1] / 2, nh = lh[k-1] / 2;
		if (nw < 2 || nh < 2) break;
		lw[k] = nw; lh[k] = nh;
		BL_TRY(cudaMalloc(&lv[k], (size_t)nw * nh * sizeof(float4)));
		dim3 g((nw + 15) / 16, (nh + 15) / 16);
		bl_down13<<<g, blk>>>(lv[k-1], lw[k-1], lh[k-1], lv[k], nw, nh,
							  (p.karis && k == 1) ? 1 : 0);
		levels = k + 1;
	}

	// --- progressive upsample + accumulate ---------------------------------
	BL_TRY(cudaMalloc(&accA, (size_t)p.CW * p.CH * sizeof(float4)));
	BL_TRY(cudaMalloc(&accB, (size_t)p.CW * p.CH * sizeof(float4)));
	{
		// `wgt` arrives UNNORMALIZED and we normalize over the levels actually
		// built - the chain can stop early when a level would get too small, and
		// normalizing over a level count we never built would dim the glow.
		float wn[BL_MAX_LEV_K];
		float sum = 0.0f;
		for (int k = 0; k < levels; k++) { wn[k] = (k < nLev) ? wgt[k] : 0.0f; sum += wn[k]; }
		if (sum > 1e-8f) { for (int k = 0; k < levels; k++) wn[k] /= sum; }
		else             { wn[0] = 1.0f; }

		const int top = levels - 1;
		{
			dim3 g((lw[top] + 15) / 16, (lh[top] + 15) / 16);
			bl_scale<<<g, blk>>>(lv[top], accA, lw[top], lh[top], wn[top]);
		}
		float4 *cur = accA, *nxt = accB;
		for (int k = top - 1; k >= 0; k--) {
			dim3 g((lw[k] + 15) / 16, (lh[k] + 15) / 16);
			bl_up_tent<<<g, blk>>>(cur, lw[k+1], lh[k+1], lv[k], nxt,
								   lw[k], lh[k], wn[k]);
			float4 *t = cur; cur = nxt; nxt = t;
		}

		// --- streaks (HALF-RESOLUTION): directional geometric-weighted sweeps of
		// the extract, added into the glow. Streaks are smooth wide rays, so we
		// sweep a 2x-downsampled source and upsample the result (~4x cheaper,
		// visually identical). A half-res step spans 2 full-res px, so Length and
		// decay halve. Same alignment-correct down/up as the pyramid (bl_down13 /
		// bl_up_tent) so the streak doesn't shift.
		if (p.streakInten > 0.0f && p.streakCount > 0 && p.streakLen >= 1.0f) {
			const int HW = p.CW / 2, HH = p.CH / 2;
			const bool useHalf = (p.streakQuality != BL_SQ_FULL_K) && HW >= 2 && HH >= 2;
			if (useHalf) {
				const float Lh    = p.streakLen * 0.5f;
				const float decay = expf(logf(BL_STREAK_EPS_K) / (Lh > 1.0f ? Lh : 1.0f));
				const float norm  = (1.0f - decay) / decay;
				const float scale = p.streakInten * norm;
				int axes = (p.streakCount < BL_STREAK_MAX_AXES_K)
						 ? p.streakCount : BL_STREAK_MAX_AXES_K;
				const size_t cropN = (size_t)p.CW * p.CH * sizeof(float4);
				const size_t halfN = (size_t)HW * HH * sizeof(float4);
				dim3 gHalf((HW + 15) / 16, (HH + 15) / 16);

				// Full-res source, gated by threshold if set.
				const float4 *srcFull = lv[0];
				if (p.streakThr > 0.0f) {
					BL_TRY(cudaMalloc(&strkGated, cropN));
					bl_streak_gate<<<gridCrop, blk>>>(lv[0], strkGated, p.CW, p.CH, p.streakThr);
					srcFull = strkGated;
				}
				// Downsample to half; sweep there; accumulate tails into strkAccHalf.
				BL_TRY(cudaMalloc(&strkSrcHalf, halfN));
				bl_down13<<<gHalf, blk>>>(srcFull, p.CW, p.CH, strkSrcHalf, HW, HH, 0);
				BL_TRY(cudaMalloc(&strkAccHalf, halfN));
				BL_TRY(cudaMemset(strkAccHalf, 0, halfN));
				BL_TRY(cudaMalloc(&strkA, halfN));
				BL_TRY(cudaMalloc(&strkB, halfN));
				// The sweep integrates each HALF-RES column independently too, so the
				// same per-column stripes appear here and the tent upsample WIDENS them
				// to ~2px (invisible over black, obvious over a light bg). Give the
				// half-res source its ~1px lateral width: srcHalf -> A -> B -> srcHalf.
				bl_streak_blur<<<gHalf, blk>>>(strkSrcHalf, strkA, HW, HH, 1);
				bl_streak_blur<<<gHalf, blk>>>(strkA, strkB, HW, HH, 0);
				BL_TRY(cudaMemcpy(strkSrcHalf, strkB, halfN, cudaMemcpyDeviceToDevice));
				for (int ax = 0; ax < axes; ax++) {
					const float ang = (p.streakAngle + ax * 180.0f / axes) * 0.017453293f;
					const float cd = cosf(ang), sd = sinf(ang);
					for (int dir = 0; dir < 2; dir++) {
						const float dx = dir ? -cd : cd;
						const float dy = dir ? -sd : sd;
						BL_TRY(cudaMemcpy(strkA, strkSrcHalf, halfN, cudaMemcpyDeviceToDevice));
						float4 *A = strkA, *B = strkB;
						float off = 1.0f;
						while (off <= Lh) {
							const float wk = powf(decay, off);
							bl_streak_step<<<gHalf, blk>>>(A, B, HW, HH,
														   dx * off, dy * off, wk);
							{ float4 *t = A; A = B; B = t; }
							const float sw = p.streakSpread * off;
							if (sw >= 0.25f) {
								bl_streak_spread<<<gHalf, blk>>>(A, B, HW, HH,
																 -dy * sw, dx * sw);
								float4 *t = A; A = B; B = t;
							}
							off *= 2.0f;
						}
						bl_streak_add<<<gHalf, blk>>>(strkAccHalf, A, strkSrcHalf, HW, HH, scale);
					}
				}
				// Upsample half-res streaks to full and add into the glow:
				// strkTmp = tentUp(strkAccHalf) + 1.0*cur.
				BL_TRY(cudaMalloc(&strkTmp, cropN));
				bl_up_tent<<<gridCrop, blk>>>(strkAccHalf, HW, HH, cur, strkTmp,
											  p.CW, p.CH, 1.0f);
				cur = strkTmp;
			} else {
				// FULL-RESOLUTION: sweep at full crop -> crisp tips, arms cross
				// exactly. ~4x slower; for hero frames.
				const float L     = p.streakLen;
				const float decay = expf(logf(BL_STREAK_EPS_K) / (L > 1.0f ? L : 1.0f));
				const float norm  = (1.0f - decay) / decay;
				const float scale = p.streakInten * norm;
				int axes = (p.streakCount < BL_STREAK_MAX_AXES_K)
						 ? p.streakCount : BL_STREAK_MAX_AXES_K;
				const size_t cropN = (size_t)p.CW * p.CH * sizeof(float4);
				const float4 *srcFull = lv[0];
				if (p.streakThr > 0.0f) {
					BL_TRY(cudaMalloc(&strkGated, cropN));
					bl_streak_gate<<<gridCrop, blk>>>(lv[0], strkGated, p.CW, p.CH, p.streakThr);
					srcFull = strkGated;
				}
				BL_TRY(cudaMalloc(&strkA, cropN));
				BL_TRY(cudaMalloc(&strkB, cropN));
				// Give the source ~1px lateral width (separable [1,4,6,4,1]) so an
				// axis-aligned integer-exact sweep does not print per-column stripes.
				// srcFull -> strkA (horiz) -> strkTmp (vert); sweep strkTmp.
				BL_TRY(cudaMalloc(&strkTmp, cropN));
				bl_streak_blur<<<gridCrop, blk>>>(srcFull, strkA,  p.CW, p.CH, 1);
				bl_streak_blur<<<gridCrop, blk>>>(strkA,  strkTmp, p.CW, p.CH, 0);
				const float4 *swSrc = strkTmp;
				for (int ax = 0; ax < axes; ax++) {
					const float ang = (p.streakAngle + ax * 180.0f / axes) * 0.017453293f;
					const float cd = cosf(ang), sd = sinf(ang);
					for (int dir = 0; dir < 2; dir++) {
						const float dx = dir ? -cd : cd;
						const float dy = dir ? -sd : sd;
						BL_TRY(cudaMemcpy(strkA, swSrc, cropN, cudaMemcpyDeviceToDevice));
						float4 *A = strkA, *B = strkB;
						float off = 1.0f;
						while (off <= L) {
							const float wk = powf(decay, off);
							bl_streak_step<<<gridCrop, blk>>>(A, B, p.CW, p.CH,
															  dx * off, dy * off, wk);
							{ float4 *t = A; A = B; B = t; }
							const float sw = p.streakSpread * off;
							if (sw >= 0.25f) {
								bl_streak_spread<<<gridCrop, blk>>>(A, B, p.CW, p.CH,
																	-dy * sw, dx * sw);
								float4 *t = A; A = B; B = t;
							}
							off *= 2.0f;
						}
						bl_streak_add<<<gridCrop, blk>>>(cur, A, swSrc, p.CW, p.CH, scale);
					}
				}
			}
		}

		// --- composite -----------------------------------------------------
		bl_composite<<<gridOut, blk>>>(src, cur, dst, p);
	}

cleanup:
	if (devBox) cudaFree(devBox);
	if (accA)   cudaFree(accA);
	if (accB)   cudaFree(accB);
	if (strkA)  cudaFree(strkA);
	if (strkB)  cudaFree(strkB);
	if (strkGated) cudaFree(strkGated);
	if (strkSrcHalf) cudaFree(strkSrcHalf);
	if (strkAccHalf) cudaFree(strkAccHalf);
	if (strkTmp) cudaFree(strkTmp);
	for (int i = 0; i < BL_MAX_LEV_K; i++) if (lv[i]) cudaFree(lv[i]);
}
