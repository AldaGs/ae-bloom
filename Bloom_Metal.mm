/*	Bloom_Metal.mm

	macOS Metal host for Bloom. Objective-C++ so all the Metal (Obj-C) API stays
	out of Bloom.cpp, which remains pure C++. Three C entry points are exposed:

	  BL_MetalCompile  - GPU_DEVICE_SETUP: compile the MSL library once per device
	                     and build one pipeline state per kernel.
	  BL_MetalDispose  - GPU_DEVICE_SETDOWN: release the pipeline states.
	  BL_Bloom_Metal   - SMART_RENDER_GPU: the whole pyramid + streaks + composite,
	                     a faithful mirror of BL_Bloom_CUDA (Bloom_Kernel.cu).

	Design choices that keep this a 1:1 port of the CUDA path:
	  - one SERIAL compute encoder runs the whole main pipeline; Metal's default
	    hazard tracking inserts the barriers between dependent dispatches (same
	    guarantee ProcAmp relies on for invert->procamp).
	  - CUDA cudaMemset / cudaMemcpy(D2D) become the bl_clear / bl_copybuf kernels
	    so nothing has to leave the compute encoder for a blit pass.
	  - the bbox pass needs a CPU readback (crop rect + straight/premult flag), so
	    it runs in its own command buffer that we waitUntilCompleted, exactly like
	    the cudaMemcpy(DeviceToHost) in the CUDA host.

	NOTE (storage mode): the 5-int bbox buffer uses Shared storage, coherent on
	Apple Silicon. On an Intel Mac with a discrete GPU this should be Managed with
	a synchronizeResource blit before readback; revisit if we ever target those.
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "Bloom_GpuParams.h"
#include "Bloom_Kernel_Metal.h"

#define BL_MAX_LEV_K			14
#define BL_STREAK_MAX_AXES_K	8
#define BL_STREAK_EPS_K			0.02f
#define BL_SQ_FULL_K			2

struct BLMetalGPUData {
	id<MTLComputePipelineState> bbox, extract, down13, scale, up_tent, composite, copythrough;
	id<MTLComputePipelineState> streak_step, streak_spread, streak_add, streak_gate, streak_blur;
	id<MTLComputePipelineState> clear, copybuf;
};

/* ------------------------------------------------------- compile / dispose */

extern "C" bool
BL_MetalCompile (void *devicePV, void **outData, char *errBuf, int errLen)
{
	@autoreleasepool {
		id<MTLDevice> device = (id<MTLDevice>)devicePV;
		NSError *error = nil;
		NSString *source = [NSString stringWithUTF8String:kBloomKernelMetalString];
		id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
		if (!lib) {
			if (errBuf && errLen > 0) {
				const char *m = error ? [[error localizedDescription] UTF8String]
									   : "unknown Metal compile error";
				strncpy(errBuf, m ? m : "nil", errLen - 1);
				errBuf[errLen - 1] = 0;
			}
			return false;
		}

		BLMetalGPUData *d = (BLMetalGPUData*)calloc(1, sizeof(BLMetalGPUData));
		bool ok = true;

		// Build one pipeline state per kernel. Any missing function or PSO
		// failure aborts and reports through errBuf.
		auto mk = [&](id<MTLComputePipelineState> __strong *slot, const char *name) {
			if (!ok) return;
			id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
			if (!fn) {
				if (errBuf && errLen > 0) { snprintf(errBuf, errLen, "missing kernel: %s", name); }
				ok = false; return;
			}
			NSError *e = nil;
			*slot = [device newComputePipelineStateWithFunction:fn error:&e];
			[fn release];
			if (!*slot) {
				if (errBuf && errLen > 0) {
					const char *m = e ? [[e localizedDescription] UTF8String] : "pso failed";
					snprintf(errBuf, errLen, "pso %s: %s", name, m ? m : "nil");
				}
				ok = false;
			}
		};

		mk(&d->bbox,          "bl_bbox");
		mk(&d->extract,       "bl_extract");
		mk(&d->down13,        "bl_down13");
		mk(&d->scale,         "bl_scale");
		mk(&d->up_tent,       "bl_up_tent");
		mk(&d->composite,     "bl_composite");
		mk(&d->copythrough,   "bl_copythrough");
		mk(&d->streak_step,   "bl_streak_step");
		mk(&d->streak_spread, "bl_streak_spread");
		mk(&d->streak_add,    "bl_streak_add");
		mk(&d->streak_gate,   "bl_streak_gate");
		mk(&d->streak_blur,   "bl_streak_blur");
		mk(&d->clear,         "bl_clear");
		mk(&d->copybuf,       "bl_copybuf");

		[lib release];

		if (!ok) { BL_MetalDispose(d); return false; }
		*outData = d;
		return true;
	}
}

extern "C" void
BL_MetalDispose (void *dataPV)
{
	BLMetalGPUData *d = (BLMetalGPUData*)dataPV;
	if (!d) return;
	[d->bbox release];          [d->extract release];    [d->down13 release];
	[d->scale release];         [d->up_tent release];    [d->composite release];
	[d->copythrough release];   [d->streak_step release];[d->streak_spread release];
	[d->streak_add release];    [d->streak_gate release];[d->streak_blur release];
	[d->clear release];         [d->copybuf release];
	free(d);
}

/* -------------------------------------------------------------- the render */

extern "C" bool
BL_Bloom_Metal (void *devicePV, void *queuePV, void *dataPV,
				void *srcMemPV, void *dstMemPV, BLGpuParams p,
				const float *wgt, int nLev)
{
	@autoreleasepool {
		id<MTLDevice>       device = (id<MTLDevice>)devicePV;
		id<MTLCommandQueue> queue  = (id<MTLCommandQueue>)queuePV;
		BLMetalGPUData     *d      = (BLMetalGPUData*)dataPV;
		id<MTLBuffer>       src    = (id<MTLBuffer>)srcMemPV;
		id<MTLBuffer>       dst    = (id<MTLBuffer>)dstMemPV;

		const MTLSize TG = MTLSizeMake(16, 16, 1);
		std::vector<id<MTLBuffer>> bufs;					// everything we allocate
		auto mkbuf = [&](size_t bytes) -> id<MTLBuffer> {
			id<MTLBuffer> b = [device newBufferWithLength:bytes
											options:MTLResourceStorageModePrivate];
			bufs.push_back(b);
			return b;
		};
		auto freeAll = [&]() { for (id<MTLBuffer> b : bufs) [b release]; bufs.clear(); };

		// --- bbox -> work rect (own command buffer, must read back) -----------
		int hostBox[5] = { p.inW, p.inH, -1, -1, 0 };
		id<MTLBuffer> boxBuf = [device newBufferWithLength:5*sizeof(int)
											options:MTLResourceStorageModeShared];
		memcpy([boxBuf contents], hostBox, sizeof(hostBox));
		{
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:d->bbox];
			[e setBuffer:src    offset:0 atIndex:0];
			[e setBytes:&p length:sizeof(p) atIndex:1];
			[e setBuffer:boxBuf offset:0 atIndex:2];
			[e dispatchThreadgroups:MTLSizeMake((p.inW+15)/16, (p.inH+15)/16, 1)
				threadsPerThreadgroup:TG];
			[e endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			memcpy(hostBox, [boxBuf contents], sizeof(hostBox));
		}
		[boxBuf release];
		p.inputStraight = hostBox[4];

		// Nothing with coverage, or an empty work rect -> copy through.
		const int PW = p.W + 2*p.pad;
		const int PH = p.H + 2*p.pad;
		bool empty = (hostBox[2] < 0);
		if (!empty) {
			int wx0 = hostBox[0] + p.offX;
			int wy0 = hostBox[1] + p.offY;
			int wx1 = hostBox[2] + p.offX + 2*p.pad;
			int wy1 = hostBox[3] + p.offY + 2*p.pad;
			if (wx0 < 0) wx0 = 0;
			if (wy0 < 0) wy0 = 0;
			if (wx1 > PW-1) wx1 = PW-1;
			if (wy1 > PH-1) wy1 = PH-1;
			if (wx1 < wx0 || wy1 < wy0) empty = true;
			else { p.wx0 = wx0; p.wy0 = wy0; p.CW = wx1-wx0+1; p.CH = wy1-wy0+1; }
		}
		if (empty) {
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:d->copythrough];
			[e setBuffer:src offset:0 atIndex:0];
			[e setBuffer:dst offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			[e dispatchThreadgroups:MTLSizeMake((p.W+15)/16, (p.H+15)/16, 1)
				threadsPerThreadgroup:TG];
			[e endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			freeAll();
			return true;
		}

		// --- main pipeline: one command buffer, one serial compute encoder ----
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

		auto disp2D = [&](id<MTLComputePipelineState> pso, int w, int h) {
			[e dispatchThreadgroups:MTLSizeMake((w+15)/16, (h+15)/16, 1)
				threadsPerThreadgroup:TG];
		};
		auto disp1D = [&](int n) {
			[e dispatchThreadgroups:MTLSizeMake((n+255)/256, 1, 1)
				threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
		};
		auto clearBuf = [&](id<MTLBuffer> b, int n) {
			[e setComputePipelineState:d->clear];
			[e setBuffer:b offset:0 atIndex:0];
			[e setBytes:&n length:sizeof(int) atIndex:1];
			disp1D(n);
		};
		auto copyBuf = [&](id<MTLBuffer> s, id<MTLBuffer> dstb, int n) {
			[e setComputePipelineState:d->copybuf];
			[e setBuffer:s    offset:0 atIndex:0];
			[e setBuffer:dstb offset:0 atIndex:1];
			[e setBytes:&n length:sizeof(int) atIndex:2];
			disp1D(n);
		};
		auto down13 = [&](id<MTLBuffer> s, int sw, int sh,
						  id<MTLBuffer> dstb, int dw, int dh, int karis) {
			[e setComputePipelineState:d->down13];
			[e setBuffer:s offset:0 atIndex:0];
			[e setBytes:&sw length:sizeof(int) atIndex:1];
			[e setBytes:&sh length:sizeof(int) atIndex:2];
			[e setBuffer:dstb offset:0 atIndex:3];
			[e setBytes:&dw length:sizeof(int) atIndex:4];
			[e setBytes:&dh length:sizeof(int) atIndex:5];
			[e setBytes:&karis length:sizeof(int) atIndex:6];
			disp2D(d->down13, dw, dh);
		};
		auto upTent = [&](id<MTLBuffer> small, int sw, int sh, id<MTLBuffer> level,
						  id<MTLBuffer> dstb, int dw, int dh, float wk) {
			[e setComputePipelineState:d->up_tent];
			[e setBuffer:small offset:0 atIndex:0];
			[e setBytes:&sw length:sizeof(int) atIndex:1];
			[e setBytes:&sh length:sizeof(int) atIndex:2];
			[e setBuffer:level offset:0 atIndex:3];
			[e setBuffer:dstb  offset:0 atIndex:4];
			[e setBytes:&dw length:sizeof(int) atIndex:5];
			[e setBytes:&dh length:sizeof(int) atIndex:6];
			[e setBytes:&wk length:sizeof(float) atIndex:7];
			disp2D(d->up_tent, dw, dh);
		};
		auto streakStep = [&](id<MTLBuffer> A, id<MTLBuffer> B, int CW, int CH,
							  float ox, float oy, float wk) {
			[e setComputePipelineState:d->streak_step];
			[e setBuffer:A offset:0 atIndex:0];
			[e setBuffer:B offset:0 atIndex:1];
			[e setBytes:&CW length:sizeof(int) atIndex:2];
			[e setBytes:&CH length:sizeof(int) atIndex:3];
			[e setBytes:&ox length:sizeof(float) atIndex:4];
			[e setBytes:&oy length:sizeof(float) atIndex:5];
			[e setBytes:&wk length:sizeof(float) atIndex:6];
			disp2D(d->streak_step, CW, CH);
		};
		auto streakSpread = [&](id<MTLBuffer> A, id<MTLBuffer> B, int CW, int CH,
								float qx, float qy) {
			[e setComputePipelineState:d->streak_spread];
			[e setBuffer:A offset:0 atIndex:0];
			[e setBuffer:B offset:0 atIndex:1];
			[e setBytes:&CW length:sizeof(int) atIndex:2];
			[e setBytes:&CH length:sizeof(int) atIndex:3];
			[e setBytes:&qx length:sizeof(float) atIndex:4];
			[e setBytes:&qy length:sizeof(float) atIndex:5];
			disp2D(d->streak_spread, CW, CH);
		};
		auto streakAdd = [&](id<MTLBuffer> acc, id<MTLBuffer> full, id<MTLBuffer> core,
							 int CW, int CH, float scale) {
			[e setComputePipelineState:d->streak_add];
			[e setBuffer:acc  offset:0 atIndex:0];
			[e setBuffer:full offset:0 atIndex:1];
			[e setBuffer:core offset:0 atIndex:2];
			[e setBytes:&CW length:sizeof(int) atIndex:3];
			[e setBytes:&CH length:sizeof(int) atIndex:4];
			[e setBytes:&scale length:sizeof(float) atIndex:5];
			disp2D(d->streak_add, CW, CH);
		};
		auto streakBlur = [&](id<MTLBuffer> s, id<MTLBuffer> dstb, int CW, int CH, int horiz) {
			[e setComputePipelineState:d->streak_blur];
			[e setBuffer:s    offset:0 atIndex:0];
			[e setBuffer:dstb offset:0 atIndex:1];
			[e setBytes:&CW length:sizeof(int) atIndex:2];
			[e setBytes:&CH length:sizeof(int) atIndex:3];
			[e setBytes:&horiz length:sizeof(int) atIndex:4];
			disp2D(d->streak_blur, CW, CH);
		};

		// level 0 = the extract
		int lw[BL_MAX_LEV_K], lh[BL_MAX_LEV_K];
		id<MTLBuffer> lv[BL_MAX_LEV_K] = { 0 };
		lw[0] = p.CW; lh[0] = p.CH;
		lv[0] = mkbuf((size_t)lw[0]*lh[0]*sizeof(float)*4);
		{
			[e setComputePipelineState:d->extract];
			[e setBuffer:src offset:0 atIndex:0];
			[e setBuffer:lv[0] offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			disp2D(d->extract, p.CW, p.CH);
		}
		int levels = 1;

		// downsample chain
		for (int k = 1; k < nLev && k < BL_MAX_LEV_K; k++) {
			int nw = lw[k-1]/2, nh = lh[k-1]/2;
			if (nw < 2 || nh < 2) break;
			lw[k] = nw; lh[k] = nh;
			lv[k] = mkbuf((size_t)nw*nh*sizeof(float)*4);
			down13(lv[k-1], lw[k-1], lh[k-1], lv[k], nw, nh, (p.karis && k==1) ? 1 : 0);
			levels = k + 1;
		}

		// progressive upsample + accumulate
		id<MTLBuffer> accA = mkbuf((size_t)p.CW*p.CH*sizeof(float)*4);
		id<MTLBuffer> accB = mkbuf((size_t)p.CW*p.CH*sizeof(float)*4);

		float wn[BL_MAX_LEV_K];
		float sum = 0.0f;
		for (int k = 0; k < levels; k++) { wn[k] = (k < nLev) ? wgt[k] : 0.0f; sum += wn[k]; }
		if (sum > 1e-8f) { for (int k = 0; k < levels; k++) wn[k] /= sum; }
		else             { wn[0] = 1.0f; }

		int top = levels - 1;
		{
			[e setComputePipelineState:d->scale];
			[e setBuffer:lv[top] offset:0 atIndex:0];
			[e setBuffer:accA    offset:0 atIndex:1];
			[e setBytes:&lw[top] length:sizeof(int) atIndex:2];
			[e setBytes:&lh[top] length:sizeof(int) atIndex:3];
			[e setBytes:&wn[top] length:sizeof(float) atIndex:4];
			disp2D(d->scale, lw[top], lh[top]);
		}
		id<MTLBuffer> cur = accA, nxt = accB;
		for (int k = top - 1; k >= 0; k--) {
			upTent(cur, lw[k+1], lh[k+1], lv[k], nxt, lw[k], lh[k], wn[k]);
			id<MTLBuffer> t = cur; cur = nxt; nxt = t;
		}

		// --- streaks -----------------------------------------------------------
		if (p.streakInten > 0.0f && p.streakCount > 0 && p.streakLen >= 1.0f) {
			const int HW = p.CW/2, HH = p.CH/2;
			const bool useHalf = (p.streakQuality != BL_SQ_FULL_K) && HW >= 2 && HH >= 2;
			const int cropN = p.CW * p.CH;

			if (useHalf) {
				const float Lh    = p.streakLen * 0.5f;
				const float decay = expf(logf(BL_STREAK_EPS_K) / (Lh > 1.0f ? Lh : 1.0f));
				const float norm  = (1.0f - decay) / decay;
				const float scale = p.streakInten * norm;
				int axes = (p.streakCount < BL_STREAK_MAX_AXES_K) ? p.streakCount : BL_STREAK_MAX_AXES_K;
				const int halfN = HW * HH;

				id<MTLBuffer> srcFull = lv[0];
				if (p.streakThr > 0.0f) {
					id<MTLBuffer> gated = mkbuf((size_t)cropN*sizeof(float)*4);
					[e setComputePipelineState:d->streak_gate];
					[e setBuffer:lv[0] offset:0 atIndex:0];
					[e setBuffer:gated offset:0 atIndex:1];
					[e setBytes:&p.CW length:sizeof(int) atIndex:2];
					[e setBytes:&p.CH length:sizeof(int) atIndex:3];
					[e setBytes:&p.streakThr length:sizeof(float) atIndex:4];
					disp2D(d->streak_gate, p.CW, p.CH);
					srcFull = gated;
				}
				id<MTLBuffer> srcHalf = mkbuf((size_t)halfN*sizeof(float)*4);
				down13(srcFull, p.CW, p.CH, srcHalf, HW, HH, 0);
				id<MTLBuffer> accHalf = mkbuf((size_t)halfN*sizeof(float)*4);
				clearBuf(accHalf, halfN);
				id<MTLBuffer> sA = mkbuf((size_t)halfN*sizeof(float)*4);
				id<MTLBuffer> sB = mkbuf((size_t)halfN*sizeof(float)*4);
				// lateral pre-width: srcHalf -> sA (h) -> sB (v) -> srcHalf
				streakBlur(srcHalf, sA, HW, HH, 1);
				streakBlur(sA, sB, HW, HH, 0);
				copyBuf(sB, srcHalf, halfN);

				for (int ax = 0; ax < axes; ax++) {
					const float ang = (p.streakAngle + ax*180.0f/axes) * 0.017453293f;
					const float cd = cosf(ang), sd = sinf(ang);
					for (int dir = 0; dir < 2; dir++) {
						const float dx = dir ? -cd : cd;
						const float dy = dir ? -sd : sd;
						copyBuf(srcHalf, sA, halfN);
						id<MTLBuffer> A = sA, B = sB;
						float off = 1.0f;
						while (off <= Lh) {
							const float wk = powf(decay, off);
							streakStep(A, B, HW, HH, dx*off, dy*off, wk);
							{ id<MTLBuffer> t = A; A = B; B = t; }
							const float sw = p.streakSpread * off;
							if (sw >= 0.25f) {
								streakSpread(A, B, HW, HH, -dy*sw, dx*sw);
								id<MTLBuffer> t = A; A = B; B = t;
							}
							off *= 2.0f;
						}
						streakAdd(accHalf, A, srcHalf, HW, HH, scale);
					}
				}
				// upsample half-res streaks to full and add into the glow
				id<MTLBuffer> strkTmp = mkbuf((size_t)cropN*sizeof(float)*4);
				upTent(accHalf, HW, HH, cur, strkTmp, p.CW, p.CH, 1.0f);
				cur = strkTmp;
			} else {
				const float L     = p.streakLen;
				const float decay = expf(logf(BL_STREAK_EPS_K) / (L > 1.0f ? L : 1.0f));
				const float norm  = (1.0f - decay) / decay;
				const float scale = p.streakInten * norm;
				int axes = (p.streakCount < BL_STREAK_MAX_AXES_K) ? p.streakCount : BL_STREAK_MAX_AXES_K;

				id<MTLBuffer> srcFull = lv[0];
				if (p.streakThr > 0.0f) {
					id<MTLBuffer> gated = mkbuf((size_t)cropN*sizeof(float)*4);
					[e setComputePipelineState:d->streak_gate];
					[e setBuffer:lv[0] offset:0 atIndex:0];
					[e setBuffer:gated offset:0 atIndex:1];
					[e setBytes:&p.CW length:sizeof(int) atIndex:2];
					[e setBytes:&p.CH length:sizeof(int) atIndex:3];
					[e setBytes:&p.streakThr length:sizeof(float) atIndex:4];
					disp2D(d->streak_gate, p.CW, p.CH);
					srcFull = gated;
				}
				id<MTLBuffer> sA = mkbuf((size_t)cropN*sizeof(float)*4);
				id<MTLBuffer> sB = mkbuf((size_t)cropN*sizeof(float)*4);
				id<MTLBuffer> swSrc = mkbuf((size_t)cropN*sizeof(float)*4);
				streakBlur(srcFull, sA, p.CW, p.CH, 1);
				streakBlur(sA, swSrc, p.CW, p.CH, 0);
				for (int ax = 0; ax < axes; ax++) {
					const float ang = (p.streakAngle + ax*180.0f/axes) * 0.017453293f;
					const float cd = cosf(ang), sd = sinf(ang);
					for (int dir = 0; dir < 2; dir++) {
						const float dx = dir ? -cd : cd;
						const float dy = dir ? -sd : sd;
						copyBuf(swSrc, sA, cropN);
						id<MTLBuffer> A = sA, B = sB;
						float off = 1.0f;
						while (off <= L) {
							const float wk = powf(decay, off);
							streakStep(A, B, p.CW, p.CH, dx*off, dy*off, wk);
							{ id<MTLBuffer> t = A; A = B; B = t; }
							const float sw = p.streakSpread * off;
							if (sw >= 0.25f) {
								streakSpread(A, B, p.CW, p.CH, -dy*sw, dx*sw);
								id<MTLBuffer> t = A; A = B; B = t;
							}
							off *= 2.0f;
						}
						streakAdd(cur, A, swSrc, p.CW, p.CH, scale);
					}
				}
			}
		}

		// --- composite ---------------------------------------------------------
		{
			[e setComputePipelineState:d->composite];
			[e setBuffer:src offset:0 atIndex:0];
			[e setBuffer:cur offset:0 atIndex:1];
			[e setBuffer:dst offset:0 atIndex:2];
			[e setBytes:&p length:sizeof(p) atIndex:3];
			disp2D(d->composite, p.W, p.H);
		}

		[e endEncoding];
		[cb commit];
		[cb waitUntilCompleted];

		freeAll();
		return true;
	}
}
