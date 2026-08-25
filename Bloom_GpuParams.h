/*	Bloom_GpuParams.h

	The POD parameter block shared by the host and the GPU kernels. Its layout
	MUST MATCH the BLGpuParams struct declared inside the kernels (CUDA:
	Bloom_Kernel.cu; Metal: the MSL string in Bloom_Kernel_Metal.h) byte for byte.

	Included by Bloom.cpp (host) and Bloom_Metal.mm (Metal host) so there is a
	single definition to keep in sync.
*/

#pragma once

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
