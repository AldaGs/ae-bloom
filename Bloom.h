/*
	Bloom.h

	Fifth learning plugin, ported from python-proto/deep_glow/ (dg_step1..6b).
	A physically-based bloom / glow: light scattering in an optical system.

	THE MODEL (see dg_step6b_artistic.py for the derivation and the A/Bs):

	    out = (1-scatter)*src  +  scatter * gain * (src (*) PSF)      [linear light]

	Everything happens in LINEAR light (blurring gamma-encoded values averages
	highlights too dark - the "dingy smudge" that cheap glows have).

	The PSF is a NORMALIZED octave pyramid, so the convolution preserves total
	light: at gain=1 the effect REDISTRIBUTES light and never creates it. That
	is what makes this a scattering model rather than a brightness effect.

	WHY A PYRAMID (the cost story):
	  A real wide Gaussian costs O(N * radius) - a radius-600 glow is unusable.
	  Instead we shrink the IMAGE, not grow the KERNEL: a small blur on a half-size
	  image covers twice the distance. Stack k halvings and a fixed tiny blur
	  reaches 2^k pixels. Total cost = N*(1 + 1/4 + 1/16 + ...) = 4/3 N,
	  INDEPENDENT OF RADIUS. Measured in the prototype: 4.85ms vs 110ms for a
	  single Gaussian at radius 200.

	  The naive pyramid (5-tap down + bilinear up) is BLOCKY - the mip grid shows
	  as square seams. We use the "dual filter" (Jimenez / Call of Duty 2014, the
	  same bloom Unreal / Unity-HDRP / Blender-EEVEE use): a 13-tap downsample and
	  a 3x3 TENT upsample. Same 4/3 N cost, seams gone.

	CONTROLS - each labelled PHYSICAL or ARTISTIC, so the plugin is honest about
	when it is departing from physics:
	    Radius    how far light scatters                        PHYSICAL
	    Falloff   PSF tail shape (compact -> heavy glare tail)   PHYSICAL
	    Scatter   fraction of light scattered into the halo      PHYSICAL
	    Gain      boost past conservation (1.0 = physical)       ARTISTIC
	    Bias      linear -> luma-weighted extract (0 = physical) ARTISTIC
	  plus Threshold (soft knee), Tone Map, Hue Preserve, Saturation, Unmult, Karis.

	TWO HARD-WON POINTS FROM THE PROTOTYPE:
	  - The extract must be LINEAR (rgb), not rgb*luma. Scattering is linear in
	    intensity; the quadratic version made a 12.8x-brighter pixel glow 230x
	    harder, which blows out HDR sources. (Kept as the `Bias` knob.)
	  - Tone mapping must preserve HUE: per-channel curves squeeze each channel
	    toward 1 at its own rate, washing saturated colors to white. Tone-map the
	    LUMINANCE and keep the chroma ratio instead.

	Like Buildable Stroke and Extended Shadow this is a SmartFX buffer-expanding
	effect (the glow lives outside the layer bounds), so I_EXPAND_BUFFER and the
	result_rect / max_result_rect contract apply.

	STAGE 1 (this file): CPU only. The CUDA scaffold is inherited from Extended
	Shadow and stays dormant (BL_GPU_RENDER 0) until the pyramid kernels exist.
*/

#pragma once

#ifndef BLOOM_H
#define BLOOM_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1	// make sure we get 16bpc pixels; AE_Effect.h checks for this.

#include "AEConfig.h"

#ifdef AE_OS_WIN
	typedef unsigned short PixelType;
	#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"	// PF_GPUDeviceSuite1, GPU cmd structs
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "Bloom_Strings.h"

/* Versioning information */
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

/* Slider ranges, in pixels at full resolution. */
#define	BL_RADIUS_MAX		2000
#define	BL_RADIUS_DFLT		200

/* Tone Map popup (1-based, as AE reports popups). */
#define	BL_TONE_NONE		1
#define	BL_TONE_REINHARD	2
#define	BL_TONE_ACES		3

/* View popup (1-based). Composite = normal; the others isolate the glow. */
#define	BL_VIEW_COMPOSITE	1
#define	BL_VIEW_GLOW		2	// glow only, its own alpha (composite elsewhere)
#define	BL_VIEW_GLOWBLACK	3	// glow flattened opaque on black (grading preview)
#define	BL_VIEW_EXTRACT		4	// DEBUG: raw extract lv[0], opaque on black - no
									// pyramid / streaks / tint / CA / tone map
#define	BL_VIEW_INPUT		5	// DEBUG: the raw pixels AE handed us, untouched.
									// Tests premultiplied-ness: legal premult => rgb <= alpha

/* Chromatic aberration on the glow: Amount 100% maps to this radial scale
   spread (fringe px = radius-from-centre * disp), so a wide halo fringes hard
   while the core stays put. Amount is +-100 so red/blue can be swapped. */
#define	BL_CA_MAX			0.05

/* Unmult divides by alpha to recover straight surface colour. At a near-zero-
   coverage edge pixel (a ~ 1/255) that division blows a bit of rounding up into
   a bright, randomly-coloured FIREFLY - normally hidden under the halo, but
   chromatic aberration splits it into visible R/B specks. Floor the divisor so
   sub-floor coverage can't explode; a>=floor still recovers colour exactly. */
#define	BL_UNMULT_AFLOOR	0.05f

/* Streak Spread: at 100% the ray fans sideways by BL_STREAK_SPREAD_MAX * Length
   at its tip. Real glare broadens along its length - and this lateral blur is
   also the ONLY thing that mixes across the sweep direction. Without it an
   axis-aligned sweep integrates every row in complete isolation and stamps any
   per-row structure down the streak's entire length as a stripe. */
#define	BL_STREAK_SPREAD_MAX	0.15f

/* Streaks (anisotropic glare): a directional geometric-weighted sum of the
   extract, summed over N axes. decay is derived from Length so the streak weight
   falls to BL_STREAK_EPS at the tip. Validated in python-proto/bloom_streaks. */
#define	BL_STREAK_MAX_AXES	8
#define	BL_STREAK_EPS		0.02f

/* Streak Quality popup (1-based). Half = sweep a 2x-downsampled source (~4x
   faster, soft rays fine); Full = sweep at full res (crisp tips, arms cross
   exactly - for hero frames). */
#define	BL_SQ_HALF			1
#define	BL_SQ_FULL			2

/* Pyramid limits. 14 halvings covers any sane frame size (16384 px). */
#define	BL_MAX_LEVELS		14

/* Parameter order. MUST match the order of PF_ADD_* in ParamsSetup. */
enum {
	BL_INPUT = 0,		// index 0 is always the input layer
	BL_TOPIC_GLOW,		// group: scattering shape
	BL_RADIUS,			// how far light scatters, px          PHYSICAL
	BL_FALLOFF,			// PSF tail shape                      PHYSICAL
	BL_SCATTER,			// fraction of light scattered, %      PHYSICAL
	BL_GAIN,			// boost past conservation             ARTISTIC
	BL_BIAS,			// linear -> luma-weighted extract, %  ARTISTIC
	BL_THRESHOLD,		// soft-knee floor, %
	BL_TOPIC_GLOW_END,
	BL_TOPIC_COLOR,		// group: tone / color shaping
	BL_TONE,			// None | Reinhard | ACES
	BL_HUEPRESERVE,		// luminance-only tone mapping, %
	BL_SATURATION,		// saturation bias, %
	BL_TINTCOLOR,		// colour the glow is tinted toward
	BL_TINTAMOUNT,		// how far the glow is tinted, %
	BL_TOPIC_COLOR_END,
	BL_TOPIC_CA,		// group: chromatic aberration on the glow
	BL_CA_AMOUNT,		// per-channel radial spread, +-%
	BL_CA_CENTER,		// centre the fringing radiates from
	BL_TOPIC_CA_END,
	BL_TOPIC_STREAKS,	// group: anisotropic glare streaks
	BL_STREAK_INTENSITY,// master strength, % (0 = off)
	BL_STREAK_THRESHOLD,// only light above this luma streaks, %
	BL_STREAK_ANGLE,	// base axis angle
	BL_STREAK_COUNT,	// number of axes (each smears both ways)
	BL_STREAK_LENGTH,	// reach, px
	BL_STREAK_SPREAD,	// lateral fan of the ray, % of its length
	BL_STREAK_QUALITY,	// Half (fast) | Full (crisp)
	BL_TOPIC_STREAKS_END,
	BL_TOPIC_OUTPUT,	// group: output / cleanup
	BL_VIEW,			// Composite | Glow Only | Glow Over Black
	BL_UNMULT,			// recover true edge colour before extracting
	BL_KARIS,			// firefly suppression on first downsample
	BL_TOPIC_OUTPUT_END,
	BL_NUM_PARAMS
};

/* Stable IDs saved in the project file. Never reuse/renumber once shipped. */
enum {
	RADIUS_DISK_ID = 1,
	FALLOFF_DISK_ID,
	SCATTER_DISK_ID,
	GAIN_DISK_ID,
	BIAS_DISK_ID,
	THRESHOLD_DISK_ID,
	TONE_DISK_ID,
	HUEPRESERVE_DISK_ID,
	SATURATION_DISK_ID,
	UNMULT_DISK_ID,
	KARIS_DISK_ID,
	// Appended for Stage 4a (UI grouping). Never renumber the IDs above.
	GLOW_TOPIC_DISK_ID,
	GLOW_TOPIC_END_DISK_ID,
	COLOR_TOPIC_DISK_ID,
	COLOR_TOPIC_END_DISK_ID,
	OUTPUT_TOPIC_DISK_ID,
	OUTPUT_TOPIC_END_DISK_ID,
	// Appended for Stage 4b (Tint + View).
	TINTCOLOR_DISK_ID,
	TINTAMOUNT_DISK_ID,
	VIEW_DISK_ID,
	// Appended for Stage 4c (chromatic aberration on the glow).
	CA_TOPIC_DISK_ID,
	CA_TOPIC_END_DISK_ID,
	CA_AMOUNT_DISK_ID,
	CA_CENTER_DISK_ID,
	// Appended for Stage 4d (streaks).
	STREAK_TOPIC_DISK_ID,
	STREAK_TOPIC_END_DISK_ID,
	STREAK_INTENSITY_DISK_ID,
	STREAK_ANGLE_DISK_ID,
	STREAK_COUNT_DISK_ID,
	STREAK_LENGTH_DISK_ID,
	STREAK_THRESHOLD_DISK_ID,
	STREAK_QUALITY_DISK_ID,
	// Appended for the streak-spread fix. Never renumber the IDs above.
	STREAK_SPREAD_DISK_ID,
};

/* Per-render snapshot. Built in PreRender, consumed in SmartRender. */
typedef struct BLInfo {
	PF_FpLong	radius;			// px, downsample-scaled
	PF_FpLong	falloff;		// tail exponent
	PF_FpLong	scatter;		// 0..1
	PF_FpLong	gain;			// >= 0 (1 = energy conserving)
	PF_FpLong	bias;			// 0..1
	PF_FpLong	threshold;		// 0..1
	A_long		tone;			// BL_TONE_*
	PF_FpLong	huePreserve;	// 0..1
	PF_FpLong	saturation;		// 0..N (1 = unchanged)
	PF_FpLong	tintR, tintG, tintB;	// tint colour, LINEAR
	PF_FpLong	tintAmt;		// 0..1 (0 = no tint)
	PF_FpLong	caAmt;			// CA radial disp (already scaled), +-BL_CA_MAX
	PF_FpLong	caCenterX, caCenterY;	// CA centre, downsampled LAYER coords
	PF_FpLong	streakInten;	// 0..N (0 = off)
	PF_FpLong	streakThr;		// 0..1 luma gate (0 = everything streaks)
	PF_FpLong	streakAngle;	// base axis, degrees
	PF_FpLong	streakLen;		// px, downsample-scaled
	PF_FpLong	streakSpread;	// lateral fan, 0..BL_STREAK_SPREAD_MAX
	A_long		streakCount;	// number of axes (1..BL_STREAK_MAX_AXES)
	A_long		streakQuality;	// BL_SQ_HALF | BL_SQ_FULL
	A_long		view;			// BL_VIEW_*
	A_long		unmult;			// bool
	A_long		karis;			// bool
	// Rect bookkeeping so SmartRender can map output pixels -> input pixels.
	PF_LRect	inRect;			// what we checked out
	PF_LRect	outRect;		// what we promised to fill
	A_long		pad;			// margin (px) the glow must SEE around the output
} BLInfo, *BLInfoP, **BLInfoH;


extern "C" {

	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#endif // BLOOM_H
