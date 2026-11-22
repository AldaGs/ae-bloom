/*	Bloom.cpp

	Bloom - fifth learning plugin. Physically-based glow: light scattering
	through a normalized point-spread function, computed in linear light.

	    out = (1-scatter)*src + scatter * gain * (src (*) PSF)

	Ported from python-proto/deep_glow/dg_step1..6b. See Bloom.h for the model,
	the pyramid cost argument, and the two hard-won corrections (linear extract,
	hue-preserving tone map).

	Like Buildable Stroke / Extended Shadow this is a SmartFX buffer-expanding
	effect: the glow lives outside the layer bounds, so I_EXPAND_BUFFER and the
	result_rect / max_result_rect contract apply.

	STAGE 1: CPU only. The CUDA scaffold is inherited from Extended Shadow and is
	dormant (BL_GPU_RENDER 0) - the kernel is a passthrough until stage 2 writes
	the real pyramid kernels.

	Revision History
	Version		Change											Engineer	Date
	=======		======											========	======
	1.0			Cloned from Extended Shadow (GPU scaffold)		aldai		8/19/2026
	1.0			Stage 1: params + CPU dual-filter pyramid		aldai		8/19/2026
*/

#define _CRT_SECURE_NO_WARNINGS

#if HAS_CUDA
	#include <cuda_runtime.h>
	// cuda_runtime.h defines these; our header needs its own versions.
	#undef MAJOR_VERSION
	#undef MINOR_VERSION
#endif

#include "Bloom.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <thread>
#include <atomic>

/* ---- GPU rendering -------------------------------------------------------
   At 0 the GPU path is COMPILED and LINKED and AE's GPU_DEVICE_SETUP runs
   (proving toolchain + device plumbing), but PreRender does NOT opt this frame
   into GPU, so every frame renders on the CPU and output stays correct while we
   build. Flip to 1 in stage 2 once the pyramid kernels exist. */
#define BL_GPU_RENDER 1		// STAGE 2: real pyramid kernels

#if HAS_CUDA || HAS_METAL
// POD mirror of the kernel's BLGpuParams - the LAYOUT MUST MATCH the struct in
// Bloom_Kernel.cu (CUDA) and Bloom_Kernel_Metal.h (Metal) exactly. Shared so
// there is a single definition to keep in sync.
#include "Bloom_GpuParams.h"
#endif

#if HAS_CUDA
// Defined in Bloom_Kernel.cu. No extern "C": nvcc uses the same MSVC host
// compiler (-ccbin), so the mangled names match. `wgt` is UNNORMALIZED - the
// launcher normalizes over however many levels it actually managed to build.
extern void BL_Bloom_CUDA(const float *src, float *dst, BLGpuParams p,
						  const float *wgt, int nLev);
#endif

#if HAS_METAL
// Defined in Bloom_Metal.mm (Objective-C++). extern "C" so the plain C names
// match across the C++ / Obj-C++ boundary.
extern "C" bool BL_MetalCompile(void *devicePV, void **outData, char *errBuf, int errLen);
extern "C" void BL_MetalDispose(void *dataPV);
extern "C" bool BL_Bloom_Metal(void *devicePV, void *queuePV, void *dataPV,
							   void *srcMemPV, void *dstMemPV, BLGpuParams p,
							   const float *wgt, int nLev);
#endif

/* Set once in GlobalSetup. Needed to reach the AEGP stream suites, which are the
   ONLY way to change param visibility at runtime in AE (learned on BS). */
static AEGP_PluginID	S_bl_id = 0L;

// Colour transfer, defined further down but needed in PreRender to linearize
// the 8-bit sRGB tint colour param.
static inline float BL_SrgbToLinear (float c);

/* Rec.709 luma, the same weights used since Gradient Map. */
#define BL_LR	0.2126f
#define BL_LG	0.7152f
#define BL_LB	0.0722f

/* =========================================================================
   Boilerplate
   ========================================================================= */

static PF_Err
About (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	suites.ANSICallbacksSuite1()->sprintf(
		out_data->return_msg,
		"%s v%d.%d\r%s",
		STR(StrID_Name), MAJOR_VERSION, MINOR_VERSION, STR(StrID_Description));
	return PF_Err_NONE;
}

static PF_Err
GlobalSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	out_data->my_version = PF_VERSION(	MAJOR_VERSION, MINOR_VERSION,
										BUG_VERSION, STAGE_VERSION, BUILD_VERSION);

	// Needed for the DynamicStream suite used to show/hide tone-map-specific
	// params at runtime.
	suites.UtilitySuite3()->AEGP_RegisterWithAEGP(NULL, STR(StrID_Name), &S_bl_id);

	// I_EXPAND_BUFFER: the glow extends beyond the layer bounds.
	// SEND_UPDATE_PARAMS_UI: needed to sync param visibility on load.
	out_data->out_flags  =	PF_OutFlag_DEEP_COLOR_AWARE |
							PF_OutFlag_I_EXPAND_BUFFER |
							PF_OutFlag_SEND_UPDATE_PARAMS_UI;

	// SmartFX + float + MFR. MUST match PiPL OutFlags_2 exactly (0x0A001400).
	out_data->out_flags2 =	PF_OutFlag2_SUPPORTS_SMART_RENDER |
							PF_OutFlag2_FLOAT_COLOR_AWARE |
							PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

	// Advertise CUDA GPU support (F32 only). Premiere negotiates GPU differently,
	// so gate it to AE. This alone makes AE call GPU_DEVICE_SETUP; a frame only
	// renders on GPU if PreRender ALSO opts in (see BL_GPU_RENDER).
	if (in_data->appl_id != 'PrMr') {
		out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}

	return PF_Err_NONE;
}

/* =========================================================================
   Parameters. Flat list, ordered PHYSICAL first, then the deliberately
   ARTISTIC ones, then output shaping.
   ========================================================================= */

static PF_Err
ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err		err = PF_Err_NONE;
	PF_ParamDef	def;

	// --- GLOW group (scattering shape): PHYSICAL then ARTISTIC ------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Topic_Glow), GLOW_TOPIC_DISK_ID);

	// --- PHYSICAL ---------------------------------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Radius_Param_Name),
							0, BL_RADIUS_MAX, 0, 600, BL_RADIUS_DFLT,
							PF_Precision_TENTHS, 0, 0, RADIUS_DISK_ID);

	// Tail shape. 0 = compact, higher = heavier glare tail (real optical glare
	// is heavy-tailed). Weights are renormalized, so this never changes total
	// energy - only where the energy goes.
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Falloff_Param_Name),
							0, 4, 0, 3, 0.5,
							PF_Precision_HUNDREDTHS, 0, 0, FALLOFF_DISK_ID);

	// Fraction of light that scatters into the halo. The direct image is dimmed
	// by exactly this much, which is what keeps the effect energy conserving.
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Scatter_Param_Name),
							0, 100, 0, 100, 50,
							PF_Precision_TENTHS, 0, 0, SCATTER_DISK_ID);

	// --- ARTISTIC (documented departures from physics) --------------------
	// gain 1.0 = energy conserving. Above that we are adding light on purpose.
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Gain_Param_Name),
							0, 20, 0, 8, 1.0,
							PF_Precision_HUNDREDTHS, 0, 0, GAIN_DISK_ID);

	// 0 = linear extract (physical). 100 = luma-weighted, i.e. quadratic in
	// intensity: only bright things glow, and they glow much harder.
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Bias_Param_Name),
							0, 100, 0, 100, 0,
							PF_Precision_TENTHS, 0, 0, BIAS_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Threshold_Param_Name),
							0, 100, 0, 100, 0,
							PF_Precision_TENTHS, 0, 0, THRESHOLD_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(GLOW_TOPIC_END_DISK_ID);

	// --- COLOR group (tone / color shaping) -------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Topic_Color), COLOR_TOPIC_DISK_ID);

	// SUPERVISE so changing it fires PF_Cmd_USER_CHANGED_PARAM, where we
	// show/hide Hue Preserve (which only means something when tone mapping).
	// PF_ADD_POPUP does not clear def, so setting def.flags first sticks.
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(	STR(StrID_Tone_Param_Name),
					3, BL_TONE_NONE, STR(StrID_Tone_Choices), TONE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_HuePreserve_Param_Name),
							0, 100, 0, 100, 85,
							PF_Precision_TENTHS, 0, 0, HUEPRESERVE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Saturation_Param_Name),
							0, 200, 0, 200, 100,
							PF_Precision_TENTHS, 0, 0, SATURATION_DISK_ID);

	// Tint the glow toward a colour (a coloured optical filter on the halo).
	// Default white + amount 0 = no-op. Applied in LINEAR light.
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(STR(StrID_TintColor_Param_Name), 255, 255, 255, TINTCOLOR_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_TintAmount_Param_Name),
							0, 100, 0, 100, 0,
							PF_Precision_TENTHS, 0, 0, TINTAMOUNT_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(COLOR_TOPIC_END_DISK_ID);

	// --- CHROMATIC ABERRATION group (fringes the glow, not the art) -------
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Topic_CA), CA_TOPIC_DISK_ID);

	// +-100. Positive = red fringes outward / blue inward; negative swaps.
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_CAAmount_Param_Name),
							-100, 100, -100, 100, 0,
							PF_Precision_TENTHS, 0, 0, CA_AMOUNT_DISK_ID);

	// Centre the radial fringing radiates from (default = layer centre).
	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT(STR(StrID_CACenter_Param_Name), 50, 50, 0, CA_CENTER_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(CA_TOPIC_END_DISK_ID);

	// --- STREAKS group (anisotropic glare - astigmatism/starburst) --------
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Topic_Streaks), STREAK_TOPIC_DISK_ID);

	// Master strength. 0 = off (no extra work / bounds growth).
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_StreakIntensity_Param_Name),
							0, 200, 0, 200, 0,
							PF_Precision_TENTHS, 0, 0, STREAK_INTENSITY_DISK_ID);

	// Only light brighter than this streaks (soft knee). 0 = everything streaks;
	// raise it so just highlights throw rays (a real starburst is on lights).
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_StreakThreshold_Param_Name),
							0, 100, 0, 100, 0,
							PF_Precision_TENTHS, 0, 0, STREAK_THRESHOLD_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE(STR(StrID_StreakAngle_Param_Name), 0, STREAK_ANGLE_DISK_ID);

	// Number of axes; each axis smears both ways (1 = astigmatism line,
	// 2 = 4-spike, 3 = 6-spike ...).
	AEFX_CLR_STRUCT(def);
	PF_ADD_SLIDER(	STR(StrID_StreakCount_Param_Name),
					1, BL_STREAK_MAX_AXES, 1, BL_STREAK_MAX_AXES, 2,
					STREAK_COUNT_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_StreakLength_Param_Name),
							0, BL_RADIUS_MAX, 0, 600, BL_RADIUS_DFLT,
							PF_Precision_TENTHS, 0, 0, STREAK_LENGTH_DISK_ID);

	// Lateral fan. Also the only cross-sweep mixing there is (see BL_StreakSweep).
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_StreakSpread_Param_Name),
							0, 100, 0, 100, 0,
							PF_Precision_TENTHS, 0, 0, STREAK_SPREAD_DISK_ID);

	// Half = sweep a 2x-downsampled source (fast, soft rays fine); Full = crisp
	// tips and exactly-crossing arms (hero frames), ~4x slower.
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(	STR(StrID_StreakQuality_Param_Name),
					2, BL_SQ_HALF, STR(StrID_StreakQuality_Choices), STREAK_QUALITY_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(STREAK_TOPIC_END_DISK_ID);

	// --- OUTPUT group (output / cleanup) ----------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Topic_Output), OUTPUT_TOPIC_DISK_ID);

	// Isolate the glow for grading / external compositing.
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(	STR(StrID_View_Param_Name),
					5, BL_VIEW_COMPOSITE, STR(StrID_View_Choices), VIEW_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(STR(StrID_Unmult_Param_Name), "", TRUE, 0, UNMULT_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(STR(StrID_Karis_Param_Name), "", TRUE, 0, KARIS_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(OUTPUT_TOPIC_END_DISK_ID);

	out_data->num_params = BL_NUM_PARAMS;
	return err;
}

/* =========================================================================
   Dynamic show/hide.

   Same mechanism as Buildable Stroke: PF_PUI_INVISIBLE is honored only when a
   param is CREATED, so runtime visibility must go through the AEGP DynamicStream
   suite. Premiere has no stream suites, so we bail there.
   ========================================================================= */

static PF_Err
BL_SetParamVisible (
	PF_InData	*in_data,
	A_long		index,
	bool		visible )
{
	PF_Err				err = PF_Err_NONE, err2 = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	AEGP_EffectRefH		meH		= NULL;
	AEGP_StreamRefH		streamH	= NULL;

	ERR(suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(
			S_bl_id, in_data->effect_ref, &meH));
	ERR(suites.StreamSuite2()->AEGP_GetNewEffectStreamByIndex(
			S_bl_id, meH, index, &streamH));
	if (!err && streamH) {
		ERR(suites.DynamicStreamSuite2()->AEGP_SetDynamicStreamFlag(
				streamH, AEGP_DynStreamFlag_HIDDEN, FALSE, visible ? FALSE : TRUE));
		ERR2(suites.StreamSuite2()->AEGP_DisposeStream(streamH));
	}
	if (meH) ERR2(suites.EffectSuite2()->AEGP_DisposeEffect(meH));
	return err;
}

/* Hue Preserve only does something when a tone curve is actually running. */
static PF_Err
BL_SyncUI (
	PF_InData	*in_data,
	PF_ParamDef	*params[] )
{
	PF_Err	err = PF_Err_NONE;
	if (in_data->appl_id == 'PrMr') return PF_Err_NONE;

	bool toning = (params[BL_TONE]->u.pd.value != BL_TONE_NONE);
	ERR(BL_SetParamVisible(in_data, BL_HUEPRESERVE, toning));
	return err;
}

static PF_Err
UserChangedParam (
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	const PF_UserChangedParamExtra	*which )
{
	PF_Err	err = PF_Err_NONE;
	// UI-only: the value already drives PreRender, so the frame re-renders anyway.
	if (which->param_index == BL_TONE)
		ERR(BL_SyncUI(in_data, params));
	return err;
}

/* =========================================================================
   SmartFX phase 1: PRE-RENDER (region negotiation only).
   ========================================================================= */

static PF_Err
PreRender (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_PreRenderExtra	*extra )
{
	PF_Err				err = PF_Err_NONE, err2 = PF_Err_NONE;
	PF_RenderRequest	req = extra->input->output_request;
	PF_CheckoutResult	in_result;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	PF_Handle infoH = suites.HandleSuite1()->host_new_handle(sizeof(BLInfo));
	if (!infoH) return PF_Err_OUT_OF_MEMORY;

	BLInfo *info = reinterpret_cast<BLInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) { suites.HandleSuite1()->host_dispose_handle(infoH); return PF_Err_OUT_OF_MEMORY; }

	extra->output->pre_render_data = infoH;
	AEFX_CLR_STRUCT(*info);

	// Distances are authored at full res; scale them to the current resolution.
	PF_FpLong dsX = (PF_FpLong)in_data->downsample_x.num / in_data->downsample_x.den;
	PF_FpLong dsY = (PF_FpLong)in_data->downsample_y.num / in_data->downsample_y.den;
	PF_FpLong ds  = (dsX + dsY) * 0.5;

	#define BL_CHECKOUT(idx, var)											\
		PF_ParamDef var;													\
		AEFX_CLR_STRUCT(var);												\
		ERR(PF_CHECKOUT_PARAM(in_data, (idx), in_data->current_time,		\
							  in_data->time_step, in_data->time_scale, &var));

	{
		BL_CHECKOUT(BL_RADIUS,		rad_p);
		BL_CHECKOUT(BL_FALLOFF,		fall_p);
		BL_CHECKOUT(BL_SCATTER,		scat_p);
		BL_CHECKOUT(BL_GAIN,		gain_p);
		BL_CHECKOUT(BL_BIAS,		bias_p);
		BL_CHECKOUT(BL_THRESHOLD,	thr_p);
		BL_CHECKOUT(BL_TONE,		tone_p);
		BL_CHECKOUT(BL_HUEPRESERVE,	hue_p);
		BL_CHECKOUT(BL_SATURATION,	sat_p);
		BL_CHECKOUT(BL_TINTCOLOR,	tintc_p);
		BL_CHECKOUT(BL_TINTAMOUNT,	tinta_p);
		BL_CHECKOUT(BL_CA_AMOUNT,	caamt_p);
		BL_CHECKOUT(BL_CA_CENTER,	cactr_p);
		BL_CHECKOUT(BL_STREAK_INTENSITY,	sinten_p);
		BL_CHECKOUT(BL_STREAK_THRESHOLD,	sthr_p);
		BL_CHECKOUT(BL_STREAK_ANGLE,		sang_p);
		BL_CHECKOUT(BL_STREAK_COUNT,		scnt_p);
		BL_CHECKOUT(BL_STREAK_LENGTH,		slen_p);
		BL_CHECKOUT(BL_STREAK_SPREAD,		sspr_p);
		BL_CHECKOUT(BL_STREAK_QUALITY,		squal_p);
		BL_CHECKOUT(BL_VIEW,		view_p);
		BL_CHECKOUT(BL_UNMULT,		unm_p);
		BL_CHECKOUT(BL_KARIS,		kar_p);

		if (!err) {
			info->radius		= rad_p.u.fs_d.value * ds;
			info->falloff		= fall_p.u.fs_d.value;
			info->scatter		= scat_p.u.fs_d.value / 100.0;
			info->gain			= gain_p.u.fs_d.value;
			info->bias			= bias_p.u.fs_d.value / 100.0;
			info->threshold		= thr_p.u.fs_d.value / 100.0;
			info->tone			= tone_p.u.pd.value;
			info->huePreserve	= hue_p.u.fs_d.value / 100.0;
			info->saturation	= sat_p.u.fs_d.value / 100.0;
			// TINT = colorize the glow (keep its brightness, replace its hue),
			// NOT a subtractive filter. A filter (glow*=tint) can only remove
			// light, so a red filter on a green glow -> black. Instead we
			// normalize the tint colour to luminance 1 and later scale it by the
			// glow's own luminance, so 100% red makes the halo red at the SAME
			// brightness. AE colour params are 8-bit sRGB regardless of depth.
			{
				PF_FpLong tr = BL_SrgbToLinear(tintc_p.u.cd.value.red   / 255.0f);
				PF_FpLong tg = BL_SrgbToLinear(tintc_p.u.cd.value.green / 255.0f);
				PF_FpLong tb = BL_SrgbToLinear(tintc_p.u.cd.value.blue  / 255.0f);
				PF_FpLong lt = 0.2126*tr + 0.7152*tg + 0.0722*tb;
				info->tintAmt = tinta_p.u.fs_d.value / 100.0;
				if (lt > 1e-4) {			// normalize to luminance 1
					info->tintR = tr / lt;
					info->tintG = tg / lt;
					info->tintB = tb / lt;
				} else {					// black/degenerate tint -> no-op
					info->tintR = info->tintG = info->tintB = 1.0;
					info->tintAmt = 0.0;
				}
			}
			info->view			= view_p.u.pd.value;
			// CA: map +-100% to a +-BL_CA_MAX radial scale spread. The centre
			// point is a PF_Fixed in downsampled LAYER space (same space as
			// output_origin / outRect); distances are NOT ds-scaled here (the
			// point already lives in render space), unlike radius. Mapped to the
			// output/cropped buffer in SmartRender via the outRect origin.
			info->caAmt			= (caamt_p.u.fs_d.value / 100.0) * BL_CA_MAX;
			info->caCenterX		= cactr_p.u.td.x_value / 65536.0;
			info->caCenterY		= cactr_p.u.td.y_value / 65536.0;
			info->streakInten	= sinten_p.u.fs_d.value / 100.0;
			info->streakThr		= sthr_p.u.fs_d.value / 100.0;
			info->streakAngle	= sang_p.u.ad.value / 65536.0;	// PF_Fixed deg
			info->streakCount	= scnt_p.u.sd.value;
			info->streakLen		= slen_p.u.fs_d.value * ds;		// px, downsampled
			info->streakSpread	= (sspr_p.u.fs_d.value / 100.0) * BL_STREAK_SPREAD_MAX;
			info->streakQuality	= squal_p.u.pd.value;
			info->unmult		= unm_p.u.bd.value ? 1 : 0;
			info->karis			= kar_p.u.bd.value ? 1 : 0;
		}

		ERR2(PF_CHECKIN_PARAM(in_data, &rad_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &fall_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &scat_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &gain_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &bias_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &thr_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &tone_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &hue_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &sat_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &tintc_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &tinta_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &caamt_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &cactr_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &sinten_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &sthr_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &sang_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &scnt_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &slen_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &sspr_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &squal_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &view_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &unm_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &kar_p));
	}
	#undef BL_CHECKOUT

	// How far the glow can reach, in render px.
	//
	// Radius alone is NOT the reach. The tent upsample spreads a little further
	// each octave, and Falloff loads the coarse levels, so the halo stays visible
	// well past `radius`. MEASURED with a half-plane (the worst case for bounds -
	// an impulse badly under-reports it, since a big bright shape is a sum of
	// many impulses): the distance at which the halo falls below 1/1000 of the
	// shape's own level is ~1.3x radius at falloff 0, rising to ~1.7x at falloff 4.
	//
	// Cropping tighter than the real reach cuts the glow off with a hard step -
	// reported in AE as a vertical seam at x=1817, which was exactly the shape's
	// right edge (1616) + the old pad (201).
	const PF_LRect reqRect = extra->input->output_request.rect;

	PF_FpLong reachF = 1.3 + 0.1 * info->falloff;
	if (reachF < 1.3) reachF = 1.3;
	if (reachF > 2.0) reachF = 2.0;

	A_long grow = (A_long)ceil(info->radius * reachF + 1.0);
	if (grow < 0) grow = 0;

	// Streaks reach ~Length along their axis (they complete the geometric tail,
	// so a bright HDR core carries a little past Length - use the same ~1.5x
	// factor as the halo). Only when streaks are on. The bound is a box, so an
	// axis-aligned streak (the worst case) needs Length in x/y.
	if (info->streakInten > 0.0 && info->streakCount > 0) {
		A_long sgrow = (A_long)ceil(info->streakLen * 1.5 + 1.0);
		if (sgrow > grow) grow = sgrow;
	}

	req.rect.left	-= grow;
	req.rect.top	-= grow;
	req.rect.right	+= grow;
	req.rect.bottom	+= grow;
	// FALSE: the extract divides RGB by coverage to recover the straight colour,
	// which assumes AE hands us PREMULTIPLIED pixels. preserve_rgb_of_zero_alpha
	// breaks exactly that assumption at a shape's antialiased fringe - it keeps
	// full-strength RGB against alpha ~ 0, and the divide then blew those pixels
	// up to ~20x (measured: a green pentagon's edge clipping to pure white in the
	// Extract view). The staircase of hot fringe pixels along a slanted edge is
	// what the axis-aligned streak sweep then smeared into parallel stripes.
	req.preserve_rgb_of_zero_alpha = FALSE;
	req.field = PF_Field_FRAME;

	ERR(extra->cb->checkout_layer(	in_data->effect_ref,
									BL_INPUT, BL_INPUT, &req,
									in_data->current_time, in_data->time_step,
									in_data->time_scale, &in_result));

	if (!err) {
		// Empty input has nothing to glow. Growing an empty rect would invent a
		// box the size of the growth and declare we cover it - the phantom
		// rectangle bug from Buildable Stroke. Stay empty instead.
		const bool inputEmpty =
			(in_result.result_rect.right  <= in_result.result_rect.left) ||
			(in_result.result_rect.bottom <= in_result.result_rect.top);

		if (inputEmpty) {
			extra->output->result_rect     = in_result.result_rect;
			extra->output->max_result_rect = in_result.max_result_rect;
			info->inRect  = in_result.result_rect;
			info->outRect = in_result.result_rect;
			info->pad     = 0;
			extra->output->solid = FALSE;
			suites.HandleSuite1()->host_unlock_handle(infoH);
			return err;
		}

		// max_result_rect MAY exceed the request (it grows the layer bounds so
		// the glow isn't clipped). result_rect MUST stay within the request
		// (per-tile promise). See the Buildable Stroke notes.
		PF_LRect m = in_result.max_result_rect;
		m.left -= grow;  m.top -= grow;  m.right += grow;  m.bottom += grow;
		extra->output->max_result_rect = m;

		PF_LRect r = in_result.result_rect;
		r.left -= grow;  r.top -= grow;  r.right += grow;  r.bottom += grow;
		if (r.left   < reqRect.left)	r.left   = reqRect.left;
		if (r.top    < reqRect.top)		r.top    = reqRect.top;
		if (r.right  > reqRect.right)	r.right  = reqRect.right;
		if (r.bottom > reqRect.bottom)	r.bottom = reqRect.bottom;
		if (r.right  < r.left)			r.right  = r.left;
		if (r.bottom < r.top)			r.bottom = r.top;
		extra->output->result_rect = r;

		// Stash the DECLARED result_rect (not the request) so SmartRender maps
		// output pixels to layer/input coords correctly.
		info->inRect	= in_result.result_rect;
		info->outRect	= r;
		info->pad		= grow;

		extra->output->solid = FALSE;

#if BL_GPU_RENDER
		// The Extract debug view exists only in the CPU path, so don't let AE route
		// the frame to the GPU while it's selected (the kernel knows nothing about
		// view 4 and would quietly render a normal composite instead).
		if (info->view != BL_VIEW_EXTRACT && info->view != BL_VIEW_INPUT)
			extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
#endif
	}

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

/* =========================================================================
   Pixel I/O (any bit depth).
   ========================================================================= */

static float
BL_ReadAlpha (const PF_EffectWorld *w, A_long bd, A_long u, A_long v)
{
	if (u < 0 || v < 0 || u >= w->width || v >= w->height) return 0.0f;
	const char *row = (const char*)w->data + (size_t)v * w->rowbytes;
	switch (bd) {
		case 32: return ((const PF_PixelFloat*)row)[u].alpha;
		case 16: return ((const PF_Pixel16*)row)[u].alpha / (float)PF_MAX_CHAN16;
		default: return ((const PF_Pixel8*)row)[u].alpha  / 255.0f;
	}
}

static void
BL_ReadRGBA (const PF_EffectWorld *w, A_long bd, A_long u, A_long v, float *rgba)
{
	rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0.0f;
	if (u < 0 || v < 0 || u >= w->width || v >= w->height) return;
	const char *row = (const char*)w->data + (size_t)v * w->rowbytes;
	switch (bd) {
		case 32: {
			const PF_PixelFloat *p = &((const PF_PixelFloat*)row)[u];
			rgba[0]=p->red; rgba[1]=p->green; rgba[2]=p->blue; rgba[3]=p->alpha;
		} break;
		case 16: {
			const PF_Pixel16 *p = &((const PF_Pixel16*)row)[u];
			rgba[0]=p->red/(float)PF_MAX_CHAN16; rgba[1]=p->green/(float)PF_MAX_CHAN16;
			rgba[2]=p->blue/(float)PF_MAX_CHAN16; rgba[3]=p->alpha/(float)PF_MAX_CHAN16;
		} break;
		default: {
			const PF_Pixel8 *p = &((const PF_Pixel8*)row)[u];
			rgba[0]=p->red/255.0f; rgba[1]=p->green/255.0f;
			rgba[2]=p->blue/255.0f; rgba[3]=p->alpha/255.0f;
		} break;
	}
}

static void
BL_WriteRGBA (PF_EffectWorld *w, A_long bd, A_long x, A_long y, const float *rgbaIn)
{
	char *row = (char*)w->data + (size_t)y * w->rowbytes;

	// A NaN slips through a naive clamp and an infinity clamps to FULL - either
	// turns a stray value into a solid opaque block. Sanitize alpha to [0,1].
	float rgba[4];
	for (A_long c = 0; c < 4; c++) { float v = rgbaIn[c]; if (!(v == v)) v = 0.0f; rgba[c] = v; }
	if (!(rgba[3] >= 0.0f))	rgba[3] = 0.0f;
	if (rgba[3] > 1.0f)		rgba[3] = 1.0f;

	#define BL_CL(v, mx) ((v) < 0 ? 0 : ((v) > (mx) ? (mx) : (v)))
	switch (bd) {
		case 32: {
			PF_PixelFloat *p = &((PF_PixelFloat*)row)[x];
			p->red=rgba[0]; p->green=rgba[1]; p->blue=rgba[2]; p->alpha=rgba[3];
		} break;
		case 16: {
			PF_Pixel16 *p = &((PF_Pixel16*)row)[x];
			p->red  =(A_u_short)BL_CL(rgba[0]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
			p->green=(A_u_short)BL_CL(rgba[1]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
			p->blue =(A_u_short)BL_CL(rgba[2]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
			p->alpha=(A_u_short)BL_CL(rgba[3]*PF_MAX_CHAN16+0.5f,(float)PF_MAX_CHAN16);
		} break;
		default: {
			PF_Pixel8 *p = &((PF_Pixel8*)row)[x];
			p->red  =(A_u_char)BL_CL(rgba[0]*255.0f+0.5f,255.0f);
			p->green=(A_u_char)BL_CL(rgba[1]*255.0f+0.5f,255.0f);
			p->blue =(A_u_char)BL_CL(rgba[2]*255.0f+0.5f,255.0f);
			p->alpha=(A_u_char)BL_CL(rgba[3]*255.0f+0.5f,255.0f);
		} break;
	}
	#undef BL_CL
}

/* =========================================================================
   Colour science: sRGB <-> linear, tone mapping, saturation.

   Blurring must happen on LIGHT (linear), not on gamma-encoded code values -
   averaging encoded values makes every soft transition too dark, which is the
   "dingy smudge" look of cheap glows.
   ========================================================================= */

static inline float
BL_SrgbToLinear (float c)
{
	if (c <= 0.0f) return 0.0f;
	if (c <= 0.04045f) return c / 12.92f;
	return powf((c + 0.055f) / 1.055f, 2.4f);
}

/* A premultiplied pixel is legal only when rgb <= alpha, so the STRAIGHT colour
   recovered by dividing out coverage can never exceed 1. Clamping there is what
   stops a fringe pixel whose RGB was preserved at full strength (see the
   preserve_rgb_of_zero_alpha note in PreRender) from being multiplied by up to
   1/BL_UNMULT_AFLOOR and clipping to white. */
static inline float
BL_Straight01 (float v)
{
	return (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
}

/* UNUSED - kept as a record of a rejected approach; see the note at the
   extract. A per-pixel legality test cannot separate straight from
   premultiplied once alpha >= max(rgb), which covers most of a bright edge.

   How much to scale RGB by to recover the STRAIGHT colour.

   AE documents premultiplied ARGB and interior pixels are premultiplied, but
   measurement on shape-layer edges (the Input debug view) shows STRAIGHT rgb at
   partial coverage: rgb == the interior colour at alpha 0.03..0.71, so
   rgb > alpha - an illegal premultiplied pixel. Dividing those by coverage
   over-brightens the entire antialiased band (up to 1/BL_UNMULT_AFLOOR = 20x)
   and clips it to white, and the resulting per-row staircase of hot pixels is
   what the axis-aligned streak sweep smears into parallel stripes.

   So decide per pixel instead of assuming: a legal premultiplied pixel has
   rgb <= alpha and needs the divide; anything above that already carries the
   straight colour and must be left alone. Correct under both conventions.
   Validated offline against a real Input+Extract render pair: 0.0000 mean
   error on the 4069 straight AND the 494 genuinely premultiplied edge pixels.

   Known gap: a DARK straight pixel whose rgb happens to fall below its alpha
   reads as premultiplied and gets lifted. It cannot blow up, but it is a
   heuristic - if AE's convention is ever pinned down, replace this with it. */
static inline float
BL_UnmultScale (const float *rgb, float a, float afloor)
{
	float smax = (rgb[0] > rgb[1]) ? rgb[0] : rgb[1];
	if (rgb[2] > smax) smax = rgb[2];
	if (smax > a + (1.0f / 255.0f)) return 1.0f;	// already straight
	return 1.0f / ((a > afloor) ? a : afloor);
}

static inline float
BL_LinearToSrgb (float c)
{
	if (c <= 0.0f) return 0.0f;
	if (c <= 0.0031308f) return c * 12.92f;
	return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

static inline float
BL_Luma (const float *rgb)
{
	return rgb[0]*BL_LR + rgb[1]*BL_LG + rgb[2]*BL_LB;
}

/* Scalar tone curves. `white` is the Reinhard white point. */
static inline float
BL_ToneScalar (float x, A_long mode, float white)
{
	if (x <= 0.0f) return 0.0f;
	if (mode == BL_TONE_REINHARD) return x / (1.0f + x / white);
	if (mode == BL_TONE_ACES) {
		const float a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
		float v = (x * (a*x + b)) / (x * (c*x + d) + e);
		return v < 0.0f ? 0.0f : v;
	}
	return x;
}

/* HUE-PRESERVING tone map. Running a tone curve on R,G,B independently squeezes
   each channel toward 1 at its own rate, so saturated colours wash out to white.
   Instead compress the LUMINANCE and keep the chroma ratio:
       L' = tm(L);  rgb' = rgb * (L'/L)
   `hp` blends toward the safe per-channel version (pure ratio scaling can push a
   channel above 1, which then clips and re-desaturates at the very top). */
static inline void
BL_ToneMap (float *rgb, A_long mode, float hp, float white)
{
	if (mode == BL_TONE_NONE) return;

	float L  = BL_Luma(rgb);
	float Lt = BL_ToneScalar(L, mode, white);
	float ratio = (L > 1e-6f) ? (Lt / L) : 0.0f;

	for (A_long c = 0; c < 3; c++) {
		float lumMapped = rgb[c] * ratio;						// hue preserving
		float perChan   = BL_ToneScalar(rgb[c], mode, white);	// hue shifting
		float v = hp * lumMapped + (1.0f - hp) * perChan;
		rgb[c] = (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
	}
}

static inline void
BL_Saturate (float *rgb, float s)
{
	if (s == 1.0f) return;
	float L = BL_Luma(rgb);
	for (A_long c = 0; c < 3; c++) {
		float v = L + (rgb[c] - L) * s;
		rgb[c] = (v < 0.0f) ? 0.0f : v;
	}
}

/* =========================================================================
   The PSF pyramid (dual filter - Jimenez / Call of Duty 2014).

   Buffers are 4-channel interleaved (R,G,B,A) in CROPPED space. Alpha rides
   along as a 4th channel so the glow gets its OWN coverage: without that the
   glow is invisible on a transparent comp (Deep Glow's "Unmult (Required for
   Alpha)" failure mode, proven in dg_step5).
   ========================================================================= */

/* The 13 downsample taps: offsets in SOURCE pixels from the destination texel
   centre, and their weights (they sum to exactly 1).

       a . b . c          a,c,g,i = 0.03125
       . j . k .          b,d,f,h = 0.0625
       d . e . f          e       = 0.125
       . l . m .          j,k,l,m = 0.125
       g . h . i

   A 5-tap Gaussian is too narrow to kill everything above the new Nyquist limit
   when 3/4 of the pixels are discarded, so aliased high frequencies fold back
   and a round blob turns SQUARE on the coarse levels. These 13 taps fix that.

   CRITICAL - HALF-PIXEL ALIGNMENT: destination pixel dx covers source pixels
   {2dx, 2dx+1}, so its centre sits at source index 2dx + 0.5, NOT 2dx. The taps
   therefore land on HALF-INTEGER positions and must be BILINEAR fetches - which
   is exactly why the published algorithm specifies 13 bilinear taps rather than
   a 25-tap integer stencil.

   Getting this wrong costs half a pixel per octave. The upsample below uses the
   standard (dx+0.5)*ratio-0.5 mapping, which is the exact inverse of a CORRECTLY
   centred downsample; centring at 2dx instead makes the two disagree by 0.5px at
   every level, i.e. 0.5*2^k full-res px at level k. Summed over ~7 levels that
   is tens of pixels of DOWN-RIGHT drift in the finished glow. */
static const float BL_TAP_X[13] = {  0.0f,
									-2.0f, 0.0f, 2.0f,
									-2.0f,       2.0f,
									-2.0f, 0.0f, 2.0f,
									-1.0f, 1.0f,-1.0f, 1.0f };
static const float BL_TAP_Y[13] = {  0.0f,
									-2.0f,-2.0f,-2.0f,
									 0.0f,       0.0f,
									 2.0f, 2.0f, 2.0f,
									-1.0f,-1.0f, 1.0f, 1.0f };
static const float BL_TAP_W[13] = {  0.125f,
									 0.03125f, 0.0625f, 0.03125f,
									 0.0625f,           0.0625f,
									 0.03125f, 0.0625f, 0.03125f,
									 0.125f, 0.125f, 0.125f, 0.125f };

/* 3x3 tent, for the upsample. Plain bilinear reconstruction of a tiny level
   shows the mip grid as diamond seams; the tent overlaps neighbouring cells and
   dissolves them. */
static const float BL_TENT3[9] = {
	1.0f/16, 2.0f/16, 1.0f/16,
	2.0f/16, 4.0f/16, 2.0f/16,
	1.0f/16, 2.0f/16, 1.0f/16
};

#define BL_CLAMPI(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

/* Below this many pixels a pass runs INLINE: std::thread creation is tens of
   microseconds each, and a deep pyramid has many tiny levels, so threading them
   would cost more than the work it saves. */
#define BL_PAR_MIN_PX	((size_t)32768)		// e.g. 256 x 128

/* Split [0,n) into contiguous ranges, one per thread, and run fn(lo,hi) on each.
   fn must touch only the rows its [lo,hi) owns. Capped at 8 threads so this stays
   sane alongside AE's multi-frame rendering (which already uses cores). Defined
   here (above the pyramid) because every pyramid pass is row-parallel. */
template <class F>
static void
BL_ParallelRanges (A_long n, F fn)
{
	if (n <= 0) return;
	A_long hw = (A_long)std::thread::hardware_concurrency();
	if (hw < 1) hw = 1;
	A_long T = (hw < 8) ? hw : 8;
	if (n < 8 || T <= 1) { fn(0, n); return; }
	if (T > n) T = n;

	const A_long chunk = (n + T - 1) / T;
	std::vector<std::thread> pool;
	pool.reserve((size_t)(T - 1));
	for (A_long t = 1; t < T; t++) {
		A_long lo = t * chunk, hi = lo + chunk;
		if (hi > n) hi = n;
		if (lo >= hi) break;
		pool.emplace_back([=]{ fn(lo, hi); });
	}
	fn(0, (chunk < n) ? chunk : n);		// main thread takes the first block
	for (auto &th : pool) th.join();
}

/* ---- radius -> pyramid, shared by the CPU and GPU paths -------------------
   Level k's blur spans ~2^(k+1) full-res px, so the level that reaches `radius`
   is lvlF = log2(radius) - 1. Keeping this in ONE place is what guarantees the
   two paths agree; the GPU normalizes the weights itself over the levels it
   actually built, so these come out UNNORMALIZED on purpose. */
static float
BL_LevelF (PF_FpLong radius)
{
	const float r = (float)((radius > 1.0) ? radius : 1.0);
	const float f = log2f(r) - 1.0f;
	return (f < 0.0f) ? 0.0f : f;
}

static A_long
BL_WantLevels (PF_FpLong radius)
{
	A_long n = (A_long)ceilf(BL_LevelF(radius)) + 1;
	if (n < 1) n = 1;
	if (n > BL_MAX_LEVELS) n = BL_MAX_LEVELS;
	return n;
}

/* w_k = decay^k * (k+1)^falloff * taper_k.
   decay keeps the core bright; falloff pushes energy into the tail (real
   optical glare is heavy-tailed); taper cuts off past the radius with a
   FRACTIONAL last level, which is what makes Radius continuous rather than
   quantized to powers of two. Normalizing (caller's job) makes the PSF
   integrate to 1, i.e. energy conserving at gain = 1. */
static void
BL_RawWeights (A_long nLev, float lvlF, float falloff, float *w)
{
	const float decay = 0.85f;
	for (A_long k = 0; k < nLev; k++) {
		float taper = lvlF - (float)k + 1.0f;
		taper = (taper < 0.0f) ? 0.0f : ((taper > 1.0f) ? 1.0f : taper);
		w[k] = powf(decay, (float)k) * powf((float)(k + 1), falloff) * taper;
	}
}

/* Downsample src(sw,sh) -> dst(dw,dh), 4 channels. When `karis` is set each tap
   is weighted by 1/(1+luma): one blazing pixel can otherwise dominate a whole
   downsample box and FLICKER as it moves sub-pixel between frames (fireflies).
   Only worth doing on the FIRST downsample, where full-res highlights live. */
/* Bilinear fetch at fractional INDEX coords (index i == centre of pixel i),
   clamped at the edges. */
static inline void
BL_Fetch (const float *src, A_long sw, A_long sh, float fx, float fy, float *out)
{
	if (fx < 0.0f) fx = 0.0f; else if (fx > sw - 1) fx = (float)(sw - 1);
	if (fy < 0.0f) fy = 0.0f; else if (fy > sh - 1) fy = (float)(sh - 1);
	const A_long x0 = (A_long)fx, y0 = (A_long)fy;
	const A_long x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
	const A_long y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
	const float tx = fx - x0, ty = fy - y0;
	const float *p00 = &src[((size_t)y0 * sw + x0) * 4];
	const float *p01 = &src[((size_t)y0 * sw + x1) * 4];
	const float *p10 = &src[((size_t)y1 * sw + x0) * 4];
	const float *p11 = &src[((size_t)y1 * sw + x1) * 4];
	for (A_long c = 0; c < 4; c++) {
		const float a0 = p00[c] + (p01[c] - p00[c]) * tx;
		const float a1 = p10[c] + (p11[c] - p10[c]) * tx;
		out[c] = a0 + (a1 - a0) * ty;
	}
}

static void
BL_Downsample (const float *src, A_long sw, A_long sh,
			   float *dst, A_long dw, A_long dh, bool karis)
{
	// Row-parallel: dest row dy reads source rows 2dy-2..2dy+2 (read-only) and
	// writes only its own row, so there is nothing to synchronize. Small levels
	// run INLINE - spawning threads for a 64x32 level costs more than the work,
	// and a deep pyramid has many such levels.
	auto rows = [&](A_long dy0, A_long dy1) {
	for (A_long dy = dy0; dy < dy1; dy++) {
		// dest pixel centre in SOURCE index space (see the alignment note above)
		const float cy = 2.0f * dy + 0.5f;
		for (A_long dx = 0; dx < dw; dx++) {
			const float cx = 2.0f * dx + 0.5f;

			float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float wsum = 0.0f;

			for (A_long t = 0; t < 13; t++) {
				float p[4];
				BL_Fetch(src, sw, sh, cx + BL_TAP_X[t], cy + BL_TAP_Y[t], p);

				float w = BL_TAP_W[t];
				if (karis) w /= (1.0f + BL_Luma(p));
				acc[0] += p[0]*w; acc[1] += p[1]*w;
				acc[2] += p[2]*w; acc[3] += p[3]*w;
				wsum += w;
			}
			const float inv = (karis && wsum > 1e-8f) ? (1.0f / wsum) : 1.0f;
			float *o = &dst[((size_t)dy * dw + dx) * 4];
			o[0]=acc[0]*inv; o[1]=acc[1]*inv; o[2]=acc[2]*inv; o[3]=acc[3]*inv;
		}
	}
	};
	if ((size_t)dw * (size_t)dh < BL_PAR_MIN_PX) rows(0, dh);
	else                                        BL_ParallelRanges(dh, rows);
}

/* Bilinear upsample src(sw,sh) into `scratch`(dw,dh), then 3x3 tent into dst.
   src, dst and scratch MUST be distinct buffers - the bilinear pass writes a
   LARGER region than it reads, so aliasing any two of them corrupts the result. */
/* FIXED factor-of-two inverse mapping. Do NOT use a generic (X+0.5)*sw/dw-0.5
   resize here: when dw is odd, sw = dw/2 (floor) so sw/dw != 0.5 and that becomes
   a SCALE error which compounds every octave. Measured on an impulse: up to
   2.6px of drift over 4 levels at dw=101, versus 0.12px with this mapping - and
   because pad = radius sets the buffer size, the drift changed magnitude AND
   direction as the Radius dial moved.

   The downsample always means "dest d <- source 2d + 0.5" regardless of parity
   (odd dimensions simply drop the last row/column), so its exact inverse is
   f = (X - 0.5)/2, parity-independent. Edge clamping absorbs the dropped edge. */
static inline float
BL_UpCoord (A_long X) { return 0.5f * (float)X - 0.25f; }

static void
BL_UpsampleTent (const float *src, A_long sw, A_long sh,
				 float *dst, A_long dw, A_long dh, float *scratch)
{
	// Pass 1 (bilinear -> scratch) is row-parallel. Pass 2 (tent) reads scratch
	// rows y-1..y+1, so it MUST NOT start until pass 1 has finished everywhere -
	// two separate parallel loops give us exactly that barrier.
	const bool par = ((size_t)dw * (size_t)dh >= BL_PAR_MIN_PX);

	auto pass1 = [&](A_long y0_, A_long y1_) {
	for (A_long dy = y0_; dy < y1_; dy++) {
		float fy = BL_UpCoord(dy);
		if (fy < 0.0f) fy = 0.0f; else if (fy > sh - 1) fy = (float)(sh - 1);
		const A_long y0 = (A_long)fy;
		const A_long y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
		const float ty = fy - y0;

		for (A_long dx = 0; dx < dw; dx++) {
			float fx = BL_UpCoord(dx);
			if (fx < 0.0f) fx = 0.0f; else if (fx > sw - 1) fx = (float)(sw - 1);
			const A_long x0 = (A_long)fx;
			const A_long x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
			const float tx = fx - x0;

			const float *p00 = &src[((size_t)y0 * sw + x0) * 4];
			const float *p01 = &src[((size_t)y0 * sw + x1) * 4];
			const float *p10 = &src[((size_t)y1 * sw + x0) * 4];
			const float *p11 = &src[((size_t)y1 * sw + x1) * 4];
			float *o = &scratch[((size_t)dy * dw + dx) * 4];
			for (A_long c = 0; c < 4; c++) {
				const float a0 = p00[c] + (p01[c] - p00[c]) * tx;
				const float a1 = p10[c] + (p11[c] - p10[c]) * tx;
				o[c] = a0 + (a1 - a0) * ty;
			}
		}
	}
	};
	if (par) BL_ParallelRanges(dh, pass1); else pass1(0, dh);

	auto pass2 = [&](A_long y0_, A_long y1_) {
	for (A_long y = y0_; y < y1_; y++) {
		for (A_long x = 0; x < dw; x++) {
			float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			for (A_long ky = -1; ky <= 1; ky++) {
				const A_long v = BL_CLAMPI(y + ky, 0, dh - 1);
				for (A_long kx = -1; kx <= 1; kx++) {
					const A_long u = BL_CLAMPI(x + kx, 0, dw - 1);
					const float k = BL_TENT3[(ky + 1) * 3 + (kx + 1)];
					const float *p = &scratch[((size_t)v * dw + u) * 4];
					acc[0]+=p[0]*k; acc[1]+=p[1]*k; acc[2]+=p[2]*k; acc[3]+=p[3]*k;
				}
			}
			float *o = &dst[((size_t)y * dw + x) * 4];
			o[0]=acc[0]; o[1]=acc[1]; o[2]=acc[2]; o[3]=acc[3];
		}
	}
	};
	if (par) BL_ParallelRanges(dh, pass2); else pass2(0, dh);
}

/* Soft-knee luma gate in [0,1], matching the extract's threshold shape (a hard
   gate pops as things cross the line during animation). thr<=0 -> passes all. */
static inline float
BL_SoftGate (float lum, float thr)
{
	if (thr <= 0.0f) return 1.0f;
	const float k = (thr * 0.5f > 1e-4f) ? thr * 0.5f : 1e-4f;
	float soft = lum - thr + k;
	soft = (soft < 0.0f) ? 0.0f : ((soft > 2.0f * k) ? 2.0f * k : soft);
	soft = soft * soft / (4.0f * k);
	const float hard = lum - thr;
	float g = (soft > hard) ? soft : hard;
	if (g < 0.0f) g = 0.0f;
	float gate = (lum > 1e-6f) ? (g / lum) : 0.0f;
	return (gate > 1.0f) ? 1.0f : gate;
}

/* Build the streak source: the extract gated by a luma threshold, so only
   bright light throws streaks (a real starburst is on light SOURCES). RGB and
   alpha (coverage) are both gated. */
static void
BL_StreakGate (const float *src, float *dst, A_long CW, A_long CH, float thr)
{
	BL_ParallelRanges(CH, [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; y++)
			for (A_long x = 0; x < CW; x++) {
				const size_t i = ((size_t)y * CW + x) * 4;
				const float lum = 0.2126f*src[i] + 0.7152f*src[i+1] + 0.0722f*src[i+2];
				const float g = BL_SoftGate(lum, thr);
				dst[i+0] = src[i+0]*g; dst[i+1] = src[i+1]*g;
				dst[i+2] = src[i+2]*g; dst[i+3] = src[i+3]*g;
			}
	});
}

/* Separable [1,4,6,4,1] blur, applied ONCE to the FULL-res streak source. At
   axis-aligned angles the integer-exact sweep integrates each column completely
   INDEPENDENTLY (zero lateral filtering), so any column-to-column step in the
   source survives the streak's whole length as a persistent parallel stripe -
   very visible on a transparent bg where the shape is a sharp-edged emitter.
   Measured (python-proto/bloom_streaks/repro_fix3): staircase roughness 0.58 raw
   -> 0.07 with this kernel (8x), tip sharpness essentially unchanged. Half-res
   never shows it because down13/tent already smooth laterally. Blur commutes
   with the shift-invariant sweep, so blurring the source once == blurring every
   axis's result, for one pass instead of one per axis. */
static void
BL_StreakBlur (const float *src, float *dst, A_long CW, A_long CH, float *scratch)
{
	static const float kf[5] = { 1.f/16, 4.f/16, 6.f/16, 4.f/16, 1.f/16 };
	// horizontal -> scratch
	BL_ParallelRanges(CH, [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; y++)
			for (A_long x = 0; x < CW; x++) {
				float a[4] = {0,0,0,0};
				for (A_long t = -2; t <= 2; t++) {
					A_long u = x + t; if (u < 0) u = 0; else if (u >= CW) u = CW - 1;
					const float w = kf[t + 2];
					const float *p = &src[((size_t)y * CW + u) * 4];
					a[0]+=p[0]*w; a[1]+=p[1]*w; a[2]+=p[2]*w; a[3]+=p[3]*w;
				}
				float *o = &scratch[((size_t)y * CW + x) * 4];
				o[0]=a[0]; o[1]=a[1]; o[2]=a[2]; o[3]=a[3];
			}
	});
	// vertical -> dst
	BL_ParallelRanges(CH, [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; y++)
			for (A_long x = 0; x < CW; x++) {
				float a[4] = {0,0,0,0};
				for (A_long t = -2; t <= 2; t++) {
					A_long v = y + t; if (v < 0) v = 0; else if (v >= CH) v = CH - 1;
					const float w = kf[t + 2];
					const float *p = &scratch[((size_t)v * CW + x) * 4];
					a[0]+=p[0]*w; a[1]+=p[1]*w; a[2]+=p[2]*w; a[3]+=p[3]*w;
				}
				float *o = &dst[((size_t)y * CW + x) * 4];
				o[0]=a[0]; o[1]=a[1]; o[2]=a[2]; o[3]=a[3];
			}
	});
}

/* One streak axis-direction: the geometric-weighted directional sum of the
   extract `src` along (dx,dy), then scale*(sum - src) added into `acc`.

   The sum is  S(p) = sum_{t=0}^inf decay^t * src(p - t*d)  built by a 2-buffer
   LOG-DOUBLING:  S_{k+1}(p) = S_k(p) + decay^(2^k) * S_k(p - 2^k*d)  (each pass
   doubles the reach). We overshoot to the next power of two past L: the extra
   terms have weight < BL_STREAK_EPS, so they just complete the infinite tail (a
   smooth exponential end, no hard cut). Every pass is a bilinear shift+add =
   the SAME ping-pong shape as the GPU (validated in python-proto/bloom_streaks).
   Only the TAIL (sum - src) is added; the core src is already the art/halo. */
static void
BL_StreakSweep (const float *src, float *acc, A_long CW, A_long CH,
				float dx, float dy, float L, float decay, float scale,
				float spread, float *bufA, float *bufB)
{
	const size_t N = (size_t)CW * CH * 4;
	memcpy(bufA, src, N * sizeof(float));

	// Lateral (perpendicular) spread. Two jobs at once:
	//   1) a real ray fans out along its length, so the tip is wider than the core;
	//   2) it is the ONLY thing mixing ACROSS the sweep direction. At an
	//      axis-aligned angle the fetch below never leaves its own row, so every
	//      row integrates in complete isolation and any per-row structure in the
	//      source is stamped down the streak's entire length as a stripe.
	// Applied per doubling pass with an offset proportional to that pass's reach,
	// which is what makes the fan grow with distance instead of blurring the core.
	// Perpendicular ONLY - it never runs along the ray, so tip sharpness is
	// untouched (measured: tip gradient 0.01115 -> 0.01121 at spread 0.05).
	const float px = -dy, py = dx;

	float off = 1.0f;
	while (off <= L) {
		const float wk = powf(decay, off);
		const float ox = dx * off, oy = dy * off;
		const float *A = bufA;
		float       *B = bufB;
		BL_ParallelRanges(CH, [&](A_long y0, A_long y1) {
			for (A_long y = y0; y < y1; y++)
				for (A_long x = 0; x < CW; x++) {
					float s[4];
					BL_Fetch(A, CW, CH, (float)x - ox, (float)y - oy, s);
					const size_t i = ((size_t)y * CW + x) * 4;
					for (A_long c = 0; c < 4; c++) B[i + c] = A[i + c] + wk * s[c];
				}
		});
		float *tmp = bufA; bufA = bufB; bufB = tmp;

		if (spread > 0.0f) {
			const float sw = spread * off;
			if (sw >= 0.25f) {			// below a quarter pixel it is a no-op
				const float qx = px * sw, qy = py * sw;
				const float *A2 = bufA;
				float       *B2 = bufB;
				BL_ParallelRanges(CH, [&](A_long y0, A_long y1) {
					for (A_long y = y0; y < y1; y++)
						for (A_long x = 0; x < CW; x++) {
							float sp[4], sm[4];
							BL_Fetch(A2, CW, CH, (float)x + qx, (float)y + qy, sp);
							BL_Fetch(A2, CW, CH, (float)x - qx, (float)y - qy, sm);
							const size_t i = ((size_t)y * CW + x) * 4;
							for (A_long c = 0; c < 4; c++)
								B2[i + c] = 0.5f * A2[i + c]
										  + 0.25f * (sp[c] + sm[c]);
						}
				});
				float *t2 = bufA; bufA = bufB; bufB = t2;
			}
		}

		off *= 2.0f;
	}

	BL_ParallelRanges(CH, [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; y++)
			for (A_long x = 0; x < CW; x++) {
				const size_t i = ((size_t)y * CW + x) * 4;
				for (A_long c = 0; c < 4; c++)
					acc[i + c] += scale * (bufA[i + c] - src[i + c]);
			}
	});
}

/* =========================================================================
   SmartFX phase 2: SMART RENDER (the bloom).
   ========================================================================= */

static PF_Err
SmartRender (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extra )
{
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL;

	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	BLInfo *info = reinterpret_cast<BLInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, BL_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));

	if (!err && inputP && outputP) {
		const A_long	bd	= extra->input->bitdepth;
		const A_long	W	= outputP->width;
		const A_long	H	= outputP->height;
		const A_long	pad	= info->pad;
		const A_long	PW	= W + 2 * pad;
		const A_long	PH	= H + 2 * pad;

		const A_long	offX = info->inRect.left - info->outRect.left;
		const A_long	offY = info->inRect.top  - info->outRect.top;

		const float	scatter	= (float)info->scatter;
		const float	gain	= (float)info->gain;
		const float	bias	= (float)info->bias;
		const float	thr		= (float)info->threshold;
		const float	hp		= (float)info->huePreserve;
		const float	sat		= (float)info->saturation;
		const bool	unmult	= (info->unmult != 0);
		const bool	karis	= (info->karis != 0);
		const A_long tone	= info->tone;

		try {
			// --- bbox of anything with coverage, in padded space --------------
			// Row-parallel with a per-row extent, then a cheap serial merge - a
			// reduction can't be done safely inside the parallel range itself.
			// (This scan is O(PW*PH) and pad is now 1.3-1.7x radius, so it is no
			// longer negligible.)
			// Piggybacked on this scan: decide ONCE PER FRAME whether the input
			// is premultiplied or straight, instead of assuming (AE documents
			// premultiplied ARGB but hands straight RGB for these shape layers).
			// A legal premultiplied pixel has rgb <= alpha, so a single partial-
			// coverage pixel above that proves the frame is straight. Deciding
			// per FRAME is what makes this safe: the per-PIXEL version of the
			// same test silently misfires wherever alpha >= max(rgb), which is
			// most of a bright shape's edge. Only partial-coverage pixels need
			// the RGB read, so this costs almost nothing.
			std::vector<A_long> rowMin((size_t)PH, PW), rowMax((size_t)PH, -1);
			std::vector<char>   rowStraight((size_t)PH, 0);
			BL_ParallelRanges(PH, [&](A_long py0, A_long py1) {
				for (A_long py = py0; py < py1; py++) {
					const A_long iv = py - pad - offY;
					A_long lo = PW, hi = -1;
					char  st = 0;
					for (A_long px = 0; px < PW; px++) {
						const float av = BL_ReadAlpha(inputP, bd, px - pad - offX, iv);
						if (av > 0.0f) {
							if (px < lo) lo = px;
							hi = px;
							if (!st && av < 1.0f) {
								float sp[4];
								BL_ReadRGBA(inputP, bd, px - pad - offX, iv, sp);
								float smax = (sp[0] > sp[1]) ? sp[0] : sp[1];
								if (sp[2] > smax) smax = sp[2];
								if (smax > av + (1.0f / 255.0f)) st = 1;
							}
						}
					}
					rowMin[(size_t)py] = lo;
					rowMax[(size_t)py] = hi;
					rowStraight[(size_t)py] = st;
				}
			});
			bool inputStraight = false;
			for (A_long py = 0; py < PH; py++)
				if (rowStraight[(size_t)py]) { inputStraight = true; break; }

			A_long minX = PW, minY = PH, maxX = -1, maxY = -1;
			for (A_long py = 0; py < PH; py++) {
				if (rowMax[(size_t)py] < 0) continue;		// empty row
				if (rowMin[(size_t)py] < minX) minX = rowMin[(size_t)py];
				if (rowMax[(size_t)py] > maxX) maxX = rowMax[(size_t)py];
				if (py < minY) minY = py;
				maxY = py;
			}

			// Nothing to scatter -> no glow anywhere. Composite copies the input
			// through; the isolate views show empty glow (transparent / black).
			if (maxX < 0) {
				const A_long vw = info->view;
				for (A_long y = 0; y < H; y++)
					for (A_long x = 0; x < W; x++) {
						if (vw == BL_VIEW_COMPOSITE || vw == BL_VIEW_INPUT) {
							float src[4];
							BL_ReadRGBA(inputP, bd, x - offX, y - offY, src);
							BL_WriteRGBA(outputP, bd, x, y, src);
						} else {
							float z[4] = { 0.0f, 0.0f, 0.0f,
								(vw == BL_VIEW_GLOWBLACK ||
								 vw == BL_VIEW_EXTRACT) ? 1.0f : 0.0f };
							BL_WriteRGBA(outputP, bd, x, y, z);
						}
					}
				suites.HandleSuite1()->host_unlock_handle(infoH);
				return err;
			}

			// --- crop every buffer to (bbox + pad) ---------------------------
			const A_long wx0 = MAX(0, minX - pad);
			const A_long wy0 = MAX(0, minY - pad);
			const A_long wx1 = MIN(PW - 1, maxX + pad);
			const A_long wy1 = MIN(PH - 1, maxY + pad);
			const A_long CW  = wx1 - wx0 + 1;
			const A_long CH  = wy1 - wy0 + 1;

			// cropped (cx,cy) -> input pixel
			#define BL_INX(cx)	((cx) + wx0 - pad - offX)
			#define BL_INY(cy)	((cy) + wy0 - pad - offY)
			// output (x,y) -> cropped
			#define BL_CX(x)	((x) + pad - wx0)
			#define BL_CY(y)	((y) + pad - wy0)

			// --- level 0 of the pyramid = the EXTRACT ------------------------
			// The physical extract is LINEAR in intensity (just the light).
			// `bias` blends toward the luma-weighted version, which is quadratic
			// and so makes bright things glow disproportionately harder.
			std::vector< std::vector<float> > lv;
			std::vector<A_long> lw, lh;
			lv.push_back(std::vector<float>((size_t)CW * CH * 4, 0.0f));
			lw.push_back(CW); lh.push_back(CH);

			{
				std::vector<float> &L0 = lv[0];
				BL_ParallelRanges(CH, [&](A_long y0, A_long y1) {
					for (A_long cy = y0; cy < y1; cy++) {
						const A_long iv = BL_INY(cy);
						for (A_long cx = 0; cx < CW; cx++) {
							float s[4];
							BL_ReadRGBA(inputP, bd, BL_INX(cx), iv, s);
							const float a = s[3];

							// Un-premultiply ONLY if this frame actually IS premultiplied. AE
							// documents premultiplied ARGB but hands STRAIGHT rgb for these shape
							// layers - across a whole antialiased ramp the RGB stays CONSTANT
							// while only alpha falls away:
							//
							//   x=1612  rgb=(0.1529,0.7373,0.2078)
							//           a = 1.0 / 0.988 / 0.741 / 0.408 / 0.094
							//
							// Dividing THAT by coverage over-brightens the whole AA band and clips
							// it to white, and the per-row staircase left behind is what the
							// axis-aligned streak sweep smears into parallel stripes.
							//
							// inputStraight is decided once per frame in the bbox scan. Do NOT
							// make this a per-pixel test: it cannot separate the two conventions
							// once alpha >= max(rgb), which covers most of a bright shape's edge.
							// (Old note kept below for the record.)
							// NO un-premultiply. AE documents premultiplied ARGB, but measurement
							// says otherwise for these layers: across a shape's whole antialiased
							// ramp the RGB is CONSTANT while only alpha falls away -
							//
							//   x=1612  rgb=(0.1529,0.7373,0.2078)
							//           a = 1.0 / 0.988 / 0.741 / 0.408 / 0.094
							//
							// i.e. straight colour plus a separate coverage channel. Dividing by
							// coverage over-brightens the whole AA band and clips it to white, and
							// the per-row staircase that leaves is what the axis-aligned streak
							// sweep smears into parallel stripes.
							//
							// A per-pixel legality test (rgb <= alpha => premultiplied) was tried
							// and is NOT sufficient: it misfires wherever alpha exceeds the shape's
							// brightest channel - for the 0.7373 green above that is the whole
							// 0.74..1.0 band, the inner quarter of every edge, still a visible
							// bright line. So don't detect and don't divide; the RGB already IS
							// the straight colour:
							//     straight light = rgb
							//     premultiplied  = rgb * a   (what the non-unmult branch wants)
							//
							// If a host ever DOES hand premultiplied pixels this inverts into a dark
							// fringe on soft edges. Check with the Input debug view: premultiplied
							// data has rgb falling off with alpha instead of holding flat.
							const float ia = inputStraight ? 1.0f
								: (1.0f / ((a > BL_UNMULT_AFLOOR) ? a : BL_UNMULT_AFLOOR));
							float linS[3], premLin[3];
							for (A_long c = 0; c < 3; c++) {
								linS[c]    = BL_SrgbToLinear(BL_Straight01(s[c] * ia));
								premLin[c] = linS[c] * a;
							}

							// UNMULT recovers the true surface colour at
							// semi-transparent edges, which otherwise emit too
							// little light (a dark fringe around soft edges).
							// Without it, the emitted light is simply the
							// premultiplied value - which IS the physical light.
							float emit[3];
							if (unmult) {
								if (a > 1e-4f) { emit[0]=linS[0]; emit[1]=linS[1]; emit[2]=linS[2]; }
								else           { emit[0]=emit[1]=emit[2]=0.0f; }
							} else {
								emit[0]=premLin[0]; emit[1]=premLin[1]; emit[2]=premLin[2];
							}

							const float lum = BL_Luma(emit);

							// artistic: blend toward the luma-weighted extract
							if (bias > 0.0f) {
								for (A_long c = 0; c < 3; c++)
									emit[c] = emit[c] * (1.0f - bias) + emit[c] * lum * bias;
							}

							// soft-knee threshold, normalized so the extract
							// stays LINEAR above the knee (a hard gate pops as
							// things cross the line).
							float gate = 1.0f;
							if (thr > 0.0f) {
								const float k = (thr * 0.5f > 1e-4f) ? thr * 0.5f : 1e-4f;
								float soft = lum - thr + k;
								soft = (soft < 0.0f) ? 0.0f : ((soft > 2.0f*k) ? 2.0f*k : soft);
								soft = soft * soft / (4.0f * k);
								const float hard = lum - thr;
								float g = (soft > hard) ? soft : hard;
								if (g < 0.0f) g = 0.0f;
								gate = (lum > 1e-6f) ? (g / lum) : 0.0f;
								if (gate > 1.0f) gate = 1.0f;
							}

							float *o = &L0[((size_t)cy * CW + cx) * 4];
							o[0] = emit[0] * gate;
							o[1] = emit[1] * gate;
							o[2] = emit[2] * gate;
							o[3] = a * gate;	// alpha scatters too, so the glow
												// is visible on transparency
						}
					}
				});
			}

			// --- build the downsample chain ----------------------------------
			// Radius -> level count via the shared helper, so the CPU and GPU
			// paths cannot drift apart.
			const float lvlF = BL_LevelF(info->radius);
			const A_long wantLevels = BL_WantLevels(info->radius);

			for (A_long k = 1; k < wantLevels && k < BL_MAX_LEVELS; k++) {
				const A_long pw = lw[k-1], ph = lh[k-1];
				const A_long nw = pw / 2, nh = ph / 2;
				if (nw < 2 || nh < 2) break;
				lv.push_back(std::vector<float>((size_t)nw * nh * 4, 0.0f));
				lw.push_back(nw); lh.push_back(nh);
				BL_Downsample(&lv[k-1][0], pw, ph, &lv[k][0], nw, nh, karis && (k == 1));
			}
			const A_long nLev = (A_long)lv.size();

			// --- level weights = the PSF shape --------------------------------
			// decay^k keeps the core bright; (k+1)^falloff pushes energy into the
			// tail (real optical glare is heavy-tailed). taper cuts off past the
			// radius, with a fractional last level so Radius is continuous.
			// NORMALIZED => the PSF integrates to 1 => energy conserving at gain 1.
			std::vector<float> wgt((size_t)nLev, 0.0f);
			{
				BL_RawWeights(nLev, lvlF, (float)info->falloff, &wgt[0]);
				float sum = 0.0f;
				for (A_long k = 0; k < nLev; k++) sum += wgt[k];
				if (sum > 1e-8f) { for (A_long k = 0; k < nLev; k++) wgt[k] /= sum; }
				else             { wgt[0] = 1.0f; }
			}

			// --- progressive upsample + accumulate ---------------------------
			// One octave at a time: upsample, add the next-finer level. Every
			// step is a 2x resample + add, which is exactly one GPU ping-pong
			// pass in stage 2. Three distinct buffers (see BL_UpsampleTent).
			std::vector<float> accBuf((size_t)CW * CH * 4, 0.0f);
			std::vector<float> tentBuf((size_t)CW * CH * 4, 0.0f);
			std::vector<float> upBuf((size_t)CW * CH * 4, 0.0f);
			{
				const A_long top = nLev - 1;
				const size_t topN = (size_t)lw[top] * lh[top] * 4;
				for (size_t i = 0; i < topN; i++) accBuf[i] = lv[top][i] * wgt[top];

				for (A_long k = top - 1; k >= 0; k--) {
					BL_UpsampleTent(&accBuf[0], lw[k+1], lh[k+1],
									&tentBuf[0], lw[k], lh[k], &upBuf[0]);
					accBuf.swap(tentBuf);
					const size_t n = (size_t)lw[k] * lh[k] * 4;
					for (size_t i = 0; i < n; i++)
						accBuf[i] += lv[k][i] * wgt[k];
				}
			}

			// --- streaks (anisotropic glare) ---------------------------------
			// Add a directional geometric-weighted sweep of the extract to the
			// glow, per axis, both directions. decay from Length so the streak
			// weight is BL_STREAK_EPS at the tip; norm keeps the tail energy ~1.
			if (info->streakInten > 0.0 && info->streakCount > 0 &&
				info->streakLen >= 1.0) {
				const A_long HW = CW / 2, HH = CH / 2;
				const bool useHalf = (info->streakQuality != BL_SQ_FULL)
									 && HW >= 2 && HH >= 2;
				if (useHalf) {
				const int   axes  = (info->streakCount < BL_STREAK_MAX_AXES)
									? (int)info->streakCount : BL_STREAK_MAX_AXES;
				const float base  = (float)info->streakAngle;
				// HALF-RESOLUTION: streaks are smooth wide rays, so sweep a
				// 2x-downsampled source and upsample the result (~4x cheaper,
				// visually identical). A half-res step spans 2 full-res px, so
				// Length and decay halve. Uses the SAME alignment-correct down/up
				// as the pyramid (2dx+0.5 down, 0.5X-0.25 up) so it doesn't shift.
				const float Lh    = (float)info->streakLen * 0.5f;
				// spread is a RATIO of length, so it is resolution-independent - the
				// same value works at half res where every step spans 2 full-res px.
				const float spreadH = (float)info->streakSpread;
				const float decay = expf(logf(BL_STREAK_EPS) / (Lh > 1.0f ? Lh : 1.0f));
				const float norm  = (1.0f - decay) / decay;
				const float scale = (float)info->streakInten * norm;

				// Full-res streak source: the extract, gated by threshold if set.
				const float *srcFull = &lv[0][0];
				std::vector<float> gated;
				if (info->streakThr > 0.0) {
					gated.resize((size_t)CW * CH * 4);
					BL_StreakGate(&lv[0][0], &gated[0], CW, CH, (float)info->streakThr);
					srcFull = &gated[0];
				}
				// Downsample to half, sweep there, accumulate into accHalf.
				std::vector<float> srcHalf((size_t)HW * HH * 4);
				BL_Downsample(srcFull, CW, CH, &srcHalf[0], HW, HH, false);
				std::vector<float> accHalf((size_t)HW * HH * 4, 0.0f);
				std::vector<float> sA((size_t)HW * HH * 4);
				std::vector<float> sB((size_t)HW * HH * 4);
				// The sweep integrates each HALF-RES column independently too, so the
				// same per-column stripes appear here - and the tent upsample then
				// WIDENS them to ~2px. (Barely visible over black; obvious over a light
				// background.) Give the half-res source its ~1px lateral width as well.
				std::vector<float> sBlur((size_t)HW * HH * 4);
				BL_StreakBlur(&srcHalf[0], &sBlur[0], HW, HH, &sA[0]);
				const float *swSrc = &sBlur[0];
				for (int i = 0; i < axes; i++) {
					const float ar = (base + i * 180.0f / axes) * 0.017453293f;
					const float dx = cosf(ar), dy = sinf(ar);
					BL_StreakSweep(swSrc, &accHalf[0], HW, HH,  dx,  dy, Lh, decay, scale, spreadH, &sA[0], &sB[0]);
					BL_StreakSweep(swSrc, &accHalf[0], HW, HH, -dx, -dy, Lh, decay, scale, spreadH, &sA[0], &sB[0]);
				}
				// Upsample the half-res streaks to full res and add into the glow.
				std::vector<float> upFull((size_t)CW * CH * 4);
				std::vector<float> upScr((size_t)CW * CH * 4);
				BL_UpsampleTent(&accHalf[0], HW, HH, &upFull[0], CW, CH, &upScr[0]);
				const size_t nFull = (size_t)CW * CH * 4;
				for (size_t j = 0; j < nFull; j++) accBuf[j] += upFull[j];
				} else {
				// FULL-RESOLUTION: sweep at full crop -> crisp tips and arms that
				// cross exactly. ~4x slower; for hero frames.
				const int   axes  = (info->streakCount < BL_STREAK_MAX_AXES)
									? (int)info->streakCount : BL_STREAK_MAX_AXES;
				const float base  = (float)info->streakAngle;
				const float L     = (float)info->streakLen;
				const float spreadF = (float)info->streakSpread;
				const float decay = expf(logf(BL_STREAK_EPS) / (L > 1.0f ? L : 1.0f));
				const float norm  = (1.0f - decay) / decay;
				const float scale = (float)info->streakInten * norm;
				const float *srcFull = &lv[0][0];
				std::vector<float> gated;
				if (info->streakThr > 0.0) {
					gated.resize((size_t)CW * CH * 4);
					BL_StreakGate(&lv[0][0], &gated[0], CW, CH, (float)info->streakThr);
					srcFull = &gated[0];
				}
				std::vector<float> sA((size_t)CW * CH * 4);
				std::vector<float> sB((size_t)CW * CH * 4);
				// Give the source a ~1px lateral width so axis-aligned sweeps don't
				// print persistent per-column stripes (see BL_StreakBlur). Reuses sA
				// as scratch; the blurred source lives in sBlur.
				std::vector<float> sBlur((size_t)CW * CH * 4);
				BL_StreakBlur(srcFull, &sBlur[0], CW, CH, &sA[0]);
				const float *swSrc = &sBlur[0];
				for (int i = 0; i < axes; i++) {
					const float ar = (base + i * 180.0f / axes) * 0.017453293f;
					const float dx = cosf(ar), dy = sinf(ar);
					BL_StreakSweep(swSrc, &accBuf[0], CW, CH,  dx,  dy, L, decay, scale, spreadF, &sA[0], &sB[0]);
					BL_StreakSweep(swSrc, &accBuf[0], CW, CH, -dx, -dy, L, decay, scale, spreadF, &sA[0], &sB[0]);
				}
				}
			}

			// --- composite ----------------------------------------------------
			const float *glowBuf = &accBuf[0];
			// Tint = colorize the glow toward tintN (luma-normalized in PreRender),
			// preserving the glow's own luminance so 100% red -> red halo of the
			// same brightness (not black, as a subtractive filter would give).
			// Applied per-pixel below; alpha (coverage) is left untouched.
			const float tintAmt = (float)info->tintAmt;
			const float tintN[3] = { (float)info->tintR, (float)info->tintG,
									 (float)info->tintB };
			// View: Composite keeps the direct beam; the isolate views drop it.
			const A_long  view = info->view;
			const float   srcW = (view == BL_VIEW_COMPOSITE) ? (1.0f - scatter) : 0.0f;
			// Chromatic aberration: sample the glow's R and B channels at radially
			// scaled positions about the centre, so the halo fringes while the art
			// (composited separately) stays sharp. The displacement (x-xC)*disp is
			// crop-INDEPENDENT: the +pad-wx0 that maps output->cropped cancels
			// between the pixel and the centre, so we work in OUTPUT coords and
			// sample the cropped buffer. G stays exact -> amount 0 is a no-op.
			const float   caDisp = (float)info->caAmt;
			const float   caXC = (float)(info->caCenterX - info->outRect.left);
			const float   caYC = (float)(info->caCenterY - info->outRect.top);
			auto renderRows = [&](A_long y0, A_long y1) {
				for (A_long y = y0; y < y1; y++) {
					const A_long cy = BL_CY(y);
					for (A_long x = 0; x < W; x++) {
						float s[4];
						BL_ReadRGBA(inputP, bd, x - offX, y - offY, s);

						const A_long cx = BL_CX(x);
						if (cx < 0 || cy < 0 || cx >= CW || cy >= CH) {
							// Beyond any reach: no glow here. Composite shows the
							// source; the isolate views show empty glow instead
							// (transparent for Glow Only, black for Glow Over Black).
							if (view == BL_VIEW_COMPOSITE || view == BL_VIEW_INPUT) {
								BL_WriteRGBA(outputP, bd, x, y, s);
							} else {
								float z[4] = { 0.0f, 0.0f, 0.0f,
									(view == BL_VIEW_GLOWBLACK ||
									 view == BL_VIEW_EXTRACT) ? 1.0f : 0.0f };
								BL_WriteRGBA(outputP, bd, x, y, z);
							}
							continue;
						}

						const float a = s[3];

						// DEBUG VIEW: the raw extract, encoded and flattened opaque on
						// black. Nothing downstream runs - no pyramid, no streaks, no
						// tint/CA/tone map - so any structure visible here is structure
						// the EXTRACT itself put in. Alpha is forced to 1 so coverage
						// can't mask what RGB is doing; read the RGB, not the alpha.
						// DEBUG VIEW: the input EXACTLY as AE handed it over, alpha and
						// all. A legal premultiplied pixel always has rgb <= alpha, so
						// comparing the two channels settles whether the extract's
						// divide-by-coverage is correct or is un-premultiplying twice.
						if (view == BL_VIEW_INPUT) {
							BL_WriteRGBA(outputP, bd, x, y, s);
							continue;
						}

						if (view == BL_VIEW_EXTRACT) {
							const float *e = &lv[0][((size_t)cy * CW + cx) * 4];
							float dbg[4];
							for (A_long c = 0; c < 3; c++) {
								const float v = BL_LinearToSrgb(e[c]);
								dbg[c] = (v > 1.0f) ? 1.0f : ((v < 0.0f) ? 0.0f : v);
							}
							dbg[3] = 1.0f;
							BL_WriteRGBA(outputP, bd, x, y, dbg);
							continue;
						}

						const float *gc = &glowBuf[((size_t)cy * CW + cx) * 4];
						float g[4];
						if (caDisp != 0.0f) {
							// dR = scaleR-1 = -disp, dB = +disp: +Amount pushes red
							// outward / blue inward. Alpha from the centre sample so
							// coverage isn't split per channel.
							const float vx = (float)x - caXC, vy = (float)y - caYC;
							float sR[4], sB[4];
							BL_Fetch(glowBuf, CW, CH, cx + vx*(-caDisp), cy + vy*(-caDisp), sR);
							BL_Fetch(glowBuf, CW, CH, cx + vx*( caDisp), cy + vy*( caDisp), sB);
							g[0] = sR[0]; g[1] = gc[1]; g[2] = sB[2]; g[3] = gc[3];
						} else {
							g[0] = gc[0]; g[1] = gc[1]; g[2] = gc[2]; g[3] = gc[3];
						}
						const float k = scatter * gain;

						// Colorize the glow toward the tint (luminance-preserving)
						// before mixing. gL = glow luminance; the tinted colour is
						// tintN*gL (same brightness, tint's hue). amt 0 = untouched.
						float gT[3] = { g[0], g[1], g[2] };
						if (tintAmt > 0.0f) {
							const float gL = 0.2126f*g[0] + 0.7152f*g[1] + 0.0722f*g[2];
							for (A_long c = 0; c < 3; c++)
								gT[c] = g[c] * (1.0f - tintAmt) + (tintN[c] * gL) * tintAmt;
						}

						// GLOW (emissive halo): premultiply-in-LINEAR + tone map,
						// so it stays correct over black. Straighten by its own
						// coverage, shape it, then re-premultiply in linear
						// (encode(E*a), NOT encode(E)*a - gamma-premult darkens
						// partially covered glow pixels >2x).
						// glowA is COVERAGE, scaled by `scatter` only, NEVER by gain:
						// gain boosts glow BRIGHTNESS, not opacity.
						float glowA = g[3] * scatter;
						if (glowA < 0.0f) glowA = 0.0f;
						if (glowA > 1.0f) glowA = 1.0f;
						float glowDisp[3] = { 0.0f, 0.0f, 0.0f };
						if (glowA > 1e-4f) {
							const float ig = 1.0f / glowA;
							float sg[3];
							for (A_long c = 0; c < 3; c++) sg[c] = (gT[c] * k) * ig;
							BL_ToneMap(sg, tone, hp, 4.0f);
							BL_Saturate(sg, sat);
							for (A_long c = 0; c < 3; c++)
								glowDisp[c] = BL_LinearToSrgb(sg[c] * glowA);
						}

						// ART (direct beam): dim the LIGHT by srcW (in LINEAR - the
						// beam carries 1-scatter of the light, and light is linear),
						// but premultiply in GAMMA (encode THEN * a). That is exactly
						// AE's native premult convention, so the source's
						// anti-aliased edge is preserved and always legal (rgb <= a).
						// THIS is the staircase fix: the old code premultiplied the
						// ART in LINEAR (a inside encode), which made bright
						// partial-coverage pixels illegal (encode(E) > a) and let the
						// legality raise force every bright edge pixel opaque. Only
						// the GLOW needs linear premult; the art keeps gamma. Dimming
						// in linear (not gamma) keeps the base brightness unchanged.
						// Same per-frame convention decision as the extract.
						// The * a below is the gamma premultiply AE wants on output.
						const float ia = inputStraight ? 1.0f
										 : ((a > 1e-4f) ? (1.0f / a) : 0.0f);
						float outPx[4];
						for (A_long c = 0; c < 3; c++) {
							const float linS = BL_SrgbToLinear(BL_Straight01(s[c] * ia));
							const float artDisp = BL_LinearToSrgb(linS * srcW) * a;
							float v = artDisp + glowDisp[c];
							outPx[c] = (v > 1.0f) ? 1.0f : v;
						}

						// Coverage = union of art and glow coverage.
						float outA;
						if (view == BL_VIEW_COMPOSITE)		outA = a;   // EMISSIVE: alpha comes from
						// the light via the max(a,mx) raise below, NOT glow coverage.
						// Coverage alpha (a+glowA*(1-a)) veils a WHITE bg with a dark
						// banded smear (glowA carries the swept-column stripes and
						// exceeds the emitted light); over BLACK both are identical.
						else if (view == BL_VIEW_GLOW)		outA = glowA;
						else /* BL_VIEW_GLOWBLACK */		outA = 1.0f;

						// One legality raise (a premultiplied pixel is legal only
						// when rgb <= alpha). The art part is already legal, so this
						// now bites ONLY on glow-dominated pixels: it keeps the halo
						// legal over black without touching the art edge.
						float mx = outPx[0];
						if (outPx[1] > mx) mx = outPx[1];
						if (outPx[2] > mx) mx = outPx[2];
						float finalA = (outA < mx) ? mx : outA;
						if (finalA > 1.0f) finalA = 1.0f;
						outPx[3] = finalA;

						BL_WriteRGBA(outputP, bd, x, y, outPx);
					}
				}
			};
			BL_ParallelRanges(H, renderRows);

			#undef BL_INX
			#undef BL_INY
			#undef BL_CX
			#undef BL_CY
		}
		catch (std::bad_alloc &) {
			err = PF_Err_OUT_OF_MEMORY;
		}
	}

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

/* =========================================================================
   GPU path (CUDA only) - SCAFFOLD stage (passthrough).

   Dormant while BL_GPU_RENDER is 0: AE still runs GPU_DEVICE_SETUP (proving the
   toolchain and device plumbing) but never dispatches a frame here.
   ========================================================================= */

static PF_Err
GPUDeviceSetup (
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_GPUDeviceSetupExtra	*extra )
{
	// CUDA (Windows): kernel is statically linked, nothing to build here - just
	// claim F32 support. Any framework we don't handle is left on the CPU path.
	if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
		out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#if HAS_METAL
	// Metal (macOS): compile the MSL library and build the pipeline states once
	// per device, stash them in gpu_data. On a compile error, surface Metal's
	// own message in AE's error dialog so it can be read directly.
	else if (extra->input->what_gpu == PF_GPU_Framework_METAL) {
		PF_GPUDeviceSuite1 *gpu = NULL;
		if (in_data->pica_basicP->AcquireSuite(kPFGPUDeviceSuite,
				kPFGPUDeviceSuiteVersion1, (const void**)&gpu) || !gpu)
			return PF_Err_BAD_CALLBACK_PARAM;

		PF_GPUDeviceInfo info;
		AEFX_CLR_STRUCT(info);
		PF_Err e = gpu->GetDeviceInfo(in_data->effect_ref,
					extra->input->device_index, &info);
		in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
		if (e) return e;

		void *metalData = NULL;
		char  errBuf[512] = {0};
		if (!BL_MetalCompile(info.devicePV, &metalData, errBuf, sizeof(errBuf))) {
			PF_STRCPY(out_data->return_msg, "Bloom Metal build failed: ");
			strncat(out_data->return_msg, errBuf,
					sizeof(out_data->return_msg) - strlen(out_data->return_msg) - 1);
			out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}
		extra->output->gpu_data = metalData;			// opaque; freed in Setdown
		out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
	return PF_Err_NONE;
}

static PF_Err
GPUDeviceSetdown (
	PF_InData					*in_data,
	PF_OutData					*out_data,
	PF_GPUDeviceSetdownExtra	*extra )
{
#if HAS_METAL
	// Release the Metal pipeline states built in GPUDeviceSetup.
	if (extra->input->what_gpu == PF_GPU_Framework_METAL && extra->input->gpu_data) {
		BL_MetalDispose(const_cast<void*>(extra->input->gpu_data));
	}
#endif
	// CUDA kernel is statically linked; nothing device-specific to release.
	return PF_Err_NONE;
}

#if HAS_CUDA || HAS_METAL
static PF_Err
SmartRenderGPU (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extra )
{
	PF_Err				err		= PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL;

	// pre_render_data is the PF_Handle from PreRender - it must be LOCKED to get
	// the BLInfo pointer, exactly as the CPU SmartRender does (treating the
	// handle as the pointer gives garbage rects and the layer disappears).
	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	BLInfo *info = reinterpret_cast<BLInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	PF_GPUDeviceSuite1 *gpu = NULL;
	err = in_data->pica_basicP->AcquireSuite(
			kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, (const void**)&gpu);
	if (err || !gpu) {
		suites.HandleSuite1()->host_unlock_handle(infoH);
		return err ? err : PF_Err_BAD_CALLBACK_PARAM;
	}

	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, BL_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));

	if (!err && inputP && outputP) {
		void *src_mem = NULL, *dst_mem = NULL;
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, inputP,  &src_mem));
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, outputP, &dst_mem));

		if (!err) {
			const int bytesPerPixel = 16;			// GPU_BGRA128 = 4 x float32

			BLGpuParams p;
			memset(&p, 0, sizeof(p));
			p.offX     = info->inRect.left - info->outRect.left;
			p.offY     = info->inRect.top  - info->outRect.top;
			p.inW      = inputP->width;
			p.inH      = inputP->height;
			p.W        = outputP->width;
			p.H        = outputP->height;
			p.srcPitch = inputP->rowbytes  / bytesPerPixel;
			p.dstPitch = outputP->rowbytes / bytesPerPixel;
			p.pad      = (int)info->pad;
			// wx0/wy0/CW/CH are filled by BL_Bloom_CUDA after its bbox pass.

			p.scatter     = (float)info->scatter;
			p.gain        = (float)info->gain;
			p.bias        = (float)info->bias;
			p.threshold   = (float)info->threshold;
			p.huePreserve = (float)info->huePreserve;
			p.saturation  = (float)info->saturation;
			p.tintR       = (float)info->tintR;
			p.tintG       = (float)info->tintG;
			p.tintB       = (float)info->tintB;
			p.tintAmt     = (float)info->tintAmt;
			// CA centre in OUTPUT coords (layer - outRect origin); the crop-
			// independent displacement lets the kernel work purely in output space.
			p.caAmt       = (float)info->caAmt;
			p.caCX        = (float)(info->caCenterX - info->outRect.left);
			p.caCY        = (float)(info->caCenterY - info->outRect.top);
			p.streakInten = (float)info->streakInten;
			p.streakSpread = (float)info->streakSpread;
			p.streakAngle = (float)info->streakAngle;
			p.streakLen   = (float)info->streakLen;
			p.streakThr   = (float)info->streakThr;
			p.streakCount = (int)info->streakCount;
			p.streakQuality = (int)info->streakQuality;
			p.tone        = (int)info->tone;
			p.unmult      = (int)info->unmult;
			// -1 = undecided; the GPU bbox pass detects it on device (see bl_bbox).
			p.inputStraight = -1;
			p.karis       = (int)info->karis;
			p.view        = (int)info->view;

			// Same radius -> levels mapping as the CPU path (shared helpers).
			// Weights go over UNNORMALIZED: the launcher normalizes across the
			// levels it actually builds, which can be fewer if the crop is small.
			const A_long nLev = BL_WantLevels(info->radius);
			float wgt[BL_MAX_LEVELS];
			memset(wgt, 0, sizeof(wgt));
			BL_RawWeights(nLev, BL_LevelF(info->radius), (float)info->falloff, wgt);

#if HAS_CUDA
			if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
				BL_Bloom_CUDA((const float*)src_mem, (float*)dst_mem, p, wgt, (int)nLev);
				if (cudaPeekAtLastError() != cudaSuccess)
					err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
#endif
#if HAS_METAL
			if (extra->input->what_gpu == PF_GPU_Framework_METAL) {
				// Metal needs the device + command queue (CUDA used the default
				// context). gpu_data holds the pipeline states built in setup.
				PF_GPUDeviceInfo devInfo;
				AEFX_CLR_STRUCT(devInfo);
				ERR(gpu->GetDeviceInfo(in_data->effect_ref,
						extra->input->device_index, &devInfo));
				if (!err) {
					if (!BL_Bloom_Metal(devInfo.devicePV, devInfo.command_queuePV,
							const_cast<void*>(extra->input->gpu_data),
							src_mem, dst_mem, p, wgt, (int)nLev))
						err = PF_Err_INTERNAL_STRUCT_DAMAGED;
				}
			}
#endif
		}
	}

	in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}
#endif // HAS_CUDA || HAS_METAL

/* =========================================================================
   Classic render path (Premiere / legacy): pass through. AE uses SmartFX.
   ========================================================================= */

static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	if (params[BL_INPUT] && output) {
		ERR(suites.WorldTransformSuite1()->copy(
				in_data->effect_ref, &params[BL_INPUT]->u.ld, output, NULL, NULL));
	}
	return err;
}

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		"Bloom",					// Name
		"aldai Bloom",				// Match Name
		"ags_utilities",					// Category
		AE_RESERVED_INFO,			// Reserved Info
		"EffectMain",				// Entry point
		"https://www.adobe.com");	// support URL

	return result;
}

PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err err = PF_Err_NONE;
	try {
		switch (cmd) {
			case PF_Cmd_ABOUT:
				err = About(in_data, out_data, params, output);
				break;
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_USER_CHANGED_PARAM:		// Tone Map popup changed
				err = UserChangedParam(in_data, out_data, params,
						reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
				break;
			case PF_Cmd_UPDATE_PARAMS_UI:		// sync visibility on load
				err = BL_SyncUI(in_data, params);
				break;
			case PF_Cmd_RENDER:
				err = Render(in_data, out_data, params, output);
				break;
			case PF_Cmd_SMART_PRE_RENDER:
				err = PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra));
				break;
			case PF_Cmd_SMART_RENDER:
				err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra));
				break;
			case PF_Cmd_GPU_DEVICE_SETUP:
				err = GPUDeviceSetup(in_data, out_data,
						reinterpret_cast<PF_GPUDeviceSetupExtra*>(extra));
				break;
			case PF_Cmd_GPU_DEVICE_SETDOWN:
				err = GPUDeviceSetdown(in_data, out_data,
						reinterpret_cast<PF_GPUDeviceSetdownExtra*>(extra));
				break;
#if HAS_CUDA || HAS_METAL
			case PF_Cmd_SMART_RENDER_GPU:
				err = SmartRenderGPU(in_data, out_data,
						reinterpret_cast<PF_SmartRenderExtra*>(extra));
				break;
#endif
		}
	}
	catch(PF_Err &thrown_err){
		err = thrown_err;
	}
	return err;
}
