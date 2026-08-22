/*
	Bloom_Strings.cpp
*/

#include "Bloom.h"

typedef struct {
	A_u_long	index;
	A_char		str[256];
} TableString;

TableString g_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						"Bloom",
	StrID_Description,				"Physically-based bloom: light scattering through a normalized PSF pyramid, in linear light.\rPorted from python-proto/deep_glow.",
	StrID_Radius_Param_Name,		"Radius",
	StrID_Falloff_Param_Name,		"Falloff",
	StrID_Scatter_Param_Name,		"Scatter",
	StrID_Gain_Param_Name,			"Gain",
	StrID_Bias_Param_Name,			"Highlight Bias",
	StrID_Threshold_Param_Name,		"Threshold",
	StrID_Tone_Param_Name,			"Tone Map",
	StrID_HuePreserve_Param_Name,	"Hue Preserve",
	StrID_Saturation_Param_Name,	"Saturation",
	StrID_Unmult_Param_Name,		"Unmult",
	StrID_Karis_Param_Name,			"Suppress Fireflies",
	StrID_Tone_Choices,				"None|Reinhard|ACES",
	StrID_Topic_Glow,				"Glow",
	StrID_Topic_Color,				"Color",
	StrID_Topic_Output,				"Output",
	StrID_TintColor_Param_Name,		"Tint Color",
	StrID_TintAmount_Param_Name,	"Tint Amount",
	StrID_View_Param_Name,			"View",
	StrID_View_Choices,				"Composite|Glow Only|Glow Over Black|Extract (Debug)|Input (Debug)",
	StrID_Topic_CA,					"Chromatic Aberration",
	StrID_CAAmount_Param_Name,		"Amount",
	StrID_CACenter_Param_Name,		"Center",
	StrID_Topic_Streaks,			"Streaks",
	StrID_StreakIntensity_Param_Name,	"Intensity",
	StrID_StreakThreshold_Param_Name,	"Threshold",
	StrID_StreakAngle_Param_Name,	"Angle",
	StrID_StreakCount_Param_Name,	"Count",
	StrID_StreakLength_Param_Name,	"Length",
	StrID_StreakSpread_Param_Name,	"Spread",
	StrID_StreakQuality_Param_Name,	"Quality",
	StrID_StreakQuality_Choices,	"Half (Fast)|Full",
};

char *GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}
