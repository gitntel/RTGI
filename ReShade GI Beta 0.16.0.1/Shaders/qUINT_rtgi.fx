/*=============================================================================

	ReShade 4 effect file
    github.com/martymcmodding

	Support me:
   		paypal.me/mcflypg
   		patreon.com/mcflypg

    Path Traced Global Illumination 

    * Unauthorized copying of this file, via any medium is strictly prohibited
 	* Proprietary and confidential

=============================================================================*/

/*
TODO: readd halfrez
	  try reducing mips for lowest roughness params
*/


/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef INFINITE_BOUNCES
 #define INFINITE_BOUNCES       0   //[0 or 1]      If enabled, path tracer samples previous frame GI as well, causing a feedback loop to simulate secondary bounces, causing a more widespread GI.
#endif

#ifndef SKYCOLOR_MODE
 #define SKYCOLOR_MODE          0   //[0 to 2]      0: skycolor feature disabled | 1: manual skycolor | 2: dynamic skycolor
#endif

#ifndef MATERIAL_TYPE
 #define MATERIAL_TYPE          0   //[0 to 1]      0: Lambert diffuse | 1: GGX BRDF
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform int UIHELP <
	ui_type = "radio";
	ui_label = " ";	
	ui_text ="This shader adds ray traced / ray marched global illumination to games\nby traversing the height field described by the depth map of the game.\n\nHover over the settings below to display more information.\n\n          >>>>>>>>>> IMPORTANT <<<<<<<<<      \n\nIf the shader appears to do nothing when enabled, make sure ReShade's\ndepth access is properly set up - no output without proper input.\n\n          >>>>>>>>>> IMPORTANT <<<<<<<<<      ";
	ui_category = "Overview / Help";
	ui_category_closed = true;
>;

uniform float RT_SAMPLE_RADIUS <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 20.0;
    ui_step = 0.01;
    ui_label = "Ray Length";
	ui_tooltip = "Maximum ray length, directly affects\nthe spread radius of shadows / bounce lighting";
    ui_category = "Path Tracing";
> = 4.0;

uniform int RT_RAY_AMOUNT <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Amount of Rays";
    ui_tooltip = "Amount of rays launched per pixel in order to\nestimate the global illumination at this location.\nAmount of noise to filter is proportional to sqrt(rays).";
    ui_category = "Path Tracing";
> = 3;

uniform int RT_RAY_STEPS <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Amount of Steps per Ray";
    ui_tooltip = "RTGI performs step-wise raymarching to check for ray hits.\nFewer steps may result in rays skipping over small details.";
    ui_category = "Path Tracing";
> = 12;

uniform float RT_Z_THICKNESS <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 4.0;
    ui_step = 0.01;
    ui_label = "Z Thickness";
	ui_tooltip = "The shader can't know how thick objects are, since it only\nsees the side the camera faces and has to assume a fixed value.\n\nUse this parameter to remove halos around thin objects.";
    ui_category = "Path Tracing";
> = 0.5;

uniform bool RT_HIGHP_LIGHT_SPREAD <
    ui_label = "Enable precise light spreading";
    ui_tooltip = "Rays accept scene intersections within a small error margin.\nEnabling this will snap rays to the actual hit location.\nThis results in sharper but more realistic lighting.";
    ui_category = "Path Tracing";
> = true;

uniform bool RT_BACKFACE_MIRROR <
    ui_label = "Enable simulation of backface lighting";
    ui_tooltip = "RTGI can only simulate light bouncing of the objects visible on the screen.\nTo estimate light coming from non-visible sides of otherwise visible objects,\nthis feature will just take the front-side color instead.";
    ui_category = "Path Tracing";
> = false;

uniform bool RT_ALTERNATE_INTERSECT_TEST <
    ui_label = "Alternate intersection test";
    ui_tooltip = "Enables an alternate way to accept or reject ray hits.\nWill remove halos around thin objects but generate\nmore illumination and shadows elsewhere.\nMore accurate lighting but less pronounced >>SSAO<< outline look.";
    ui_category = "Path Tracing";
> = false;

#if MATERIAL_TYPE == 1
uniform float RT_SPECULAR <
	ui_type = "drag";
	ui_min = 0.01; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Specular";
    ui_tooltip = "Specular Material parameter for GGX Microfacet BRDF";
    ui_category = "Material";
> = 1.0;

uniform float RT_ROUGHNESS <
	ui_type = "drag";
	ui_min = 0.05; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Roughness";
    ui_tooltip = "Roughness Material parameter for GGX Microfacet BRDF";
    ui_category = "Material";
> = 1.0;
#endif

#if SKYCOLOR_MODE != 0

#if SKYCOLOR_MODE == 1
uniform float3 SKY_COLOR <
	ui_type = "color";
	ui_label = "Sky Color";
    ui_category = "Blending";
> = float3(1.0, 0.0, 0.0);
#endif

#if SKYCOLOR_MODE == 2
uniform float SKY_COLOR_SAT <
	ui_type = "drag";
	ui_min = 0; ui_max = 5.0;
    ui_step = 0.01;
    ui_label = "Auto Sky Color Saturation";
    ui_category = "Blending";
> = 1.0;
#endif

uniform float SKY_COLOR_AMBIENT_MIX <
	ui_type = "drag";
	ui_min = 0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Sky Color Ambient Mix";
    ui_tooltip = "How much of the occluded ambient color is considered skycolor\n\nIf 0, Ambient Occlusion removes white ambient color,\nif 1, Ambient Occlusion only removes skycolor";
    ui_category = "Blending";
> = 0.2;

uniform float SKY_COLOR_AMT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Sky Color Intensity";
    ui_category = "Blending";
> = 4.0;
#endif

uniform float RT_AO_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Ambient Occlusion Intensity";
    ui_category = "Blending";
> = 4.0;

uniform float RT_IL_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Bounce Lighting Intensity";
    ui_category = "Blending";
> = 4.0;

#if INFINITE_BOUNCES != 0
    uniform float RT_IL_BOUNCE_WEIGHT <
        ui_type = "drag";
        ui_min = 0; ui_max = 2.0;
        ui_step = 0.01;
        ui_label = "Next Bounce Weight";
        ui_category = "Blending";
    > = 0.0;
#endif

uniform float2 RT_FADE_DEPTH <
	ui_type = "drag";
    ui_label = "Fade Out Start / End";
	ui_min = 0.00; ui_max = 1.00;
	ui_tooltip = "Distance where GI starts to fade out | is completely faded out.";
    ui_category = "Blending";
> = float2(0.0, 0.5);

uniform int RT_DEBUG_VIEW <
	ui_type = "radio";
    ui_label = "Enable Debug View";
	ui_items = "None\0Lighting Channel\0";
	ui_tooltip = "Different debug outputs";
    ui_category = "Debug";
> = 0;
/*
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);
*/
uniform bool RT_DO_RENDER <
    ui_label = "Render a still frame (for screenshots)";
    ui_category = "Experimental";
    ui_tooltip = "This will progressively render a still frame with extreme quality (and filter off, for now).\nTo start rendering, check the box and wait until the result is sufficiently noise-free.\nYou can still adjust blending and toggle debug mode, but do not touch anything else.\nTo resume the game, uncheck the box.\n\nRequires a scene with no moving objects to work properly.";
> = false;

/*=============================================================================
	Textures, Samplers, Globals
=============================================================================*/

#define RESHADE_QUINT_COMMON_VERSION_REQUIRE 202
#define RESHADE_QUINT_EFFECT_DEPTH_REQUIRE
#include "qUINT_common.fxh"

//only works for positive numbers up to 8 bit but I don't expect buffer_width to exceed 61k pixels
#define CONST_LOG2(v)   (((v >> 1u) != 0u) + ((v >> 2u) != 0u) + ((v >> 3u) != 0u) + ((v >> 4u) != 0u) + ((v >> 5u) != 0u) + ((v >> 6u) != 0u) + ((v >> 7u) != 0u))

//for 1920x1080, use 3 mip levels
//double the screen size, use one mip level more
//log2(1920/240) = 3
//log2(3840/240) = 4
#define MIP_AMT 	CONST_LOG2(BUFFER_WIDTH / 240)
#define MIP_BIAS_IL	2
texture ZTex               < pooled = true; >  { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;      MipLevels = MIP_AMT;};
texture ColorTex           < pooled = true; >  { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGB10A2;   MipLevels = MIP_AMT + MIP_BIAS_IL;  };
texture GBufferTex      					    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GBufferTex1      					    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GBufferTex2      					    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITex0	            					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITex1	            					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITex2	            					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITexFilterTemp0 /*infinite bounces*/	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITexFilterTemp1 < pooled = true; >	    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture SkyCol                                  { Width = 1;   			  Height = 1;   			Format = RGBA8; };
texture SkyColPrev                              { Width = 1;   			  Height = 1;   			Format = RGBA8; };
texture JitterTex < source = "bluenoise.png"; > { Width = 32; 			  Height = 32; 				Format = RGBA8; };

sampler sZTex	            					{ Texture = ZTex;	    };
sampler sColorTex	        					{ Texture = ColorTex;	};
sampler sGBufferTex								{ Texture = GBufferTex;	};
sampler sGBufferTex1							{ Texture = GBufferTex1;	};
sampler sGBufferTex2							{ Texture = GBufferTex2;	};
sampler sGITex0       							{ Texture = GITex0;    };
sampler sGITex1       							{ Texture = GITex1;    };
sampler sGITex2       							{ Texture = GITex2;    };
sampler sGITexFilterTemp0       				{ Texture = GITexFilterTemp0;    };
sampler sGITexFilterTemp1       				{ Texture = GITexFilterTemp1;    };
sampler sSkyCol	        						{ Texture = SkyCol;	};
sampler sSkyColPrev	    						{ Texture = SkyColPrev;	};
sampler	sJitterTex          					{ Texture = JitterTex; AddressU = WRAP; AddressV = WRAP;};

/*=============================================================================
	Vertex Shader
=============================================================================*/

struct VSOUT
{
	float4                  vpos        : SV_Position;
    float2                  uv          : TEXCOORD0;
};

VSOUT VS_RT(in uint id : SV_VertexID)
{
    VSOUT o;
    PostProcessVS(id, o.vpos, o.uv); //use original fullscreen triangle VS
    return o;
}

/*=============================================================================
	Functions
=============================================================================*/

struct RTInputs
{
	//per pixel
    float3 pos;
    float3 normal;
    float3 eyedir;
    float3x3 tangent_base;
    float3 jitter;

    //runtime pixel independent
    int nrays;
    int nsteps;
};

#include "RTGI/Projection.fxh"
#include "RTGI/Normal.fxh"
#include "RTGI/RaySorting.fxh"
#include "RTGI/RayTracing.fxh"
#include "RTGI\Denoise.fxh"

RTInputs init(VSOUT i)
{
	RTInputs o;
	o.nrays   = RT_RAY_AMOUNT;
    o.nsteps  = RT_RAY_STEPS;

	o.pos = Projection::uv_to_proj(i.uv);
	o.eyedir = -normalize(o.pos);
	o.normal = tex2D(sGBufferTex, i.uv).xyz;
	o.jitter =                 tex2Dfetch(sJitterTex, int4( i.vpos.xy 	    % 32, 0, 0)).xyz;
    o.jitter = frac(o.jitter + tex2Dfetch(sJitterTex, int4((i.vpos.xy / 32) % 32, 0, 0)).xyz);   
	o.tangent_base = Normal::base_from_vector(o.normal);

    if(RT_DO_RENDER)
    {    
        o.nsteps  = 255;
        o.nrays   = 1;
    }

	return o;
}

void unpack_hdr(inout float3 color)
{
  color = color * rcp(1.01 - saturate(color)); 
}

void pack_hdr(inout float3 color)
{
  color = 1.01 * color * rcp(color + 1.0);  
}

float3 dither(in VSOUT i)
{
    const float2 magicdot = float2(0.75487766624669276, 0.569840290998);
    const float3 magicadd = float3(0, 0.025, 0.0125) * dot(magicdot, 1);

    const int bit_depth = 8; //TODO: add BUFFER_COLOR_DEPTH once it works
    const float lsb = exp2(bit_depth) - 1;

    float3 dither = frac(dot(i.vpos.xy, magicdot) + magicadd);
    dither /= lsb;
    
    return dither;
}

float3 ggx_vndf(float2 uniform_disc, float2 alpha, float3 v)
{
	//scale by alpha, 3.2
	float3 Vh = normalize(float3(alpha * v.xy, v.z));
	//point on projected area of hemisphere
	float2 p = uniform_disc;
	p.y = lerp(sqrt(1.0 - p.x*p.x), 
		       p.y,
		       Vh.z * 0.5 + 0.5);

	float3 Nh =  float3(p.xy, sqrt(saturate(1.0 - dot(p, p)))); //150920 fixed sqrt() of z

	//reproject onto hemisphere
	Nh = mul(Nh, Normal::base_from_vector(Vh));

	//revert scaling
	Nh = normalize(float3(alpha * Nh.xy, saturate(Nh.z)));

	return Nh;
}

float3 schlick_fresnel(float vdoth, float3 f0)
{
	vdoth = saturate(vdoth);
	return lerp(pow(vdoth, 5), 1, f0);
}

float ggx_g2_g1(float3 l, float3 v, float2 alpha)
{
	//smith masking-shadowing g2/g1, v and l in tangent space
	l.xy *= alpha;
	v.xy *= alpha;
	float nl = length(l);
	float nv = length(v);

    float ln = l.z * nv;
    float lv = l.z * v.z;
    float vn = v.z * nl;
    //in tangent space, v.z = ndotv and l.z = ndotl
    return (ln + lv) / (vn + ln + 1e-7);
}

/*=============================================================================
	Pixel Shaders
=============================================================================*/

void PS_InputSetup(in VSOUT i, out float4 color : SV_Target0, out float depth : SV_Target1, out float4 gbuffer : SV_Target2)
{ 
    depth = qUINT::linear_depth(i.uv);
    color = tex2D(qUINT::sBackBufferTex, i.uv);
    color *= saturate(999.0 - depth * 1000.0); //mask sky
    depth = Projection::depth_to_z(depth);
    gbuffer.xyz = Normal::normal_from_depth(i);
    gbuffer.w = depth;
}

//1 -> 2
void PS_Copy_1_to_2(in VSOUT i, out float4 o0 : SV_Target0, out float4 o1 : SV_Target1)
{
	o0 = tex2D(sGITex1, i.uv);
	o1 = tex2D(sGBufferTex1, i.uv);
}

//0 -> 1
void PS_Copy_0_to_1(in VSOUT i, out float4 o0 : SV_Target0, out float4 o1 : SV_Target1)
{
	o0 = tex2D(sGITex0, i.uv);
	o1 = tex2D(sGBufferTex, i.uv);
}

float3 weyl3d(float3 p0, int n) 
{    
   float g = 1.22074408460575947536;
   float3 a = float3(g, g*g, g*g*g);
   return frac(p0 + n * rcp(a));
}

//update 0
void PS_RTMain(in VSOUT i, out float4 o : SV_Target0)
{
	RTInputs parameters = init(i);
	//bias position a bit to fix precision issues
	parameters.pos *= 0.999;
	parameters.pos += parameters.normal * Projection::z_to_depth(parameters.pos.z);
	SampleSet sampleset = ray_sorting(i, qUINT::FRAME_COUNT, parameters.jitter.x); 

#if MATERIAL_TYPE == 1
    float3 specular_color = tex2D(qUINT::sBackBufferTex, i.uv).rgb; 
    specular_color = normalize(specular_color + 0.5) * rsqrt(3.0);
#endif
    o = 0;

    [loop]
    for(int r = 0; r < 0 + parameters.nrays; r++)
    {
        RayTracing::RayDesc ray;
        ray.pos = parameters.pos;

#if MATERIAL_TYPE == 0
        //lambert cosine distribution without TBN reorientation

        ray.dir.z = (r + sampleset.index) / parameters.nrays * 2.0 - 1.0;       
        ray.dir.xy = sampleset.dir_xy * sqrt(1.0 - ray.dir.z * ray.dir.z); //build sphere
        ray.dir = normalize(ray.dir + parameters.normal);
        
if(RT_DO_RENDER) 
{
		//use x and z here as y is used for step jittering
        parameters.jitter = weyl3d(parameters.jitter, (qUINT::FRAME_COUNT % 3000) * parameters.nrays + r);

        ray.dir.z = parameters.jitter.x;
        ray.dir.xy = 1 - ray.dir.z;
        ray.dir = sqrt(ray.dir);
        ray.dir.xy *= float2(sin(parameters.jitter.z * 3.1415927 * 2), cos(parameters.jitter.z * 3.1415927 * 2));
        //reorient ray to surface alignment
        ray.dir = mul(ray.dir, parameters.tangent_base); 
}

#elif MATERIAL_TYPE == 1
        float alpha = RT_ROUGHNESS * RT_ROUGHNESS; //isotropic  
        float3 f0 = specular_color * RT_SPECULAR;
        float3 v = mul(parameters.eyedir, transpose(parameters.tangent_base)); //v to tangent space
        //"random" point on disc - do I have to do sqrt() ?
        float2 uniform_disc = sqrt((r + sampleset.index) / parameters.nrays) * sampleset.dir_xy;

		if(RT_DO_RENDER) //generate a pseudorandom ray for each iteration
		{
			parameters.jitter = weyl3d(parameters.jitter, (qUINT::FRAME_COUNT % 3000) * parameters.nrays + r);
			sincos(parameters.jitter.x * 3.1415927 * 2.0, uniform_disc.y, uniform_disc.x);
			uniform_disc *= sqrt(parameters.jitter.z);
		}

        float3 h = ggx_vndf(uniform_disc, alpha.xx, v);
        float3 l = reflect(-v, h);

        //single scatter lobe
        float3 brdf = ggx_g2_g1(l, v , alpha.xx); //if l.z > 0 is checked later
        brdf = l.z < 1e-7 ? 0 : brdf; //test?
        float vdoth = dot(parameters.eyedir, h);
        brdf *= schlick_fresnel(vdoth, f0);

        ray.dir = mul(l, parameters.tangent_base); //l from tangent to projection
#endif      
        ray.maxlen = RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS;

        //advance to next ray dir
        sampleset.dir_xy = mul(sampleset.dir_xy, sampleset.nextdir); 

        if (dot(ray.dir, parameters.normal) < 0.0)
            continue;

        float cos_view = dot(normalize(parameters.pos), ray.dir);
        ray.steplen = ray.maxlen  * rsqrt(1.0 - cos_view * cos_view) / parameters.nsteps;
        ray.currlen = ray.steplen * parameters.jitter.y;  
        
        float intersected = RayTracing::compute_intersection(ray, parameters, i);
        o.w += intersected;

        if(RT_IL_AMOUNT * intersected == 0) 
            continue;

        float3 albedo           = tex2Dlod(sColorTex,    float4(ray.uv, 0, ray.width + MIP_BIAS_IL)).rgb; unpack_hdr(albedo);
        float3 intersect_normal = tex2Dlod(sGBufferTex,  float4(ray.uv, 0, 0)).xyz;

#if INFINITE_BOUNCES != 0
        float3 nextbounce       = tex2Dlod(sGITexFilterTemp0,  float4(ray.uv, 0, 0)).rgb; unpack_hdr(nextbounce);            
        albedo += nextbounce * RT_IL_BOUNCE_WEIGHT;
#endif
        float backface_check = saturate(dot(-intersect_normal, ray.dir) * 100.0);
        
        //since we searched systematically for an occluder, we can assume there is a direct line of sight between occluder and source point
        //hence all we have to do
        if(RT_BACKFACE_MIRROR)                             
            backface_check = lerp(backface_check, 1.0, 0.1);

        albedo *= backface_check;

#if MATERIAL_TYPE == 0
        o.rgb += albedo;   // * cos(theta) / pdf == 1 here for cosine weighted sampling  
#elif MATERIAL_TYPE == 1

        //albedo.rgb *= fresnel * smith;
        albedo *= brdf;
        albedo *= 10.0;

        o.rgb += albedo;
#endif   
    }

    o /= parameters.nrays; 

//temporal integration stuff

#define read_counter(tex) tex2Dfetch(tex, int4(0,0,0,0)).w
#define store_counter(val) o.w = max(i.vpos.x, i.vpos.y) <= 1.0 ? val : o.w;

    if(!RT_DO_RENDER)
    {
    	store_counter(0);
    }
    else
    {
    	float counter = read_counter(sGITex1);
    	counter++;
    	float4 last_accumulated = tex2D(sGITex1, i.uv);
    	unpack_hdr(last_accumulated.rgb);
    	o = lerp(last_accumulated, o, rcp(counter));
    	store_counter(counter);
    }

	pack_hdr(o.rgb);
}

void PS_Combine(in VSOUT i, out float4 o : SV_Target0)
{
	float4 gi[2], gbuf[2];
	gi[0] = tex2D(sGITex1, i.uv);
	gi[1] = tex2D(sGITex2, i.uv);
	gbuf[0] = tex2D(sGBufferTex1, i.uv);
	gbuf[1] = tex2D(sGBufferTex2, i.uv);

	float4 combined = tex2D(sGITex0, i.uv);
	float sumweight = 1.0;
	float4 gbuf_reference = tex2D(sGBufferTex, i.uv);

	[unroll]
	for(int j = 0; j < 2; j++)
	{
		float4 delta = abs(gbuf_reference - gbuf[j]);

		float normal_sensitivity = 2.0;
		float z_sensitivity = 1.0;

		//TODO: investigate corner cases, if this is actually useful
		float time_delta = qUINT::FRAME_TIME; 
		time_delta = max(time_delta, 1.0) / 16.7; //~1 for 60 fps, expected range
		delta /= time_delta;

		float d = dot(delta, float4(delta.xyz * normal_sensitivity, z_sensitivity)); //normal squared, depth linear
		float w = exp(-d);

		combined += gi[j] * w;
		sumweight += w;
	}
	combined /= sumweight;
	o = combined;
}

void PS_Filter0(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sGITexFilterTemp0, 0, RT_DO_RENDER);
}
void PS_Filter1(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sGITexFilterTemp1, 1, RT_DO_RENDER);
}
void PS_Filter2(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sGITexFilterTemp0, 2, RT_DO_RENDER);
}
void PS_Filter3(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sGITexFilterTemp1, 3, RT_DO_RENDER);
}

void PS_Disp(in VSOUT i, out float4 o : SV_Target0)
{
    float4 gi = tex2D(sGITexFilterTemp0, i.uv);
    float3 color = tex2D(qUINT::sBackBufferTex, i.uv).rgb;

    unpack_hdr(color);
    unpack_hdr(gi.rgb);  

    if(RT_DEBUG_VIEW == 1) color.rgb = 1;

#if SKYCOLOR_MODE != 0
 #if SKYCOLOR_MODE == 1
    float3 skycol = SKY_COLOR;
 #else
    float3 skycol = tex2Dfetch(sSkyCol, int4(0,0,0,0)).rgb;
    skycol = lerp(dot(skycol, 0.333), skycol, SKY_COLOR_SAT * 0.2);
 #endif

    float fade = smoothstep(RT_FADE_DEPTH.y+0.001 /*so that both at 1 don't make it disappear*/, RT_FADE_DEPTH.x, qUINT::linear_depth(i.uv));   
    gi *= fade;  
    skycol *= fade;  

    color = color * (1.0 + gi.rgb * RT_IL_AMOUNT * RT_IL_AMOUNT); //apply GI
    color = color / (1.0 + lerp(1.0, skycol, SKY_COLOR_AMBIENT_MIX) * gi.w * RT_AO_AMOUNT); //apply AO as occlusion of skycolor
    color = color * (1.0 + skycol * SKY_COLOR_AMT);
#else
    float fade = smoothstep(RT_FADE_DEPTH.y+0.001 /*so that both at 1 don't make it disappear*/, RT_FADE_DEPTH.x, qUINT::linear_depth(i.uv));   
    gi *= fade;

	float similarity = distance(normalize(color + 0.00001), normalize(gi.rgb + 0.00001));
	similarity = saturate(similarity * 3.0);
	gi.rgb = lerp(dot(gi.rgb, 0.3333), gi.rgb, saturate(similarity * 0.5 + 0.5));

    color = color * (1.0 + gi.rgb * RT_IL_AMOUNT * RT_IL_AMOUNT); //apply GI
    color = color / (1.0 + gi.w * RT_AO_AMOUNT);

#endif
    pack_hdr(color.rgb);

    //dither a little bit as large scale lighting might exhibit banding
    color.rgb += dither(i);
    o = float4(color, 1);
}

void PS_ReadSkycol(in VSOUT i, out float4 o : SV_Target0)
{
    float2 gridpos;
    gridpos.x = qUINT::FRAME_COUNT % 64;
    gridpos.y = floor(qUINT::FRAME_COUNT / 64) % 64;

    float2 unormgridpos = gridpos / 64.0;

    int searchsize = 10;

    float4 skycolor = 0.0;

    for(float x = 0; x < searchsize; x++)
    for(float y = 0; y < searchsize; y++)
    {
        float2 loc = (float2(x, y) + unormgridpos) * rcp(searchsize);

        float z = qUINT::linear_depth(loc);
        float issky = z == 1;

        skycolor += float4(tex2Dlod(qUINT::sBackBufferTex, float4(loc, 0, 0)).rgb, 1) * issky;
    }

    skycolor.rgb /= skycolor.w + 0.000001;

    float4 prevskycolor = tex2D(sSkyColPrev, 1);

    bool skydetectedthisframe = skycolor.w > 0.000001;
    bool skydetectedatall = prevskycolor.w; //0 if skycolor has not been read yet at all

    float interp = 0;

    //no skycol yet stored, now we have skycolor, use it
    if(!skydetectedatall && skydetectedthisframe)
        interp = 1;

    if(skydetectedatall && skydetectedthisframe)
        interp = saturate(0.1 * 0.01 * qUINT::FRAME_TIME);

    o.rgb = lerp(prevskycolor.rgb, skycolor.rgb, interp);
    o.w = skydetectedthisframe || skydetectedatall;
}

void PS_CopyPrevSkycol(in VSOUT i, out float4 o : SV_Target0)
{
    o = tex2D(sSkyCol, 1.0);
}

/*=============================================================================
	Techniques
=============================================================================*/

technique RTGlobalIllumination
< ui_tooltip = "              >> qUINT::RTGI 0.16 <<\n\n"
               "         EARLY ACCESS -- PATREON ONLY\n"
               "Official versions only via patreon.com/mcflypg\n"
               "\nRTGI is written by Marty McFly / Pascal Gilcher\n"
               "Early access, featureset might be subject to change"; >
{
#if SKYCOLOR_MODE == 2
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_ReadSkycol;
        RenderTarget = SkyCol;
    }
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_CopyPrevSkycol;
        RenderTarget = SkyColPrev;
    }
#endif
	//Update history chain
	pass
    {
		VertexShader = VS_RT;
		PixelShader  = PS_Copy_1_to_2; //1 -> 2
		RenderTarget0 = GITex2; 
		RenderTarget1 = GBufferTex2; 
    }
    pass
    {
		VertexShader = VS_RT;
		PixelShader  = PS_Copy_0_to_1; //0 -> 1
		RenderTarget0 = GITex1;
		RenderTarget1 = GBufferTex1; 
    }
    //Create new inputs
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_InputSetup;
        RenderTarget0 = ColorTex;
        RenderTarget1 = ZTex;
        RenderTarget2 = GBufferTex;
    }
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_RTMain; //update 0
		RenderTarget0 = GITex0;      
	}
	//Combine temporal layers
	pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Combine;
		RenderTarget0 = GITexFilterTemp0;
	}
	//Filter
	pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter0;
        RenderTarget0 = GITexFilterTemp1;
    }
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter1;
        RenderTarget0 = GITexFilterTemp0;
    } 
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter2;
        RenderTarget0 = GITexFilterTemp1;
    } 
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter3;
        RenderTarget = GITexFilterTemp0;
    }
    //Blend
    pass
	{
		VertexShader = VS_RT;
        PixelShader  = PS_Disp;
	}
}