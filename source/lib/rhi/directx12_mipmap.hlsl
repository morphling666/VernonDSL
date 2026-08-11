cbuffer MipmapConstants : register(b0) {
    uint source_width;
    uint source_height;
    uint source_depth_or_layers;
    uint encode_srgb;
};

#if VERNON_MIPMAP_3D
Texture3D<float4> source_3d : register(t0);
RWTexture3D<float4> destination_3d : register(u0);
#else
Texture2DArray<float4> source_2d : register(t0);
RWTexture2DArray<float4> destination_2d : register(u0);
#endif

float3 linear_to_srgb(float3 value) {
    float3 low = value * 12.92;
    float3 high = 1.055 * pow(max(value, 0.0), 1.0 / 2.4) - 0.055;
    return lerp(high, low, 1.0 - step(0.0031308, value));
}

float4 encode_output(float4 value) {
    if (encode_srgb != 0)
        value.rgb = linear_to_srgb(value.rgb);
    return value;
}

#if VERNON_MIPMAP_3D
[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint3 destination_extent =
        uint3(max(1, source_width >> 1), max(1, source_height >> 1), max(1, source_depth_or_layers >> 1));
    if (any(id >= destination_extent))
        return;
    const uint3 source_base = id * 2;
    const uint3 source_limit = uint3(source_width - 1, source_height - 1, source_depth_or_layers - 1);
    float4 value = 0.0;
    [unroll]
    for (uint z = 0; z < 2; ++z)
        [unroll]
        for (uint y = 0; y < 2; ++y)
            [unroll]
            for (uint x = 0; x < 2; ++x)
                value += source_3d.Load(int4(min(source_base + uint3(x, y, z), source_limit), 0));
    destination_3d[id] = encode_output(value * 0.125);
}
#else
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint2 destination_extent = uint2(max(1, source_width >> 1), max(1, source_height >> 1));
    if (id.x >= destination_extent.x || id.y >= destination_extent.y || id.z >= source_depth_or_layers)
        return;
    const uint2 source_base = id.xy * 2;
    const uint2 source_limit = uint2(source_width - 1, source_height - 1);
    float4 value = source_2d.Load(int4(min(source_base, source_limit), id.z, 0));
    value += source_2d.Load(int4(min(source_base + uint2(1, 0), source_limit), id.z, 0));
    value += source_2d.Load(int4(min(source_base + uint2(0, 1), source_limit), id.z, 0));
    value += source_2d.Load(int4(min(source_base + uint2(1, 1), source_limit), id.z, 0));
    destination_2d[id] = encode_output(value * 0.25);
}
#endif
