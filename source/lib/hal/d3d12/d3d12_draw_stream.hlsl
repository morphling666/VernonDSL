// Expands a device draw stream's records, {index_count, instance_count, first_index, base_vertex, first_instance},
// into ExecuteIndirect arguments that first set the vertex stage's draw offsets, {base_vertex, first_instance}, which
// SV_VertexID and SV_InstanceID exclude, then draw.
cbuffer StreamConstants : register(b0) {
    uint record_count;
};

ByteAddressBuffer records : register(t0);
RWByteAddressBuffer arguments : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= record_count)
        return;
    const uint source = id.x * 20;
    const uint4 head = records.Load4(source);
    const uint first_instance = records.Load(source + 16);
    const uint destination = id.x * 28;
    arguments.Store2(destination, uint2(head.w, first_instance));
    arguments.Store4(destination + 8, head);
    arguments.Store(destination + 24, first_instance);
}
