/* A VernonTensorView of a Host TensorStorage field, so engine code passes Host-owned elements to Vernon programs.
 *
 * The view describes the record's dense row-major host memory; it does not own it. It stays valid while the record
 * holds the same allocation and the strides array lives: Host code assigning the field, or the instance's fini,
 * frees the elements. Link VernonRuntime, which gives the element layout.
 */
#ifndef VERNON_HOST_TENSOR_VIEW_H
#define VERNON_HOST_TENSOR_VIEW_H

#include "VernonRuntime.h"
#include "vernon_host/reflect.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The view of the `rank`-axis storage record at `storage`, whose elements are `dtype` values. `byte_strides`
 * receives the `rank` dense strides the view points at. */
static inline VernonTensorView vh_storage_tensor_view(const void *storage, uint32_t rank, VernonDataType dtype,
                                                      int64_t *byte_strides, VernonValueAccess access) {
    VernonTensorView view;
    memset(&view, 0, sizeof view);
    view.struct_size = sizeof view;
    view.storage = VERNON_TENSOR_HOST;
    view.host_data = vh_storage_data(storage);
    view.element_layout = vernonRuntimeGetScalarValueLayout(dtype);
    view.access = access;
    view.rank = rank;
    view.shape = vh_storage_shape(storage);
    view.byte_strides = byte_strides;
    int64_t stride = (int64_t)view.element_layout.byte_size;
    for (uint32_t axis = rank; axis-- > 0;) {
        byte_strides[axis] = stride;
        stride *= (int64_t)view.shape[axis];
    }
    view.byte_offset = 0;
    view.byte_size = (size_t)(vh_storage_count(storage, rank) * view.element_layout.byte_size);
    return view;
}

#ifdef __cplusplus
}
#endif

#endif /* VERNON_HOST_TENSOR_VIEW_H */
