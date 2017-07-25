cdef extern from "<ctc.h>":
    ctypedef enum ctcStatus_t:
        CTC_STATUS_SUCCESS
        CTC_STATUS_MEMOPS_FAILED
        CTC_STATUS_INVALID_VALUE
        CTC_STATUS_EXECUTION_FAILED
        CTC_STATUS_UNKNOWN_ERROR

    ctypedef enum ctcComputeLocation:
        CTC_CPU
        CTC_GPU

    ctypedef struct CUstream:
        pass

    int get_warpctc_version()
    const char* ctcGetStatusString(ctcStatus_t status)

    ctypedef struct ctcOptions:
        ctcComputeLocation loc
        unsigned int num_threads
        CUstream stream
        int blank_label

    ctcStatus_t compute_ctc_loss(
            const float* activations,
            float* gradients,
            const int* flat_labels,
            const int* label_lengths,
            const int* input_length,
            int alphabet_size,
            int minibatch,
            float * costs,
            void* workspace,
            ctcOptions options
    )

    ctcStatus_t get_workspace_size(
            const int* label_lengths,
            const int* input_lengths,
            int alphabet_size,
            int minibatch,
            ctcOptions info,
            size_t* size_bytes,
    )
