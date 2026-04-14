#include <type_traits>
#include <utility>
#include <variant>

typedef enum CUDTXresult_enum {
    /// No errors
    CUDTX_SUCCESS = 0ULL,
    /// Generic error
    CUDTX_ERROR = 1ULL,
    /// Invalid parameter
    CUDTX_ERROR_INVALID_PARAM = 2ULL,
    /// Input buffer too small
    CUDTX_ERROR_BUFFER_TOO_SMALL = 3ULL,
    CUDTX_ERROR_OUT_OF_MEMORY = 4ULL,
    CUDTX_ERROR_OUT_OF_RESOURCES = 5ULL,
    CUDTX_ERROR_INVALID_HANDLE = 6ULL,
    CUDTX_ERROR_NOT_IMPLEMENTED = 7ULL,
    CUDTX_ERROR_UNINITIALIZED = 8ULL,
    CUDTX_ERROR_ALREADY_INITIALIZED = 9ULL,
    CUDTX_ERROR_TIMEOUT = 10ULL,
    CUDTX_ERROR_TASK_ALREADY_REGISTERED = 11ULL,
    CUDTX_ERROR_INITIALIZATION_FAILED = 12ULL,
    CUDTX_ERROR_CONTROL_STORE_DATA_NOT_READY = 13ULL,
    CUDTX_ERROR_LAUNCH_TASK_FAILED = 14ULL,
    CUDTX_ERROR_NO_TASK_AVAILABLE = 15ULL,
    CUDTX_ERROR_NOT_FOUND = 16ULL,
    CUDTX_ERROR_NOT_ALLOCATED = 17ULL,
    CUDTX_ERROR_NOT_READY = 18ULL,
    CUDTX_ERROR_EXISTS = 19ULL,
    CUDTX_ERROR_OUT_OF_BOUNDS = 20ULL,
    CUDTX_ERROR_INVALID_SUBPROCESSOR_ID = 21ULL,
    CUDTX_ERROR_SHUTDOWN_INITIATED = 22ULL,
    CUDTX_ERROR_INVALID_PLUGIN = 23ULL,
    CUDTX_ERROR_NOT_SUPPORTED = 24ULL,
    CUDTX_ERROR_MEMORY_INACCESSIBLE = 25ULL,
    CUDTX_ERROR_ALREADY_ALLOCATED = 26ULL,
    CUDTX_ERROR_CUDA = 28ULL,
    CUDTX_ERROR_INTERNAL = 29ULL,
    CUDTX_ERROR_NVML = 30ULL,
    CUDTX_ERROR_MAX_VALUE = 0x7FFFFFFF
} CUDTXresult;

template <typename T>
class ResultOr {
public:
    template <typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, ResultOr>>>
    ResultOr(U &&value)
        : value_(std::forward<U>(value))
    {
    }

    ResultOr(ResultOr &&other) = default;
    ResultOr &operator=(ResultOr &&other) = default;

    ResultOr(const ResultOr &other) = default;
    ResultOr &operator=(const ResultOr &other) = default;

    ResultOr(CUDTXresult result)
        : value_(result)
    {
    }

    T &&operator*() &&
    {
        return std::move(std::get<T>(value_));
    }
    T &operator*() &
    {
        return std::get<T>(value_);
    }

    CUDTXresult result() const
    {
        if (std::holds_alternative<T>(value_)) {
            return CUDTX_SUCCESS;
        }
        return std::get<CUDTXresult>(value_);
    }

    bool ok() const
    {
        return std::holds_alternative<T>(value_);
    }

private:
    std::variant<T, CUDTXresult> value_;
};
