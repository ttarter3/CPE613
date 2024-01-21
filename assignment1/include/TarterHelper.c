#include <cstdio>

static const char *_getErrorEnum(cudaError_t error) {
		return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const * const func, const char * const file, int const line) {
  if (result) {
    fprintf(stderr, "Error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _getErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkErrors(val) check((val), #val, __FILE__, __LINE__)

void checkKernelError() {

  // always check for synchronous errors from kernel launch
  checkErrors(cudaGetLastError());
  
  // only wat to do cuda device synchronize when we're debugging DISABLE before release
  #ifndef NDEBUG
    checkErrors(cudaDeviceSynchronize());
  #endif

}