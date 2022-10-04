#ifndef SV_SPIKELINKAPITYPES_H_
#define SV_SPIKELINKAPITYPES_H_

#include <stdint.h>

#define NUM_DATA_POINTERS 3

#define SV_CALLTYPE_VCC __stdcall
#define SV_EXPORT_VCC   __declspec(dllexport)
#define SV_IMPORT_VCC

#define SV_CALLTYPE_GCC
#define SV_EXPORT_GCC   __attribute__((visibility("default")))
#define SV_IMPORT_GCC

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__CYGWIN__)
#if !defined(SV_CALLTYPE)
#define SV_CALLTYPE SV_CALLTYPE_VCC
#endif
#if !defined(SV_EXPORT)
#define SV_EXPORT   SV_EXPORT_VCC
#endif
#if !defined(SV_IMPORT)
#define SV_IMPORT   SV_IMPORT_VCC
#endif
#else
#if !defined(SV_CALLTYPE)
#define SV_CALLTYPE SV_CALLTYPE_GCC
#endif
#if !defined(SV_EXPORT)
#define SV_EXPORT   SV_EXPORT_GCC
#endif
#if !defined(SV_IMPORT)
#define SV_IMPORT   SV_IMPORT_GCC
#endif
#endif

#if defined(SV_LIB)
#define SV_API
#elif defined(SV_EXPORTS)
#define SV_API SV_EXPORT
#else
#define SV_API SV_IMPORT
#endif

// Type Declarations

typedef int64_t SVTimeValue;
typedef int64_t SVTimeScale;


#endif // SPSLIB_DEF_H_
