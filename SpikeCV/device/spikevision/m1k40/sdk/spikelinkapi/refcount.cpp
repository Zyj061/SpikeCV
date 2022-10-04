
#include "refcount.h"

uint16_t SV_CALLTYPE RefCount::AddRef() {
    ++refCount_;
    return refCount_;
}

uint16_t SV_CALLTYPE RefCount::Release() {
    --refCount_;

    if (refCount_ == 0) {
        delete this;
    }

    return refCount_;
}
