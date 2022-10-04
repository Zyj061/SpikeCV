#ifndef SPSLIB_DEV_REF_COUNT_H_
#define SPSLIB_DEV_REF_COUNT_H_

#include <atomic>

#include "spikelinkapitypes.h"
class RefCount {
  public:

    virtual uint16_t SV_CALLTYPE AddRef();

    virtual uint16_t SV_CALLTYPE Release();

  protected:

    RefCount() {
        refCount_ = 1;
    }

    virtual ~RefCount() {
    }

  private:
    std::atomic<uint16_t> refCount_;
};

#endif // SPSLIB_DEV_REF_COUNT_H_
