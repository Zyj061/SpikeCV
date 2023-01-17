#include "spikelinkapi.h"
#include "spikelinkinput.h"

#include <cassert>
namespace SpikeLinkInput {
SV_API ISpikeLinkInput* SV_CALLTYPE CreateSpikeLinkInput(SpikeLinkInitParams *params, ISpikeLinkInputObserver *obsver) {
    assert(params != NULL);
    assert(obsver != NULL);

    if (params == NULL || obsver == NULL) {
        return (NULL);
    }
   
   ////.so 

    SpikeLinkInputAdapter* instance = SpikeLinkInputAdapter::CreateInstance();

    if (instance->Init(params, obsver) != 0) {
        instance->Release();
        instance = NULL;
    }

    return (instance);
}

SV_API void* SV_CALLTYPE CreateSpikeLinkInputPython() {

    SpikeLinkInputAdapter* instance = SpikeLinkInputAdapter::CreateInstance();

    return (instance);
}

SV_API int32_t SV_CALLTYPE Init(void *input, SpikeLinkInitParams *params) {
    assert(input != nullptr);
    assert(params != nullptr);
    if(input == nullptr || params == nullptr) {
        return (1);
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;

    if (adapter->Init(params) != 0) {
        adapter->Release();

        return (-1);
    }
    return (0);
}


SV_API void SV_CALLTYPE DeleteSpikeLinkInput(void* input) {
    assert(input != nullptr);
    if (input == NULL) {
        return;
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    adapter->Release();
}

SV_API void SV_CALLTYPE SetCallbackPython(void* input, InputCallBack callback) {
    assert(input != nullptr);
    if(input == nullptr) {
        return ;
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    return adapter->SetCallbackPython(callback);
}

SV_API bool SV_CALLTYPE IsOpen(void* input) {
    assert(input != nullptr);
    if(input == nullptr) {
        return (false);
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    return adapter->IsOpen();
}

SV_API int32_t SV_CALLTYPE Open(void* input) {
    assert(input != nullptr);
    if(input == nullptr) {
        return (-1);
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    return adapter->Open();
}

SV_API int32_t SV_CALLTYPE Close(void* input) {
    assert(input != nullptr);
    if(input == nullptr) {
        return (-1);
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    return adapter->Close();
}

SV_API int32_t SV_CALLTYPE Start(void* input) {
    assert(input != nullptr);
    if(input == nullptr) {
        return (-1);
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    return adapter->Start();
}

SV_API int32_t SV_CALLTYPE Stop(void* input) {
    assert(input != nullptr);
    if(input == nullptr) {
        return (-1);
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    return adapter->Stop();
}

SV_API int32_t SV_CALLTYPE GetState(void* input) {
    assert(input != nullptr);
    if(input == nullptr) {
        return (-1);
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    return adapter->GetState();
}

SV_API void SV_CALLTYPE Fini(void *input) {
    assert(input != nullptr);
    if(input == nullptr) {
        return;
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    adapter->Fini();
}

SV_API void SV_CALLTYPE ReleaseFrame(void* input, void* frame) {
    assert(input != nullptr);
    if(input == nullptr) {
        return ;
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    adapter->ReleaseFrame(frame);
}

SV_API void SV_CALLTYPE GetFrames(void* input, SpikeLinkVideoFrame** frames, int32_t *nFrame) {
    assert(input != nullptr);
    if(input == nullptr) {
        return ;
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    adapter->GetFrames(frames, nFrame);
}

SV_API void SV_CALLTYPE SaveFile(void* input, int8_t* filePath, int64_t nFrame, SaveDoneCallBack callback) {
    assert(input != nullptr);
    if(input == nullptr) {
        return;
    }

    SpikeLinkInputAdapter* const adapter = (SpikeLinkInputAdapter*)input;
    adapter->SaveFile(filePath, nFrame, callback);
}

} // end SpikeLinkInput