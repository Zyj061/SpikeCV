#ifndef SV_SPIKELINKAPI_H_
#define SV_SPIKELINKAPI_H_

/* SpikeLink API */
#include <stdint.h>
#include <string>
#include "spikelinkapitypes.h"

// Type Declarations
/* Enum SVDeviceInterface - Device interface type */

typedef uint32_t SVDeviceInterface;
enum _SVDeviceInterface {
    svDeviceInterfacePCI                                        = /* 'pci ' */ 0x00000020,
    svDeviceInterfaceUSB                                        = /* 'usb ' */ 0x00000021,
    svDeviceInterfaceThunderbolt                                = /* 'thun' */ 0x00000022,
    svDeviceInterfaceEthnet                                     = /* 'eth'  */ 0x00000023,
    svDeviceInterfaceDummy                                      = /* 'dum'  */ 0x00000024,
    svDeviceInterfaceUnspecified                            
};

/* Enum SVDisplayMode - SVDisplayMode enumerates the video modes supported. */

typedef uint32_t SVDisplayMode;
enum _SVDisplayMode {
    /* SD Modes K10:400x250*/
    svMODEK10p20000                                             = /* '20000ps' */ 0x00001000,
    svMODEK10p40000                                             = /* '40000ps' */ 0x00001001,

    /* 1K Modes M1:1000x1000*/
    svMODEM1p20000                                              = /* '20000ps' */ 0x00002000,
    svMODEM1p40000                                              = /* '40000ps' */ 0x00002001,

    /* 2K Modes M2:2000x1000*/
    svMODEM2p20000                                              = /* '20000ps' */ 0x00003000,
    svMODEM2p40000                                              = /* '40000ps' */ 0x00003001,

    /* Special Modes */
    svModeUnknown                                               = /* 'iunk' */    0x00008000
};

/* Enum SVPixelFormat - Video pixel formats supported for input */

typedef uint32_t SVPixelFormat;
enum _SVPixelFormat {
    svFormatUnspecified                                         = 0,
    svFormat1BitGray                                            = 0x00010000,
    svFormat8BitGray                                            = 0x00010001,
    svFormat16BitGray                                           = 0x00010002,
    svFormat8BitYUV                                             = 0x00010003,
    svFormat10BitYUV                                            = 0x00010004,
    svFormat8BitRGB                                             = 0x00010005,
    svFormat10BitRGB                                            = 0x00010006,
    svFormat12BitRGB                                            = 0x00010007,
    svFormat8BitARGB                                            = 0x00010008
};

#if defined(__cplusplus)

//#pragma pack(push,1)

typedef struct SVRational {
    int64_t num; ///< Numerator
    int64_t den; ///< Denominator
} SVRational;

typedef struct SVPicture {  
    int32_t        width;
    int32_t        height;
    SVPixelFormat  format;
    SVRational     fps;
    SVRational     time_base;
} SPSPicture;

/* SpikeLink initialization params */

typedef struct SpikeLinkInitParams {
        void*               opaque;
    SVDeviceInterface   type;
    SVDisplayMode       mode;
    SVPixelFormat       format;
    int32_t             buff_size;  // number of cached frames
    SVPicture           picture;
} SPSDevInitParams;

typedef struct SpikeLinkInitParams2 {
    SVDeviceInterface   type;
    SVDisplayMode       mode;
    SVPixelFormat       format;
    int32_t             width;
    int32_t             height;
    int32_t             buff_size;  // number of cached frames
    void*               opaque;
} SPSDevInitParams2;


/* SpikeLink  PCIe(QSFP40) input device  */

typedef struct SpikeLinkQSFPInitParams {
    int8_t          devName[256];
    int32_t         channels;   //0~3
    int32_t         channelMode;//通道操作模式:0=只读取；1=只发送；2=读写模式
} SpikeLinkQSFPInitParams;

/* SpikeLink Dummy input device */

typedef struct SpikeLinkDummyInitParams {
    char            fileName[256];
    SVRational      fps;
    int64_t         duration;
    int32_t         skip;
    int32_t         start;
    int32_t         end;
    bool            repeat;
} SpikeLinkDummyInitParams;

/* struct SpikeLinkVideoFrame - Interface to encapsulate a video frame */

typedef struct SpikeLinkVideoFrame {
    uint8_t* data[NUM_DATA_POINTERS];
    int32_t linesize[NUM_DATA_POINTERS];
    int32_t format;
    int32_t size;
    int32_t width;
    int32_t height;
    int64_t pts;
    int64_t dts;
    int64_t duration;
    SVRational time_base;
    void* opaque;
} SpikeLinkVideoFrame;

typedef struct SpikeLinkVideoPacket {

}SpikeLinkVideoPacket;

//#pragma pack(pop)

// Forward Declarations
class ISpikeLinkInputObserver;
class ISpikeLinkInput;

/* Interface ISpikeLinkInputObserver - Frame completion callback. */

class SV_API ISpikeLinkInputObserver {
  public:
    virtual void SV_CALLTYPE OnReceive(SpikeLinkVideoFrame* frame, int devId) = 0;
    virtual void SV_CALLTYPE OnReceive(SpikeLinkVideoPacket* packet, int devId) = 0;

    virtual uint16_t SV_CALLTYPE AddRef() = 0;

    virtual uint16_t SV_CALLTYPE Release() = 0;
};

/* Interface ISpikeLinkInput - Interface to encapsulate a spikelink input device */
extern "C" {
//typedef void(*InputCallBack2)(uint8_t* data, int32_t size, int32_t width, int32_t height, int64_t pts);
typedef void(*InputCallBack)(void *frame);
}

class SV_API ISpikeLinkInput {
  public:
  #if SPIKECPLUSPLUS
    virtual void Fini() = 0;
    virtual int32_t Open() = 0;
    virtual int32_t Close() = 0;
    virtual bool IsOpen() = 0;
    virtual int32_t Start() = 0;
    virtual int32_t Stop() = 0;
    virtual int32_t GetState() = 0;
    virtual uint16_t SV_CALLTYPE AddRef() = 0;
    virtual uint16_t SV_CALLTYPE Release() = 0;
    virtual void SetCallback(ISpikeLinkInputObserver *obsver) = 0;
  #else

#endif 
};

/* Functions */

extern "C" {

namespace SpikeLinkInput {
/* Called by c or c++*/    
SV_API ISpikeLinkInput* SV_CALLTYPE CreateSpikeLinkInput(SpikeLinkInitParams *params, ISpikeLinkInputObserver *obsver);
SV_API void SV_CALLTYPE DeleteSpikeLinkInput(ISpikeLinkInput *input);

/* Called by python */
SV_API void* SV_CALLTYPE CreateSpikeLinkInputPython();
SV_API int32_t SV_CALLTYPE Init(void *input, SpikeLinkInitParams *params);
SV_API void SV_CALLTYPE SetCallbackPython(void *input, InputCallBack callback);
SV_API bool SV_CALLTYPE IsOpen(void *input);
SV_API int32_t SV_CALLTYPE Open(void *input);
SV_API int32_t SV_CALLTYPE Close(void *input);
SV_API int32_t SV_CALLTYPE Start(void *input);
SV_API int32_t SV_CALLTYPE Stop(void *input);
SV_API int32_t SV_CALLTYPE GetState(void *input);
SV_API void SV_CALLTYPE Fini(void *input);
SV_API void SV_CALLTYPE ReleaseFrame(void* input, void* frame);

} // SpikeLinkInput

} //extern "C"


#endif /* defined(__cplusplus) */
#endif // SV_SPIKELINKAPI_H_
