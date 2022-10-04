#ifndef SV_SPIKELINKINPUT_H_
#define SV_SPIKELINKINPUT_H_

#include "spikelinkapi.h"
#include "refcount.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <fstream>

#include "common.h"

#define SV_FRAME_BUFF_SIZE     30
#define SV_Block_SIZE          64
#define SV_M1K40_WIDTH         1024
#define SV_MB_SIZE             1024*1024

using namespace std;

//struct DevInf_t;
static InputCallBack callback_ = nullptr;
class LoadLibRAII;
enum SpikeLinkInputState {
    EXCEPTION,
    CLOSED,
    OPENED,
    STARTED,
    UNKNOWN
};

typedef s32_t (*SpikeLinkDevOpen)(DevInf_t* defInfp, s32_t flag);
typedef s32_t (*SpikeLinkDevClose)(void);
typedef s32_t (*SpikeLinkDevOpenCh)(u32_t chx, s08_t mode);
typedef s32_t (*SpikeLinkDevCloseCh)(u32_t chx);
typedef s32_t (*SpikeLinkDevSendMBAlloc)(u32_t chx, MB_Des_t *PMB);
typedef s32_t (*SpikeLinkDevReadMBFree)(u32_t chx, MB_Des_t *PMB);
typedef s32_t (*SpikeLinkDevMBRead)(u32_t chx, MB_Des_t *PMB);
typedef s32_t (*SpikeLinkDevMBSend)(u32_t chx, MB_Des_t *PMB);
typedef struct SpikeLinkDev {
    SpikeLinkDevOpen DevOpen;
    SpikeLinkDevClose DevClose;
    SpikeLinkDevOpenCh DevOpenCh;
    SpikeLinkDevCloseCh DevCloseCh;
    SpikeLinkDevSendMBAlloc DevSendMBAlloc;
    SpikeLinkDevReadMBFree DevReadMBFree;
    SpikeLinkDevMBRead DevMBRead;
    SpikeLinkDevMBSend DevMBSend;
} SpikeLinkDev;

class SpikeFramePool {
  public:
    SpikeFramePool();
    ~SpikeFramePool();

    int32_t Init(int32_t format, int32_t width, int32_t height, int32_t nframe);
    int32_t Init(int32_t frameSize, int32_t nframe);
    void Fini();
    static SpikeLinkVideoFrame* BuildFrame(int32_t format, int32_t width, int32_t height);
    static SpikeLinkVideoFrame* BuildFrame(uint8_t* buff, int32_t size);
    SpikeLinkVideoFrame* PopFrame(bool idle);
    SpikeLinkVideoFrame* Front(bool idle);
    SpikeLinkVideoFrame* At(int index, bool idle);
    void PushFrame(SpikeLinkVideoFrame*, bool idle);
    void Flush();
    int32_t Size();
    static int32_t GetFrameSize(int32_t format, int32_t width, int32_t height);
    static int32_t GetFrameSize(int32_t format, int32_t width, int32_t height, int32_t nchannels);
  protected:

  private:
    int32_t BuildFrameList(int32_t frameSize, int32_t nframe);
    void DistoryFrameList();

  private:
    mutex mtx_;
    unique_ptr<uint8_t[]> framePool_;
    deque<SpikeLinkVideoFrame*> list_;
    deque<SpikeLinkVideoFrame*> list2_;
};

class SpikeLinkBaseInput {
  public:
    SpikeLinkBaseInput();
    ~SpikeLinkBaseInput();
    virtual int32_t Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver);
    virtual int32_t Init(SpikeLinkInitParams *initParams);
    virtual void    Fini();
    virtual int32_t Open() = 0;
    virtual bool    IsOpen();
    virtual int32_t Close() = 0;
    virtual int32_t Start() = 0;
    virtual int32_t Stop() = 0;
    virtual int32_t GetState();
    void ReleaseFrame(void* frame);
    virtual void SetCallback(ISpikeLinkInputObserver *obsver);
    virtual void SetCallbackPython(InputCallBack callback);
  protected:

  private:
    void RecvSpikeThrd();

  protected:
    condition_variable  cond_;
    mutex frmListMtx_;
    mutex mtx_;
    bool bExit_;
    int32_t devId_;
    SpikeLinkInputState state_;
    shared_ptr<ISpikeLinkInputObserver> obsver_;
    unique_ptr<SpikeLinkInitParams> initParams_;
    unique_ptr<SpikeLinkInitParams2> initParams2_;
    unique_ptr<SpikeFramePool> framePool_;
  
  private:
    unique_ptr<thread> recvThrd_;

};

#ifdef __GNUC__
/**
 * Class SPS_QSFP Define
 *  The Capture interface of the Spikesee Cameras that is to obtain images from the QSFP 40G interface
*/
class SpikeLinkQSFP : public SpikeLinkBaseInput {
  public:
    static SpikeLinkQSFP* CreateInstance();
    virtual int32_t Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver);
    virtual int32_t Init(SpikeLinkInitParams *initParams);
    virtual void    Fini();
    virtual int32_t Open();
    virtual int32_t Close();
    virtual int32_t Start();
    virtual int32_t Stop();
    SpikeLinkQSFP();
    virtual ~SpikeLinkQSFP();
  protected:
  private:

    void ReadSpikeThrd();
    void DecodeThrd();

    int32_t InitLinkDev();
  private:
    unique_ptr<DevInf_t> devInfo_;
    unique_ptr<SpikeLinkQSFPInitParams> params_;
    unique_ptr<SpikeFramePool> dataPool_;
    unique_ptr<std::thread> readThrd_;
    unique_ptr<std::thread> decodeThrd_;
    int32_t frameSize_;
    unique_ptr<LoadLibRAII> loadLib_;
    unique_ptr<SpikeLinkDev> linkDev_;
};
#endif

class SpikeLinkDummy : public SpikeLinkBaseInput {
  public:
    static SpikeLinkDummy* CreateInstance();
    virtual int32_t Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver);
    virtual int32_t Init(SpikeLinkInitParams *initParams);
    virtual void    Fini();
    virtual int32_t Open();
    virtual int32_t Close();
    virtual int32_t Start();
    virtual int32_t Stop();

    virtual ~SpikeLinkDummy();
  protected:
  private:
    SpikeLinkDummy();

    void ReadSpikeThrd();
  private:
    unique_ptr<SpikeLinkDummyInitParams> params_;
    unique_ptr<thread> readThrd_;
    string filename_;
    ifstream ifs_;
};

class SpikeLinkInputAdapter : public ISpikeLinkInput, public RefCount {
  public:
    static SpikeLinkInputAdapter* CreateInstance();
    int32_t Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver);
    int32_t Init(SpikeLinkInitParams *initParams);
    void Fini();
    bool IsOpen();
    int32_t Open();
    int32_t Close();
    int32_t Start();
    int32_t Stop();
    int32_t GetState();
    uint16_t SV_CALLTYPE AddRef();
    uint16_t SV_CALLTYPE Release();
    void ReleaseFrame(void* frame);
    void SetCallback(ISpikeLinkInputObserver *obsver);
    void SetCallbackPython(InputCallBack callback);
  protected:

  private:
    SpikeLinkInputAdapter();
    ~SpikeLinkInputAdapter();
};

#endif // SPIKELINKINPUT_H
