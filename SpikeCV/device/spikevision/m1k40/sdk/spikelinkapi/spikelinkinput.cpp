#include "spikelinkinput.h"

#include <cassert>
#include <chrono>
#if (_MSC_VER >= 1920)
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include <string.h>
#include <iostream>

#include "loadlib.h"

#if (_WIN32 | __GNUC__)
#if (_MSC_VER >= 1920)
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif
#else
#pragma warning("unsupported platform, please check~!!!")
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Class SPSFramePool

SpikeFramePool::SpikeFramePool() : framePool_(nullptr) {
}

SpikeFramePool::~SpikeFramePool() {
    Fini();
}

int32_t SpikeFramePool::GetFrameSize(int32_t format, int32_t width, int32_t height) {
    int32_t frmSize = 0;

    switch (format) {
        case svFormat1BitGray:
            frmSize = width * height >> 3;
            break;

        case svFormat8BitGray:
            frmSize = width * height;
            break;

        case svFormat16BitGray:
            frmSize = width * height * 2;
            break;

        case svFormat8BitRGB:
            frmSize = width * height << 3;
            break;

        default:
            break;
    }

    return frmSize;
}

int32_t GetFrameSize(int32_t format, int32_t width, int32_t height, int32_t nchannels) {
    int32_t frmSize = 0;
    switch (format) {
        case svFormat1BitGray:
            frmSize = width * height >> 3;
            break;

        case svFormat8BitGray:
            frmSize = width * height;
            break;

        case svFormat16BitGray:
            frmSize = width * height * 2;
            break;

        case svFormat8BitRGB:
            frmSize = width * height << 3;
            break;

        default:
            break;
    }

    frmSize *= nchannels;

    return frmSize;
}

int32_t SpikeFramePool::Init(int32_t format, int32_t width, int32_t height, int32_t nframe) {
    assert(format >= 0);
    assert(width > 0);
    assert(height > 0);
    assert(nframe > 0);

    if (format < 0 || width <= 0 || height <= 0 || nframe <= 0) {
        return (-1);
    }

    int32_t rtv = -1;
    {
        unique_lock<mutex> lock(mtx_);

        if (framePool_ != nullptr) {
            DistoryFrameList();
            framePool_.reset(nullptr);
        }

        int32_t frameSize = GetFrameSize(format, width, height);

        if (frameSize <= 0) {
            return (-1);
        }

        rtv = BuildFrameList(frameSize, nframe);
    }
    return (rtv);
}

int32_t SpikeFramePool::Init(int32_t frameSize, int32_t nframe) {
    assert(frameSize > 0);
    assert(nframe > 0);

    if (frameSize <= 0 || nframe <= 0) {
        return (-1);
    }

    int32_t rtv = -1;
    {
        unique_lock<mutex> lock(mtx_);

        if (framePool_ != nullptr) {
            DistoryFrameList();
            framePool_.reset(nullptr);
        }

        rtv = BuildFrameList(frameSize, nframe);
    }
    return (0);
}

void SpikeFramePool::Fini() {
    {
        unique_lock<mutex> lock(mtx_);
        DistoryFrameList();
    }
}

int32_t SpikeFramePool::Size() {
    {
        unique_lock<mutex> lock(mtx_);
        return (int32_t)list2_.size();
    }
}

SpikeLinkVideoFrame* SpikeFramePool::BuildFrame(int32_t format, int32_t width, int32_t height) {
    int32_t frameSize = GetFrameSize(format, width, height);

    if (frameSize <= 0) {
        return (nullptr);
    }

    SpikeLinkVideoFrame* frame = BuildFrame(nullptr, frameSize);

    if (frame != nullptr) {
        frame->format = format;
        frame->width = width;
        frame->height = height;
    }

    return (frame);
}

SpikeLinkVideoFrame* SpikeFramePool::BuildFrame(uint8_t* buff, int32_t size) {
    SpikeLinkVideoFrame* frame = new SpikeLinkVideoFrame();

    if (buff != nullptr) {
        frame->data[0] = buff;
    } else {
        frame->data[0] = new uint8_t[size];
    }

    frame->size = size;
    return (frame);
}

SpikeLinkVideoFrame* SpikeFramePool::PopFrame(bool idle) {
    SpikeLinkVideoFrame* frame = nullptr;
    {
        unique_lock<mutex> lock(mtx_);
        if(idle) {
            if (list_.size() <= 0) {
                return (nullptr);
            }

            frame = list_.front();
            list_.pop_front();
        } else {
            if (list2_.size() <= 0) {
                return (nullptr);
            }

            frame = list2_.front();
            list2_.pop_front();
        }
    }
    return frame;
}

SpikeLinkVideoFrame* SpikeFramePool::Front(bool idle){
    SpikeLinkVideoFrame* frame = nullptr;
    {
        unique_lock<mutex> lock(mtx_);
        if(idle) {
            if (list_.size() <= 0) {
                return (nullptr);
            }

            frame = list_.front();
        } else {
            if (list2_.size() <= 0) {
                return (nullptr);
            }

            frame = list2_.front();
        }
    }
    return frame;
}

SpikeLinkVideoFrame* SpikeFramePool::At(int32_t index, bool idle) {
    SpikeLinkVideoFrame* frame = nullptr;
    {
        unique_lock<mutex> lock(mtx_);
        if(idle) {
            if (list_.size() <= 0) {
                return (nullptr);
            }

            frame = list_.at(index);
        } else {
            if (list2_.size() <= 0) {
                return (nullptr);
            }

            frame = list2_.at(index);
        }
    }
    return frame;
}

void SpikeFramePool::PushFrame(SpikeLinkVideoFrame* frame, bool idle) {
    if (frame == nullptr) {
        return;
    }

    {
        unique_lock<mutex> lock(mtx_);
        if(idle) {
            deque<SpikeLinkVideoFrame*>::iterator iter = list_.begin();
            for(; iter != list_.end();iter++) {
                if(*iter == frame){
                    cout << "double push" << endl;
                }
            }
            list_.push_back(frame);
        } else {
            list2_.push_back(frame);
        }
    }
}

void SpikeFramePool::Flush() {
    SpikeLinkVideoFrame* frame = nullptr;
    while((frame = PopFrame(false)) != nullptr) {
        PushFrame(frame, true);
    }
}

int32_t SpikeFramePool::BuildFrameList(int32_t frameSize, int32_t nframe) {
    uint64_t buffSize = frameSize * nframe;
    framePool_.reset(new uint8_t[buffSize]);

    if (framePool_.get() == nullptr) {
        return (-1);
    }

    uint8_t* p = framePool_.get();

    for (int i = 0; i < nframe; ++i) {
        SpikeLinkVideoFrame* frame = BuildFrame(p, frameSize);
        list_.push_back(frame);
        p += frameSize;
    }

    return (0);
}

void SpikeFramePool::DistoryFrameList() {
    deque<SpikeLinkVideoFrame*>::iterator iter = list_.begin();

    for (; iter != list_.end(); ) {
        SpikeLinkVideoFrame* frame = *iter;
        iter = list_.erase(iter);
        delete frame;
    }

    iter = list2_.begin();
    for(; iter != list2_.end(); ) {
        SpikeLinkVideoFrame* frame = *iter;
        iter = list2_.erase(iter);
        delete frame;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Class SpikeLinkBaseInput

SpikeLinkBaseInput::SpikeLinkBaseInput() : bExit_(false), devId_(-1), state_(UNKNOWN), initParams_(nullptr),initParams2_(nullptr),
    obsver_(nullptr), framePool_(nullptr), recvThrd_(nullptr) {
}

SpikeLinkBaseInput::~SpikeLinkBaseInput() {
    Fini();
}

int32_t SpikeLinkBaseInput::Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver) {
    if (obsver_ != nullptr || framePool_.get() != nullptr) {
        return (1);
    }

    framePool_.reset(new SpikeFramePool);

    if (framePool_->Init(initParams->picture.format, initParams->picture.width,
                         initParams->picture.height, initParams->buff_size) != 0) {
        return (-1);
    }

    if (initParams_ == nullptr) {
        initParams_.reset(new SPSDevInitParams);
    }

    memcpy(initParams_.get(), initParams, sizeof(SPSDevInitParams));
    obsver_.reset(obsver);
    obsver->AddRef();
    bExit_ = false;
    recvThrd_.reset(new thread(&SpikeLinkBaseInput::RecvSpikeThrd, this));
    return (0);
}

int32_t SpikeLinkBaseInput::Init(SpikeLinkInitParams *initParams) {
    if (obsver_ != nullptr || framePool_.get() != nullptr) {
        return (1);
    }

    framePool_.reset(new SpikeFramePool);

    if (framePool_->Init(initParams->picture.format, initParams->picture.width,
                         initParams->picture.height, initParams->buff_size) != 0) {
        return (-1);
    }

    if (initParams_ == nullptr) {
        initParams_.reset(new SPSDevInitParams);
    }
    memcpy(initParams_.get(), initParams, sizeof(SPSDevInitParams));

    bExit_ = false;
    recvThrd_.reset(new thread(&SpikeLinkBaseInput::RecvSpikeThrd, this));

    return (0);    
}

void SpikeLinkBaseInput::Fini() {
    bExit_ = true;
    {
        cond_.notify_one();
    }

    if (recvThrd_.get() != nullptr) {
        recvThrd_->join();
        recvThrd_.reset();
    }
}

bool SpikeLinkBaseInput::IsOpen() {
    return (state_ == SpikeLinkInputState::OPENED || state_ == SpikeLinkInputState::STARTED );
}

int32_t SpikeLinkBaseInput::GetState() {
    return state_;
}

void SpikeLinkBaseInput::RecvSpikeThrd() {
    SpikeLinkVideoFrame* frame = nullptr;
    while (!bExit_) {
        {
            unique_lock<mutex> lock(mtx_);

            while (framePool_->Size() <= 0 && !bExit_) {
                cond_.wait(lock);
            }
        }

        if (bExit_) {
            break;
        }

        frame = framePool_->PopFrame(false);

        if (frame != nullptr && obsver_ != nullptr) {
            obsver_->OnReceive(frame, 0);
        }

         if (frame != nullptr && callback_ != nullptr) {
            callback_((void*)frame);
        }
    }
}

void SpikeLinkBaseInput::ReleaseFrame(void* frame) {
    {
        unique_lock<mutex> lock(mtx_);
        framePool_->PushFrame((SpikeLinkVideoFrame*)frame, true);
    }
}

void SpikeLinkBaseInput::SetCallback(ISpikeLinkInputObserver *obsver){
    {
        unique_lock<mutex> lock(mtx_);
        obsver_.reset(obsver);
    }
}

void SpikeLinkBaseInput::SetCallbackPython(InputCallBack callback) {
    {
        unique_lock<mutex> lock(mtx_);
        callback_ = callback;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Class SpikeLinkQSFP
#ifdef __GNUC__

SpikeLinkQSFP* SpikeLinkQSFP::CreateInstance() {
    SpikeLinkQSFP* const instance = new SpikeLinkQSFP;
    return (instance);
}

SpikeLinkQSFP::SpikeLinkQSFP(): devInfo_(nullptr), params_(nullptr), dataPool_(nullptr),
    readThrd_(nullptr), loadLib_(nullptr), linkDev_(nullptr) {
}

SpikeLinkQSFP::~SpikeLinkQSFP() {
    Fini();
}

int32_t SpikeLinkQSFP::Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver) {
    assert(initParams != nullptr);
    assert(obsver != nullptr);

    if (initParams == nullptr || initParams->opaque == nullptr || obsver == nullptr) {
        return (-1);
    }

    if (obsver_ != nullptr || dataPool_ != nullptr) {
        return (1);
    }

    {
        unique_lock<mutex> lock(mtx_);

        if (SpikeLinkBaseInput::Init(initParams, obsver) != 0) {
            return (-1);
        }

        dataPool_.reset(new SpikeFramePool);

        if (dataPool_->Init(SV_MB_SIZE, SV_FRAME_BUFF_SIZE) != 0) {
            return (-1);
        }

        if (params_ == nullptr) {
            params_.reset(new SpikeLinkQSFPInitParams());
        }

        memcpy(params_.get(), initParams->opaque, sizeof(SpikeLinkQSFPInitParams));

        if(loadLib_ == nullptr) {
            loadLib_.reset(new LoadLibRAII((char*)params_->devName));

            if(loadLib_ == nullptr) {
                return (-1);
            }
        }

        if(linkDev_ == nullptr) {
            linkDev_.reset(new SpikeLinkDev());

            InitLinkDev();
        }
    }
    return (0);
}


int32_t SpikeLinkQSFP::Init(SpikeLinkInitParams *initParams) {
    assert(initParams != nullptr);
    if (initParams == nullptr) {
        return (-1);
    }
    if (dataPool_ != nullptr) {
        return (1);
    }

    {
        unique_lock<mutex> lock(mtx_);

        if (SpikeLinkBaseInput::Init(initParams) != 0) {
            return (-1);
        }

        dataPool_.reset(new SpikeFramePool);
        if (dataPool_->Init(SV_MB_SIZE, SV_FRAME_BUFF_SIZE) != 0) {
            return (-1);
        }

        if (params_ == nullptr) {
            params_.reset(new SpikeLinkQSFPInitParams());
        }

        memcpy(params_.get(), initParams->opaque, sizeof(SpikeLinkQSFPInitParams));

        if(loadLib_ == nullptr) {
            loadLib_.reset(new LoadLibRAII((char*)params_->devName));

            if(loadLib_ == nullptr) {
                return (-1);
            }
        }

        if(linkDev_ == nullptr) {
            linkDev_.reset(new SpikeLinkDev());

            InitLinkDev();
        }
    }

    return (0);
}

void SpikeLinkQSFP::Fini() {
    Stop();
    {
        unique_lock<mutex> lock(mtx_);
        SpikeLinkBaseInput::Fini();
    }
}

int32_t SpikeLinkQSFP::Open() {
    if (readThrd_ != NULL || IsOpen()) {
        return (1);
    }

    {
        unique_lock<mutex> lock(mtx_);
        devInfo_.reset(new DevInf_t());
        int retv = linkDev_->DevOpen(devInfo_.get(), 0);

        if (retv != 0) {
            return (retv);
        }

        state_ = OPENED;
    }

    return (0);
}

int32_t SpikeLinkQSFP::Close() {
    int32_t retv = -1;
    {
        unique_lock<mutex> lock(mtx_);

        if (!IsOpen()) {
            return (retv);
        }

        if (state_ == SpikeLinkInputState::STARTED) {
            retv = Stop();
        }

        state_ = SpikeLinkInputState::CLOSED;

        if (readThrd_ != nullptr) {
            readThrd_->join();
        }

        if (decodeThrd_ != nullptr) {
            decodeThrd_->join();
        }

        retv = linkDev_->DevClose();
    }
    return (retv);
}

int32_t SpikeLinkQSFP::Start() {
    int retv = -1;

    if (state_ == SpikeLinkInputState::STARTED) {
        return (1);
    }

    {
        unique_lock<mutex> lock(mtx_);

        if (readThrd_ != nullptr || decodeThrd_ != nullptr) {
            return (retv);
        }

        retv = linkDev_->DevOpenCh(0, 0);
        if (retv != 0) {
//            return (retv);
        }

        state_ = STARTED;
        readThrd_.reset(new thread(&SpikeLinkQSFP::ReadSpikeThrd, this));
        decodeThrd_.reset(new thread(&SpikeLinkQSFP::DecodeThrd, this));
    }

    return (0);
}

int32_t SpikeLinkQSFP::Stop() {
    if (state_ != SpikeLinkInputState::STARTED) {
        return (1);
    }

    state_ = OPENED;
    {
        if (readThrd_ != nullptr) {
            readThrd_->join();
            readThrd_.reset(nullptr);
        }

        if (decodeThrd_ != nullptr) {
            decodeThrd_->join();
            decodeThrd_.reset(nullptr);
        }     
    }

    return (0);
}

void SpikeLinkQSFP::ReadSpikeThrd() {
    int32_t err = -1;
    MB_Des_t mb;
    SpikeLinkVideoFrame* frame = dataPool_->PopFrame(true);

    while (state_ == SpikeLinkInputState::STARTED) {
        err = linkDev_->DevMBRead(0, &mb);

        if (err != 0) {
            this_thread::sleep_for(chrono::microseconds(10));
            continue;
        }

        if (mb.pbuf == nullptr || mb.buf_size % SV_Block_SIZE != 0) {
            linkDev_->DevReadMBFree(0, &mb);
            continue;
        }

        if (frame != nullptr) {
            memcpy(frame->data[0], mb.pbuf, mb.buf_size);
            frame->size = mb.buf_size;
            dataPool_->PushFrame(frame, false);
            cond_.notify_one();
            frame = NULL;
        } else {
            printf("Drop frame\n");
        }

        linkDev_->DevReadMBFree(0, &mb);
        frame = dataPool_->PopFrame(true);
    }

    if (frame != nullptr) {
        dataPool_->PushFrame(frame, true);
    }

    cond_.notify_all();
}

static bool findFrameStartCode(uint8_t* pInData, int size, int height, uint8_t** pStart) {
    bool bRet = false;
    int w = SV_M1K40_WIDTH;
    int h = height;
    int lineNum = height / 2;
    uint8_t* p = pInData;
    uint8_t* q = p + size;

    do {
        int chId = (p[(SV_Block_SIZE / 2 - 1)] >> 6) & 0x03;
        int lineId = (((uint16_t*)p)[(SV_Block_SIZE >> 2) - 1] >> 4) & 0x03ff;
        int chId2 = (p[95] >> 6) & 0x03;
        int lineId2 = (((uint16_t*)p)[47] >> 4) & 0x03ff;

        if (chId == 3 && lineId == lineNum && chId2 == 0 && lineId2 == lineNum) {
            p += SV_Block_SIZE / 2;
            *pStart = p;
            bRet = true;
            break;
        }

        p += SV_Block_SIZE / 2;
    } while ((p + SV_Block_SIZE * 3 / 2) < q);

    return (bRet);
}

void SpikeLinkQSFP::DecodeThrd() {
    SpikeLinkVideoFrame* packet = nullptr;
    SpikeLinkVideoFrame* frame = nullptr;
    bool bFoundFreamHead = false;
    int32_t heigth = initParams_->picture.height;
    int32_t frameSize = SpikeFramePool::GetFrameSize(initParams_->picture.format, initParams_->picture.width, initParams_->picture.height);
    int32_t leftSize = 0;
    int64_t pts = 0;
    while (state_ == STARTED) {
        {
            unique_lock<mutex> lock(mtx_);

            while (dataPool_->Size() <= 0 && state_ == STARTED) {
                cond_.wait(lock);
            }
        }
        if (state_ != STARTED) {
            break;
        }

        packet = dataPool_->PopFrame(false);
        uint8_t* p = packet->data[0];
        uint8_t* q = p + packet->size;

        if (!bFoundFreamHead) {
            cout << packet->size << endl;
            if (!(bFoundFreamHead = findFrameStartCode(p, packet->size, heigth, &p))) {
                dataPool_->PushFrame(packet, true);
                packet = nullptr;
                cond_.notify_one();
                continue;
            }

            bFoundFreamHead = true;
        }

        if (frame == nullptr) {
            frame = framePool_->PopFrame(true);
            if(frame == nullptr) {
                dataPool_->PushFrame(packet, true);
                packet = nullptr;
                cond_.notify_one();
                bFoundFreamHead = false;
                cout << "data buff overflow" << endl;
                continue;
            }
            frame->size = 0;
        }

        do {
            leftSize = 0;

            if (frame->size + (q - p) <= frameSize) {
                memcpy(frame->data[0] + frame->size, p, (q - p));
                frame->size += (q - p);
            } else {
                memcpy(frame->data[0] + frame->size, p, frameSize - frame->size);
                p += (frameSize - frame->size);
                frame->size = frameSize;
                leftSize = (q - p);
            }

            if (frame->size < frameSize) {
                break;
            }

            frame->width = initParams_->picture.width;
            frame->height = initParams_->picture.height;
            frame->pts = frame->dts = pts++;
            framePool_->PushFrame(frame, false);
            frame = nullptr;
            cond_.notify_one();

            frame = framePool_->PopFrame(true);
            if(frame != nullptr) {
                frame->size = 0;
            } else {
                bFoundFreamHead = false;
                cout << "frame buff overflow" << endl;
                break;
            }
        } while (leftSize > 0);

        dataPool_->PushFrame(packet, true);
        cond_.notify_one();
        packet = nullptr;
    }

    if (packet != nullptr) {
        dataPool_->PushFrame(frame, true);
    }

    if (frame != nullptr) {
        framePool_->PushFrame(frame, true);
    }
}

int32_t SpikeLinkQSFP::InitLinkDev() {
    linkDev_->DevOpen = (SpikeLinkDevOpen)(loadLib_->GetLibAPI("DevOpen"));
    if(linkDev_->DevOpen == nullptr) {
        return (-1);
    }

    linkDev_->DevClose = (SpikeLinkDevClose)(loadLib_->GetLibAPI("DevClose"));
    if(linkDev_->DevClose == nullptr) {
        return (-1);
    }

    linkDev_->DevOpenCh = (SpikeLinkDevOpenCh)(loadLib_->GetLibAPI("DevOpenCh"));
    if(linkDev_->DevOpenCh == nullptr) {
        return (-1);
    }

    linkDev_->DevCloseCh = (SpikeLinkDevCloseCh)(loadLib_->GetLibAPI("DevCloseCh"));
    if(linkDev_->DevCloseCh == nullptr) {
        return (-1);
    }

    linkDev_->DevSendMBAlloc = (SpikeLinkDevSendMBAlloc)(loadLib_->GetLibAPI("DevSendMBAlloc"));
    if(linkDev_->DevSendMBAlloc == nullptr) {
        return (-1);
    }

    linkDev_->DevReadMBFree = (SpikeLinkDevReadMBFree)(loadLib_->GetLibAPI("DevReadMBFree"));
    if(linkDev_->DevReadMBFree == nullptr) {
        return (-1);
    }

    linkDev_->DevMBRead = (SpikeLinkDevMBRead)(loadLib_->GetLibAPI("DevMBRead"));
    if(linkDev_->DevMBRead == nullptr) {
        return (-1);
    }

    linkDev_->DevMBSend = (SpikeLinkDevMBSend)(loadLib_->GetLibAPI("DevMBSend"));
    if(linkDev_->DevMBSend == nullptr) {
        return (-1);
    }
}

#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Class SPSQSFPInput

SpikeLinkDummy* SpikeLinkDummy::CreateInstance() {
    SpikeLinkDummy* const instance = new SpikeLinkDummy;
    return (instance);
}

SpikeLinkDummy::SpikeLinkDummy() : params_(nullptr), readThrd_(nullptr),  filename_("") {
}

SpikeLinkDummy::~SpikeLinkDummy() {
    Fini();
}

int32_t SpikeLinkDummy::Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver) {
    assert(initParams != nullptr);
    assert(obsver != nullptr);

    if (initParams == nullptr || initParams->opaque == nullptr || obsver == nullptr) {
        return (-1);
    }

    if (obsver_ != nullptr) {
        return (1);
    }

    {
        unique_lock<mutex> lock(mtx_);
        unique_ptr<SpikeLinkDummyInitParams> params((SpikeLinkDummyInitParams*)initParams->opaque);
        filename_ = params->fileName;

        if (filename_.empty() || !fs::exists(filename_)) {
            return (-1);
        }

        if (SpikeLinkBaseInput::Init(initParams, obsver) != 0) {
            return (-1);
        }

        if (params_ == nullptr) {
            params_.reset(new SpikeLinkDummyInitParams());
        }

        memcpy(params_.get(), initParams->opaque, sizeof(SpikeLinkDummyInitParams));
    }

    return (0);
}

int32_t SpikeLinkDummy::Init(SpikeLinkInitParams *initParams) {
    assert(initParams != nullptr);

    if (initParams == nullptr) {
        return (-1);
    }

    {
        unique_lock<mutex> lock(mtx_);
        unique_ptr<SpikeLinkDummyInitParams> params((SpikeLinkDummyInitParams*)initParams->opaque);
        filename_ = params->fileName;

        if (filename_.empty() || !fs::exists(filename_)) {
            return (-1);
        }

        if (SpikeLinkBaseInput::Init(initParams) != 0) {
            return (-1);
        }

        if (params_ == nullptr) {
            params_.reset(new SpikeLinkDummyInitParams());
        }

        memcpy(params_.get(), initParams->opaque, sizeof(SpikeLinkDummyInitParams));
        params.release();
    }

    return (0);
}

void SpikeLinkDummy::Fini() {
    Stop();
    {
        bExit_ = true;
        cond_.notify_one();
        unique_lock<mutex> lock(mtx_);
        SpikeLinkBaseInput::Fini();
    }
}

int32_t SpikeLinkDummy::Open() {
    if (readThrd_ != NULL || IsOpen()) {
        return (1);
    }

    int retv = -1;
    {
        unique_lock<mutex> lock(mtx_);
        ifs_.open(filename_, ios_base::in | ios_base::binary);

        if (!ifs_.is_open()) {
            return (retv);
        }

        state_ = OPENED;
    }
    return (0);
}

int32_t SpikeLinkDummy::Close() {
    int32_t retv = 0;
    {
        unique_lock<mutex> lock(mtx_);

        if (!IsOpen()) {
            return (1);
        }

        if (state_ == SpikeLinkInputState::STARTED) {
            retv = Stop();
        }

        state_ = SpikeLinkInputState::CLOSED;

        if (readThrd_ != nullptr) {
            readThrd_->join();
        }

        ifs_.close();
    }
    return (retv);
}

int32_t SpikeLinkDummy::Start() {
    int retv = -1;

    if (state_ == SpikeLinkInputState::STARTED) {
        return (1);
    }

    {
        unique_lock<mutex> lock(mtx_);

        if (readThrd_ != nullptr) {
            return (retv);
        }

        state_ = STARTED;
        readThrd_.reset(new thread(&SpikeLinkDummy::ReadSpikeThrd, this));
    }

    return (0);
}

int32_t SpikeLinkDummy::Stop() { 
    if (state_ != SpikeLinkInputState::STARTED) {
        return (1);
    }

    {
        unique_lock<mutex> lock(mtx_);
        state_ = SpikeLinkInputState::OPENED;
        if (readThrd_ != nullptr) {
            readThrd_->join();
            readThrd_.reset(nullptr);
        }

        state_ = OPENED;
    }

    return (0);
}



void SpikeLinkDummy::ReadSpikeThrd() {
    SpikeLinkVideoFrame* frame = framePool_->PopFrame(true);

    int32_t frameSize = SpikeFramePool::GetFrameSize(initParams_->picture.format,
                        initParams_->picture.width, initParams_->picture.height);
    int64_t pts = 0;

    while (state_ == SpikeLinkInputState::STARTED) {
        if (ifs_.peek() == EOF || ifs_.eof()) {
            if (params_->repeat) {
                ifs_.seekg(0, ios::beg);
            } else {
                break;
            }
        }

        if (frame != nullptr) {
            ifs_.read((char*)frame->data[0], frameSize);
            frame->size = frameSize;
            frame->width = initParams_->picture.width;
            frame->height = initParams_->picture.height;
            frame->pts = frame->dts = pts++;
            framePool_->PushFrame(frame, false);
            cond_.notify_one();
            frame = NULL;
        } else {
            printf("Drop frame\n");
        }

        frame = framePool_->PopFrame(true);
//        this_thread::sleep_for(chrono::microseconds((params_->fps.den * 1000000) / params_->fps.num));
        this_thread::sleep_for(chrono::milliseconds(2));
    }

    if (frame != nullptr) {
        framePool_->PushFrame(frame, true);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////
static std::unique_ptr<SpikeLinkBaseInput> input_(nullptr);
static int32_t type_ = svDeviceInterfaceUnspecified;
static mutex mutex_;

SpikeLinkInputAdapter* SpikeLinkInputAdapter::CreateInstance() {
    SpikeLinkInputAdapter* const instance = new SpikeLinkInputAdapter;
    return (instance);
}

SpikeLinkInputAdapter::SpikeLinkInputAdapter() {
}

SpikeLinkInputAdapter::~SpikeLinkInputAdapter() {
    Fini();
}

int32_t SpikeLinkInputAdapter::Init(SpikeLinkInitParams *initParams, ISpikeLinkInputObserver *obsver) {
    assert(initParams != nullptr);
    assert(obsver != nullptr);

    if (input_ != nullptr) {
        return (1);
    }

    if (initParams == nullptr || initParams->opaque == nullptr || obsver == nullptr) {
        return (-1);
    }

    int retv = -1;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ != nullptr) {
            return (1);
        }

        switch (initParams->type) {
            case svDeviceInterfacePCI:
#ifdef __GNUC__
                input_.reset(SpikeLinkQSFP::CreateInstance());
                retv = ((SpikeLinkQSFP*)input_.get())->Init(initParams, obsver);
#else
#endif // __GNUC__
                break;

            case svDeviceInterfaceDummy:
                input_.reset(SpikeLinkDummy::CreateInstance());
                retv = ((SpikeLinkDummy*)input_.get())->Init(initParams, obsver);
                break;

            default:
                break;
        }

        type_ = initParams->type;
    }
    return (retv);
}

int32_t SpikeLinkInputAdapter::Init(SpikeLinkInitParams *initParams) {
    assert(initParams != nullptr);

    if (input_ != nullptr) {
        return (1);
    }

    if (initParams == nullptr) {
        return (-1);
    }

    int retv = -1;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ != nullptr) {
            return (1);
        }

        switch (initParams->type) {
            case svDeviceInterfacePCI:
#ifdef __GNUC__
                input_.reset(SpikeLinkQSFP::CreateInstance());
                retv = ((SpikeLinkQSFP*)input_.get())->Init(initParams);
#else
#endif // __GNUC__
                break;

            case svDeviceInterfaceDummy:
                input_.reset(SpikeLinkDummy::CreateInstance());
                retv = ((SpikeLinkDummy*)input_.get())->Init(initParams);
                break;

            default:
                break;
        }

        type_ = initParams->type;
    }
    return (retv);
}

void SpikeLinkInputAdapter::Fini() {
    SpikeLinkBaseInput* input = nullptr;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_.get() != nullptr) {
            input = input_.release();
            input_.reset(nullptr);
        }
    }

    if (input != nullptr) {
        switch (type_) {
            case svDeviceInterfacePCI: {
#ifdef __GNUC__
                unique_ptr<SpikeLinkQSFP> qsfp((SpikeLinkQSFP*)input);
                qsfp->Fini();
#else
#endif //__GNUC__
                break;
            }

            case svDeviceInterfaceDummy: {
                unique_ptr<SpikeLinkDummy> fileInput((SpikeLinkDummy*)input);
                fileInput->Fini();
            }
            break;

            default:
                break;
        }
    }
}

int32_t SpikeLinkInputAdapter::Open() {
    assert(input_ != nullptr);

    if (input_ == nullptr) {
        return (-1);
    }

    int retv = -1;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ == nullptr) {
            return (-1);
        }

        if (input_->IsOpen()) {
            return (0);
        }

        retv = input_->Open();
    }
    return (retv);
}

bool SpikeLinkInputAdapter::IsOpen() {
    assert(input_ != nullptr);

    if (input_ == nullptr) {
        return (false);
    }

    int retv = false;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ == nullptr) {
            return (false);
        }

        retv = input_->IsOpen();
    }
    return (retv);
}

int32_t SpikeLinkInputAdapter::Close() {
    assert(input_ != nullptr);

    if (input_ == nullptr) {
        return (-1);
    }

    int retv = -1;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ == nullptr) {
            return (-1);
        }

        if (!input_->IsOpen()) {
            return (0);
        }

        retv = input_->Close();
    }
    return (retv);
}

int32_t SpikeLinkInputAdapter::Start() {
    assert(input_ != nullptr);

    if (input_ == nullptr) {
        return (-1);
    }

    int retv = -1;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ == nullptr) {
            return (-1);
        }

        retv = input_->Start();
    }
    return (retv);
}

int32_t SpikeLinkInputAdapter::Stop() {
    assert(input_ != nullptr);

    if (input_ == nullptr) {
        return (-1);
    }

    int retv = -1;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ == nullptr) {
            return (-1);
        }

        retv = input_->Stop();
    }
    return (retv);
}

int32_t SpikeLinkInputAdapter::GetState() {
    assert(input_ != nullptr);

    if (input_ == nullptr) {
        return (false);
    }

    int retv = UNKNOWN;
    {
        unique_lock<mutex> lock(mutex_);

        if (input_ == nullptr) {
            return (false);
        }

        retv = input_->GetState();
    }
    return (retv);
}

uint16_t SpikeLinkInputAdapter::AddRef() {
    const int16_t refCount = RefCount::AddRef();
    return (refCount);
}

uint16_t SpikeLinkInputAdapter::Release() {
    const int16_t refCount = RefCount::Release();
    return (refCount);
}

void SpikeLinkInputAdapter::SetCallback(ISpikeLinkInputObserver *obsver){
    assert(obsver != nullptr);
    if(obsver == nullptr) {
        return;
    }

    {
        unique_lock<mutex> lock(mutex_);
        if (input_ == nullptr) {
            return;
        }

        input_->SetCallback(obsver);
    }
}

void SpikeLinkInputAdapter::SetCallbackPython(InputCallBack callback) {
    assert(callback != nullptr);
    if(callback == nullptr) {
        return;
    }

    {
        unique_lock<mutex> lock(mutex_);
        if (input_ == nullptr) {
            return;
        }

        input_->SetCallbackPython(callback);
    }
}

void SpikeLinkInputAdapter::ReleaseFrame(void* frame) {
    assert(input_ != nullptr);

    if (input_ == nullptr || frame == nullptr) {
        return;
    }

    input_->ReleaseFrame(frame);

}
