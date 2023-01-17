
#include <atomic>
#include <cassert>
#include <chrono>
#include <deque>
#include <iostream>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "spikelinkapi.h"
#include "refcount.h"

using namespace std;

namespace spslib_test {

#define SPS_BLOCK_SIZE 10
int32_t index = 0;

#if SPIKECPLUSPLUS
class SPSLibTest : public RefCount, public ISpikeLinkInputObserver {
  public:
    SPSLibTest() : input(nullptr) {
    }
    ~SPSLibTest() {
        Fini();
    }
    int32_t Init(SpikeLinkInitParams* initParams) {
        assert(initParams != nullptr);

        if (initParams == nullptr) {
            return (-1);
        }

        input.reset(CreateSpikeLinkInput(initParams, this));
        ofs.open("//home/spike/work/data/test_out.dat", ios::out | ios::binary);
        return (0);
    }
    void Fini() {
        Stop();
    }
    bool IsOpen() {
        if (input == nullptr) {
            return false;
        }

        return input->IsOpen();
    }
    int32_t Open() {
        if (input == nullptr) {
            return (-1);
        }

        return input->Open();
    }
    int32_t Close() {
        if (input != nullptr) {
            input->Close();
        }

        if (ofs.is_open()) {
            ofs.close();
        }

        return (0);
    }
    int32_t Start() {
        if (input == nullptr) {
            return (-1);
        }

        return input->Start();
    }
    int32_t Stop() {
        if (input == nullptr) {
            return (-1);
        }

        return input->Stop();
    }
    int32_t GetState() {
        if (input == nullptr) {
            return (-1);
        }

        return input->GetState();
    }

    virtual void SV_CALLTYPE OnReceive(SpikeLinkVideoFrame* frame, int devId) {
        if (frame == nullptr) {
            return;
        }

        if(index++ < SPS_BLOCK_SIZE) {
            ofs.write((char*)frame->data[0], frame->size);
        }
        cout << "frame index :" << frame->pts << ", size:" << frame->size << endl;
    }

    virtual void SV_CALLTYPE OnReceive(SpikeLinkVideoPacket* packet, int devId) {
    }

    virtual uint16_t SV_CALLTYPE AddRef() {
        const uint16_t refCount = RefCount::AddRef();
        return (refCount);
    }

    virtual uint16_t SV_CALLTYPE Release() {
        const uint16_t refCount = RefCount::Release();
        return (refCount);
    }

  protected:
  private:
    unique_ptr<ISpikeLinkInput> input;
    std::ofstream ofs;
};

#else
class SPSLibTest {
  public:
    SPSLibTest() : input(nullptr) {
    }
    ~SPSLibTest() {
        Fini();
    }
    int32_t Init(SpikeLinkInitParams* initParams) {
        assert(initParams != nullptr);

        if (initParams == nullptr) {
            return (-1);
        }

        input = SpikeLinkInput::CreateSpikeLinkInputPython();
        ofs.open("//home/spike/work/data/test_out2.dat", ios::out | ios::binary);
        if(SpikeLinkInput::Init(input, initParams) != 0) {
            return (-1);
        }
        return (0);
    }
    void Fini() {
        Stop();
    }
    bool IsOpen() {
        if (input == nullptr) {
            return false;
        }

        return SpikeLinkInput::IsOpen(input);
    }
    int32_t Open() {
        if (input == nullptr) {
            return (-1);
        }

        return SpikeLinkInput::Open(input);
    }
    int32_t Close() {
        if (input != nullptr) {
            SpikeLinkInput::Close(input);
        }

        if (ofs.is_open()) {
            ofs.close();
        }

        return (0);
    }
    int32_t Start() {
        if (input == nullptr) {
            return (-1);
        }

        return SpikeLinkInput::Start(input);
    }
    int32_t Stop() {
        if (input == nullptr) {
            return (-1);
        }

        return SpikeLinkInput::Stop(input);
    }
    int32_t GetState() {
        if (input == nullptr) {
            return (-1);
        }

        return SpikeLinkInput::GetState(input);
    }

    void ReleaseFrame(void *frame) {
        if(input == nullptr) {
            return;
        }

        return SpikeLinkInput::ReleaseFrame(input, frame);
    }

    void SetCallBack(InputCallBack callback) {
        if(input == nullptr) {
            return;
        }

        return SpikeLinkInput::SetCallbackPython(input, callback);
    }

    void SaveFile(int8_t* path, int64_t nFrame, SaveDoneCallBack callback) {
        if(input == nullptr) {
            return;
        }

        return SpikeLinkInput::SaveFile(input, path, nFrame, callback);
    }
public:
    void* input;
    std::ofstream ofs;
};
#endif

void show_help() {
    cout << "Usage: " << endl;
    cout << "      <command> [option] [value]" << endl;
    cout << "      options:" << endl;
    cout << "             -h --help\t show this help" << endl;
    cout << "             -t --type the input device interface type: PCI, USB, Ethnet" << endl;
    cout << "             -pix_fmt --pixelFormat the input pixel format" << endl;
    cout << "             -s --size  input image width and height, 1000x1000" << endl;
    cout << "             -buff_size --buffer_size  receive frame buffer size" << endl;
    cout << "             -r --rate frame per second" << endl;
}

int parse_args(int argc, char* argv[], SpikeLinkInitParams* params) {
    if (argc < 1) {
        show_help();
        return (-1);
    }

    SpikeLinkDummyInitParams* fileParams = (SpikeLinkDummyInitParams*)params->opaque;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            show_help();
            return 1;
        } else if (arg == "-t" || arg == "--type") {
            params->type = (SVDeviceInterface)atoi(argv[++i]);
        } else if (arg == "-pix_fmt" || arg == "--pixel_format") {
            params->picture.format = (SVPixelFormat)atoi(argv[++i]);
        } else if (arg == "-s" || arg == "--size") {
            string imgSize = argv[++i];
            size_t index = imgSize.find('x');
            params->picture.width = atoi(imgSize.substr(0, index).c_str());
            params->picture.height = atoi(imgSize.substr(index + 1).c_str());
        } else if (arg == "-i" || arg == "--input") {
            snprintf(fileParams->fileName, 256, "%s", argv[++i]);
        } else if (arg == "-buff_size" || arg == "--buffer_size") {
            params->buff_size = atoi(argv[++i]);
        } else if (arg == "-r" || arg == "--rate") {
            params->picture.fps = {atoi(argv[++i]), 1};
        } else {
            cout << "unknown option " << arg.c_str() << endl;
            show_help();
            return 1;
        }
    }

    return (0);
}
}

using namespace spslib_test;

unique_ptr<spslib_test::SPSLibTest> test(new spslib_test::SPSLibTest);

void input_callback(void* frame) {
    SpikeLinkVideoFrame *frm = (SpikeLinkVideoFrame*)frame;
    cout << frm->pts << endl;

    if(spslib_test::index++ < SPS_BLOCK_SIZE) {
        test->ofs.write((char*)frm->data[0], frm->size);
    }

    test->ReleaseFrame(frame);
}

void save_callback() {
    cout << "save finished" << endl;
}

int main(int argc, char *argv[])
{
    unique_ptr<SpikeLinkInitParams> params(new SpikeLinkInitParams());
//    unique_ptr<SPSFileInitParams> fileParams(new SPSFileInitParams());
    unique_ptr<SpikeLinkQSFPInitParams> fileParams(new SpikeLinkQSFPInitParams());
    snprintf((char*)fileParams->devName, 256, "%s", "./libhda100.so");
    params->opaque = fileParams.get();

    if (parse_args(argc, argv, params.get()) != 0) {
        return (-1);
    }
    params->type = svDeviceInterfacePCI;
    params->picture.format = svFormat1BitGray;
    params->picture.width = 1024;
    params->picture.height = 1000;
    params->buff_size = 30;
    params->cusum = 400;

    if (test->Init(params.get()) != 0) {
        return (-1);
    }

    test->SetCallBack(input_callback);

    test->Open();

    if (!test->IsOpen()) {
        return (1);
    }

    test->Start();
    test->SaveFile((int8_t*)"/home/spike/Work/data/test_save", 400*200, save_callback);
    getchar();
    test->Stop();
    test->Close();
}
