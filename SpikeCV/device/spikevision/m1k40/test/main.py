from ast import arg
import ctypes
import sys
sys.path.append('./')
from sdk import spikelinkapi as link
import threading
import time
import argparse
import struct
import os

input = link.spikelinkInput("./sdk/lib/Debug/spikelinkapi.dll")
framepool = link.spikeframepool()
count = 1
brunning = True
DEBUG_OUT = False
SAVE_DAT_FILE = True
SAVE_DAT_FILENAME = "test.dat"

def timer() :
    global brunning
    while brunning :
        if count > 1000 :
            brunning = False
            break
        if framepool.size() > 0 :
            frame = framepool.pop()
            if SAVE_DAT_FILE :
                frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
                spkdata = frame2.contents.data[0]
                CharArr = ctypes.c_char * frame2.contents.size
                char_arr = CharArr(*spkdata[:frame2.contents.size])
                with open(SAVE_DAT_FILENAME, 'ab') as f: # append binary
                    f.write(char_arr.raw)
                # save yaml
                with open(SAVE_DAT_FILENAME[:-4] + ".yaml", 'w') as f: # write text
                    f.write("filename: {:s}".format(SAVE_DAT_FILENAME))
                    f.write("size: {:d}".format(frame2.contents.size))
                    f.write("width: {:d}".format(frame2.contents.width))
                    f.write("height: {:d}".format(frame2.contents.height))
                    # f.write("pts: {:d}".format(frame2.contents.pts))
                    # f.write("dts: {:d}".format(frame2.contents.dts))
                    # f.write("duration: {:d}".format(frame2.contents.duration))

            if DEBUG_OUT :
                frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
                spkdata = frame2.contents.data[0]
                CharArr = ctypes.c_char * frame2.contents.size
                char_arr = CharArr(*spkdata[:frame2.contents.size])
                print("Data: {:}".format(char_arr.raw)) 
                print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height, frame2.contents.pts)
            input.releaseFrame(frame)
        else :
            time.sleep(0.01)

def inputcallback(frame) :
    global count
    if not brunning or count > 1000 :
        return

    framepool.push(frame)
    if DEBUG_OUT :   
        print(frame)
        frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
        print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height, frame2.contents.pts)
    if count % 100 == 0 :
        frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
        print("index:", frame2.contents.pts)
    count += 1
    # input.releaseFrame(frame)

def savedonecallback() :
    print("save finished")

input_callback = link.LinkInputCallBack(inputcallback)
save_callback = link.SaveDoneCallBack(savedonecallback)

if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=int, default=1, help='link input type')
    parser.add_argument('-fn', '--filename', type=str, default='', help='file name of the dummy data')
    parser.add_argument('-dn', '--devname', type=str, default='', help='dev name of the camera link')
    args = parser.parse_args()
    print('args', args)

    filename = args.filename

    input.linkinputlib.ReleaseFrame.argtypes = [ctypes.c_void_p,ctypes.c_void_p]

    params = link.SpikeLinkInitParams()
    params2 = link.SpikeLinkQSFPInitParams()
    params3 = link.SpikeLinkDummyInitParams()
    picture = link.SVPicture()
    picture.width = 1000
    picture.height = 1000
    picture.format = 0x00010000
    picture.fps.num = 20000
    picture.fps.den = 1
    params.type = 0x00000024
    params.mode = 0x00002000
    params.format = 0x00010000
    params.picture = picture
    params.buff_size = 30
    params.cusum = 400
    params2.devName = b"./libhda100.so"
    params2.channels = 0
    params2.channelMode = 0

    params3.fps.num = 20000
    params3.fps.den = 1
    params3.duration = 0
    params3.skip = 0
    params3.start = 0
    params3.end = 0
    params3.repeat = 0
 #   params3.fileName = b"/home/spike/work/wrj.bin"
    params3.fileName = b"F://wurenji//wurenji_20220526//wrj.bin"

    
    if args.type ==  0 :           
        ''' dummy camera source '''
        params.type = 0x00000024
        params3.fileName = bytes(args.filename, 'utf-8')
        params.opaque = ctypes.cast(ctypes.byref(params3), ctypes.c_void_p)
    else :
        ''' Vidar camera source '''
        params.type = 0x00000020
        params2.devName = bytes(args.devname, 'utf-8')
        params.opaque = ctypes.cast(ctypes.byref(params2), ctypes.c_void_p)

    readthrd = threading.Thread(target=timer)
    readthrd.start()

    input.init(ctypes.byref(params))
    input.setcallback(input_callback)
    input.open()
    input.start()
    input.saveFile("/home/spike/Work/data/test_save", 400*200, save_callback)      

    readthrd.join()
    brunning = False

    input.stop()
    input.close()