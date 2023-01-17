from calendar import c
import ctypes
import threading

LinkInputCallBack = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
SaveDoneCallBack = ctypes.CFUNCTYPE(ctypes.c_void_p)
class SVRational(ctypes.Structure) :
    _fields_ = [("num", ctypes.c_int64), 
                ("den", ctypes.c_int64)]

class SpikeLinkVideoFrame(ctypes.Structure) :
    _fields_ = [("data", ctypes.POINTER(ctypes.c_char) * 3), 
                ("linesize", ctypes.c_int32 * 3),
                ("format", ctypes.c_int32),
                ("size", ctypes.c_int32), 
                ("width", ctypes.c_int32), 
                ("height", ctypes.c_int32), 
                ("pts", ctypes.c_int64),
                ("dts", ctypes.c_int64),
                ("duration", ctypes.c_int64),
                ("time_base", SVRational),
                ("opaque", ctypes.c_void_p)]

class SVPicture(ctypes.Structure) :
    _fields_ = [("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("format", ctypes.c_uint32),
                ("fps", SVRational),
                ("time_base", SVRational)]
class SpikeLinkQSFPInitParams(ctypes.Structure) :
    _fields_ = [("devName", ctypes.c_char * 256),
                ("channels", ctypes.c_int32),
                ("channelMode", ctypes.c_int32)]

class SpikeLinkDummyInitParams(ctypes.Structure) :
    _fields_ = [("fileName", ctypes.c_char * 256),
                ("fps", SVRational),
                ("duration", ctypes.c_int64),
                ("skip", ctypes.c_int32),
                ("start", ctypes.c_int32),
                ("end", ctypes.c_int32),
                ("repeat", ctypes.c_bool)]

class SpikeLinkInitParams(ctypes.Structure) :
    _fields_ = [("opaque", ctypes.c_void_p),
                ("type", ctypes.c_uint32),
                ("mode", ctypes.c_uint32),
                ("format", ctypes.c_uint32),
                ("buff_size", ctypes.c_int32),
                ("cusum", ctypes.c_int32),
                ("picture", SVPicture)]

class spikelinkInput :
    def __init__(self, path) :
        print('load spikelinkapi lib')        
        self.linkinputlib = ctypes.cdll.LoadLibrary(path)
        self.obj = self.linkinputlib.CreateSpikeLinkInputPython()
        self.brunning = False

    def init(self, params) :
        return self.linkinputlib.Init(self.obj, params)

    def setcallback(self, callback) :
        print('function : init camera device')
        self.linkinputlib.SetCallbackPython(self.obj, callback)

    def release(self) :
        print('function : release resources')
        self.linkinputlib.Fini(self.obj)

    def start(self) :
        print('function : start capture')
        if self.linkinputlib == None :
           return False
        return self.linkinputlib.Start(self.obj)
    
    def stop(self) :
        print('function : stop capture')
        if self.linkinputlib == None :
           return False
        return self.linkinputlib.Stop(self.obj)

    def open(self) :
        print('function : open camera device')
        if self.linkinputlib == None :
            return False
        return self.linkinputlib.Open(self.obj)

    def is_open(self) :
        if self.linkinputlib == None :
            return False
        return self.linkinputlib.IsOpen(self.obj)

    def close(self) :
        print('function : close camera device')
        if self.linkinputlib == None :
           return False
        self.linkinputlib.Close(self.obj)

    def getState(self) :
        print('function : get camera device state')
        if self.linkinputlib == None :
            return False
        return self.linkinputlib.GetState(self.obj)

    def releaseFrame(self, frame) :
        if self.linkinputlib == None :
            return
        p = ctypes.cast(frame, ctypes.c_void_p)
        self.linkinputlib.ReleaseFrame(self.obj, frame)

    def saveFile(self, path, nFrame, callback) :
        if self.linkinputlib == None :
            return
        self.linkinputlib.SaveFile(self.obj, path, nFrame, callback)

class spikeframepool:
    def __init__(self):
        self.items_ = []
        self.lock_ = threading.Lock()

    def is_empty(self):
        with self.lock_:
            return self.items_ == 0

    def size(self):
        with self.lock_:
            return len(self.items_)

    def top(self):
        with self.lock_:
            return self.items_[len(self.items_) - 1]

    def at(self, index):
        if index >= self.size() or index < 0:
            return None
        with self.lock_:
            return self.items_[index]

    def push(self, value):
        with self.lock_:
            return self.items_.append(value)

    def pop(self):
        with self.lock_:
            return self.items_.pop()
