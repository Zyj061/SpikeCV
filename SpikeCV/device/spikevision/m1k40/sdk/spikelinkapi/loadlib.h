#ifndef LOADLIB_H
#define LOADLIB_H

#ifdef WIN32
#include <windows.h>
typedef HMODULE HandleType;
#define LoadLib(filename) LoadLibraryA(filename)
#define UnLoadLib(handle) FreeLibrary(handle)
#define GetProcAddressByName(handle, name) GetProcAddress(handle, name)
#else
#include <dlfcn.h>

typedef void * HandleType;
#define LoadLib(filename) dlopen(filename, RTLD_NOW)
#define UnLoadLib(handle) dlclose(handle)
#define GetProcAddressByName(handle, name) dlsym(handle, name)
#endif

#include <iostream>

class LoadLibRAII {
public:
    HandleType GetHandle() {
        return (handle_);
    }

    LoadLibRAII(const char *name) {
        handle_ = LoadLib(name);
        if(handle_ == nullptr) {
#ifdef WIN32
            fprintf(stderr, "%d\n", GetLastError());
#else
            fprintf(stderr, "%s\n", dlerror());
#endif
        }
    }

    void* GetLibAPI(const char *name) {
        int32_t errorId = 0;
#ifdef WIN32
        GetLastError();
#else
        dlerror();
#endif
        void* api = GetProcAddressByName(handle_, name);
#ifdef WIN32
        if((errorId = GetLastError()) != 0) {
            fprintf(stderr, "%d\n", errorId);
            return (nullptr);
        }
#else
        char* error;
        if((error = dlerror()) != nullptr) {
            fprintf(stderr, "%s\n", error);
            return (nullptr);
        }
#endif

        return api;
    }

    ~LoadLibRAII() {
        if(handle_) {
            UnLoadLib(handle_);
        }
    }

private:
    HandleType handle_;
};

#endif // LOADLIB_H
