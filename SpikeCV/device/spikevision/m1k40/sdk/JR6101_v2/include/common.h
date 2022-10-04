/********************************************************************************************************************************
Copyright (C), 2018-2021
File name  :   common.h
Author     :
Version    :   1.0
Date       :   2021-07-23
Description:   
Others     :
Function List:
History    :   历史修改记录
          <author>    <time>   <version >   <desc>
          Jingtech  2021-07-23    1.0       JR6101：第一版建立。
********************************************************************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#pragma pack(push,1)

#include    "type.h"
#define CH_NUM_MAX                  3               // 通道0---3

//设备版本信息
typedef struct DevInf_t
{
    s32_t         DriverRev;  // 驱动程序版本号，以十进制表示，例如1000表示1.0版本。
    s32_t         DeviceRev;  // 授时卡版本号，以十进制表示，例如1000表示1.0版本。
    s32_t         FPGARev;    // FPGA固件版本号，以十进制表示，例如1000表示1.0版本。
}DevInf_t;

typedef struct
{
    s64_t       buf_size;         //有效数据的总长度
    s64_t       RxMBNum_SendCh;   //等待应用程序读取的内存块个数，发送通道
    s64_t       Flag;             //应用程序不应修改
    s32_t       *pbuf;            //内存起始地址
    s32_t       rezv;
}MB_Des_t;

typedef struct
{
    s32_t        nanosecond; // 纳秒。单位为1纳秒，以十进制表示。正常范围是0-999,999,999。只有特殊情况下会出现比该值大的情况，但不会超过1,000,000,000。

    u08_t        zone;       // 时区。
    u08_t        second;     // 秒。
    u08_t        minute;     // 分。
    u08_t        hour;       // 小时。

    u08_t        day;        // 日，以有符号十进制表示，例如7表示7日；
    u08_t        month;      // 月，以有符号十进制表示，例如6表示6月；
    s16_t        year;       // 年，以有符号十进制表示，例如2013表示2013年。
}DevTime_t;

typedef struct 
{
    s08_t        Mode;
    s08_t        WorkFlag;//bit_0抓包使能   ,bit_1:发包使能     ,  bit_2:PTP使能(同时使能抓包和发包)
    s08_t        RunFlag;

    MB_Des_t     Read;
    s32_t        RRemainLen;
    s08_t        *PRead;
    MB_Des_t     Send;
    s32_t        SRemainLen;
    s08_t        *PSend;

    s64_t        RReq;
    s64_t        ROver;
}ChState_t;

typedef struct
{
    s32_t          DeviceHand;
    s32_t             OpenFlag;                 //打开设备时指定，包模式/内存块模式
    bool            Run;
    DevInf_t        Version;

    s32_t          AssistantHandl;            //辅助线程,负责数据包模式下发送数据块，抓包中断。

    ChState_t       ChState[CH_NUM_MAX+1];
}Device_inf_t;

typedef struct
{
    long long	SysFrequency;//LARGE_INTEGER

    int          WaitHand_R;//HANDLE
    int          WaitHand_W;//HANDLE
    int          WaitHand_RW;//HANDLE
    int             mode;
    int             Run;

    int             RCount;
    int             RErrCount;      //丢包： 无可读取的内存块 错误
    int             RLostCount;
    int             WCount;
    int             WErrCount;

    long long   TimeBegin_R;//LARGE_INTEGER
    long long   TimeEnd_R;//LARGE_INTEGER
    long long       TimeCount_R;
    long long       Len_R;
    double          Speed_R;//long long

    long long   TimeBegin_W;//LARGE_INTEGER
    long long   TimeEnd_W;//LARGE_INTEGER
    long long       TimeCount_W;
    long long       Len_W;
    double          Speed_W;//long long

    long long       pkt_num_last;
}speed_t;

typedef struct
{
    u32_t            pkt_len;            //以太网数据包长度
    u08_t            status;             //数据包状态标记
                                            //bit[0]是CRC校验结果。0=CRC校验正确；1=CRC校验错误
                                            //bit[1]超长包指示，1=超长包
                                            //bit[2]是上一次丢包状态（Catch_drop_last）。该位表示该包之前有因为fifo满而导致的丢包。0=上一次未丢包；1=上一次有丢包。
                                            //bit[3]反射使能标记
                                            //bit[4]包解析使能标记
                                            //bit[5]反射匹配成功标记，
                                            //bit[6]包解析匹配成功，
    u08_t            speed;              //当前网络速度:
                                            //0=无连接或回环
                                            //1=10M
                                            //2=100M
                                            //4=1G
                                            //8=10G
    s16_t            smart_ch;           //包匹配通道号:  0xFF=未匹配
    u64_t            pkt_num;            //数据包编号
    DevTime_t           time;               //数据包时间戳
    u08_t            res_1[4];           //预留
}PacketInf_t;

#pragma pack(pop)

#endif



