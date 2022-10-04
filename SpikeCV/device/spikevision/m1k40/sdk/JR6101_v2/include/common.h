/********************************************************************************************************************************
Copyright (C), 2018-2021
File name  :   common.h
Author     :
Version    :   1.0
Date       :   2021-07-23
Description:   
Others     :
Function List:
History    :   ��ʷ�޸ļ�¼
          <author>    <time>   <version >   <desc>
          Jingtech  2021-07-23    1.0       JR6101����һ�潨����
********************************************************************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#pragma pack(push,1)

#include    "type.h"
#define CH_NUM_MAX                  3               // ͨ��0---3

//�豸�汾��Ϣ
typedef struct DevInf_t
{
    s32_t         DriverRev;  // ��������汾�ţ���ʮ���Ʊ�ʾ������1000��ʾ1.0�汾��
    s32_t         DeviceRev;  // ��ʱ���汾�ţ���ʮ���Ʊ�ʾ������1000��ʾ1.0�汾��
    s32_t         FPGARev;    // FPGA�̼��汾�ţ���ʮ���Ʊ�ʾ������1000��ʾ1.0�汾��
}DevInf_t;

typedef struct
{
    s64_t       buf_size;         //��Ч���ݵ��ܳ���
    s64_t       RxMBNum_SendCh;   //�ȴ�Ӧ�ó����ȡ���ڴ�����������ͨ��
    s64_t       Flag;             //Ӧ�ó���Ӧ�޸�
    s32_t       *pbuf;            //�ڴ���ʼ��ַ
    s32_t       rezv;
}MB_Des_t;

typedef struct
{
    s32_t        nanosecond; // ���롣��λΪ1���룬��ʮ���Ʊ�ʾ��������Χ��0-999,999,999��ֻ����������»���ֱȸ�ֵ�������������ᳬ��1,000,000,000��

    u08_t        zone;       // ʱ����
    u08_t        second;     // �롣
    u08_t        minute;     // �֡�
    u08_t        hour;       // Сʱ��

    u08_t        day;        // �գ����з���ʮ���Ʊ�ʾ������7��ʾ7�գ�
    u08_t        month;      // �£����з���ʮ���Ʊ�ʾ������6��ʾ6�£�
    s16_t        year;       // �꣬���з���ʮ���Ʊ�ʾ������2013��ʾ2013�ꡣ
}DevTime_t;

typedef struct 
{
    s08_t        Mode;
    s08_t        WorkFlag;//bit_0ץ��ʹ��   ,bit_1:����ʹ��     ,  bit_2:PTPʹ��(ͬʱʹ��ץ���ͷ���)
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
    s32_t             OpenFlag;                 //���豸ʱָ������ģʽ/�ڴ��ģʽ
    bool            Run;
    DevInf_t        Version;

    s32_t          AssistantHandl;            //�����߳�,�������ݰ�ģʽ�·������ݿ飬ץ���жϡ�

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
    int             RErrCount;      //������ �޿ɶ�ȡ���ڴ�� ����
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
    u32_t            pkt_len;            //��̫�����ݰ�����
    u08_t            status;             //���ݰ�״̬���
                                            //bit[0]��CRCУ������0=CRCУ����ȷ��1=CRCУ�����
                                            //bit[1]������ָʾ��1=������
                                            //bit[2]����һ�ζ���״̬��Catch_drop_last������λ��ʾ�ð�֮ǰ����Ϊfifo�������µĶ�����0=��һ��δ������1=��һ���ж�����
                                            //bit[3]����ʹ�ܱ��
                                            //bit[4]������ʹ�ܱ��
                                            //bit[5]����ƥ��ɹ���ǣ�
                                            //bit[6]������ƥ��ɹ���
    u08_t            speed;              //��ǰ�����ٶ�:
                                            //0=�����ӻ�ػ�
                                            //1=10M
                                            //2=100M
                                            //4=1G
                                            //8=10G
    s16_t            smart_ch;           //��ƥ��ͨ����:  0xFF=δƥ��
    u64_t            pkt_num;            //���ݰ����
    DevTime_t           time;               //���ݰ�ʱ���
    u08_t            res_1[4];           //Ԥ��
}PacketInf_t;

#pragma pack(pop)

#endif



