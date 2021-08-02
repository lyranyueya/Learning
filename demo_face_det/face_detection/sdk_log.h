#ifndef __SDK_LOG__
#define __SDK_LOG__

#ifdef __cplusplus
extern "C"{
#endif


//#define ANDROID_SDK
//#define ANDROID_SDK_APK
#define AML_NNSDK_STATUS_OK                              0

#ifdef ANDROID_SDK
#include<android/log.h>
#endif

typedef enum{
  SDK_LOG_NULL = -1,	        ///< close all log output
  SDK_LOG_TERMINAL,	            ///< set log print to terminal
  SDK_LOG_FILE,		            ///< set log print to file
  SDK_LOG_SYSTEM                ///< set log print to system
}sdk_log_format_t;

typedef enum{
	SDK_DEBUG_LEVEL_RELEASE = -1,	///< close debug
	SDK_DEBUG_LEVEL_ERROR,		    ///< error level,hightest level system will exit and crash
	SDK_DEBUG_LEVEL_WARN,		    ///< warning, system continue working,but something maybe wrong
	SDK_DEBUG_LEVEL_INFO,		    ///< info some value if needed
	SDK_DEBUG_LEVEL_PROCESS,	    ///< default,some process print
	SDK_DEBUG_LEVEL_DEBUG		    ///< debug level,just for debug
}sdk_debug_level_t;

#ifndef ANDROID_SDK_APK
#define LOGE( fmt, ... ) \
    nn_sdk_LogMsg(SDK_DEBUG_LEVEL_ERROR, "E %s[%s:%d]" fmt, SDK_API, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOGE(fmt, args...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##args)
#endif
#define DEBUG_BUFFER_LEN 512
#define SDK_API  "NN_SDK: "
#define LOG_TAG "NN_SDK"
#define LOGW( fmt, ... ) \
    nn_sdk_LogMsg(SDK_DEBUG_LEVEL_WARN,  "W %s[%s:%d]" fmt, SDK_API, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOGI( fmt, ... ) \
    nn_sdk_LogMsg(SDK_DEBUG_LEVEL_INFO,  "I %s[%s:%d]" fmt, SDK_API, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOGP( fmt, ... ) \
    nn_sdk_LogMsg(SDK_DEBUG_LEVEL_PROCESS, "P %s[%s:%d]" fmt, SDK_API, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOGD( fmt, ... ) \
    nn_sdk_LogMsg(SDK_DEBUG_LEVEL_DEBUG, "D %s[%s:%d]" fmt, SDK_API, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define __LOG__( fmt, ... ) \
    nn_sdk_LogMsg(SDK_DEBUG_LEVEL_DEBUG, "%s[%s:%d]" fmt, SDK_API, __FUNCTION__, __LINE__, ##__VA_ARGS__)

void nn_sdk_LogMsg(sdk_debug_level_t level, const char *fmt, ...);
void det_set_log_level(sdk_debug_level_t level,sdk_log_format_t output_format);

#ifdef __cplusplus
}
#endif
#endif