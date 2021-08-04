一、SDK编译
	1、修改nnsdk文件夹下的CMakeLists.txt
		A、修改GCC_PATH
		修改line 6 :SET(GCC_PATH path_to)
		设置gcc-linaro路径，path_to为gcc-linaro上一级路径
		B、编译ARM64 SDK可执行文件
		将line4及line5修改为如下所示：
		#SET(ARM_32 arm_32)
		SET(ARM_64 arm_64)
		C、编译ARM32 SDK可执行文件
		将line4及line5修改为如下所示：
		SET(ARM_32 arm_32)
		#SET(ARM_64 arm_64)
		D、line28修改可执行文件名称，当前默认为sdkdemo
	2、编译
		在nnsdk路径下执行命令：
		A)cmake .    //当前路径生成makefile
		B)make      //编译
		注：
		1、用户可自行下载编译工具链gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu和gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf，下载链接：
			https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
		https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/arm-linux-gnueabihf/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf.tar.xz
		2、aarch64-linux-gnu安装过程
			a、解压，并放置在自己需要的文件夹内 
			   tar -xvJf ***.tar.xz
			b、编辑bash.bashrc文件 
			   sudo vi ~/.bashrc
			c、添加变量   
			   export PATH=path_to/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin:$PATH
			d、更新环境变量   
			   source ~/.bashrc
			e、检查环境是否加入成功   
			   echo $PATH     
			   下方出现刚添加的PATH即为成功
			f、运行aarch64-linux-gnu-gcc -v，查看gcc安装版本
			详细过程可参考：https://www.cnblogs.com/flyinggod/p/9468612.html
		3、用户自建工程可将CMakeLists.txt文件拷贝到工程main函数同级目录下。
		4、CMakeLists.txt中ADD_EXECUTABLE生成可执行文件，ADD_LIBRARY生成so文件
			
	3、Opencv依赖
		SDK默认不依赖opencv环境，SDK中提供的BODY_POSE和TEXT_DETECTION 两demo需要依赖opencv环境，否则无法正常运行。若用户模型需要依赖opencv，则可获取opencv版本的libnnsdk.so。

二、SDK可执行文件执行
	1、资源拷贝
	将编译的可执行文件和xxx.nb和图片拷贝到板子上
	将libnnsdk.so拷贝到板子的/usr/lib/ 路径下
	注：
	1、face_detect示例的nbg文件在\sdk_release\tool\package\face_detect_nbg路径下
	2、其余的nbg文件可在github下载：https://github.com/Amlogic-NN/AML_NN_SDK
	3、目前libnnsdk.so在sdk_release\linux\package\lib32或\lib64路径下
	2、执行
	命令：./nnsdk xxx.nb type xxx.jpg
	示例：./nnSDK mobilenetv1_a1.nb 0 224*224*3.jpeg
	结果：

	注：
	1、type表示网络的序号，nbg文件需要与type相对应，详细说明可参见nn_sdk.h，或者板端运行时执行命令./nnsdk --help查看相关信息。
	2、SDK只支持JPEG格式的图片，下一版本会支持jpeg,bmp,png等格式
	3、目前版本只能处理与模型大小相同尺寸的图片，下一版本会实现输入图像尺寸自适应
	4、face_detect示例可单独编译出可执行文件，只需按照
	命令： ./sdkdemo face_detect_a1.nb 640x384x3.jpg

