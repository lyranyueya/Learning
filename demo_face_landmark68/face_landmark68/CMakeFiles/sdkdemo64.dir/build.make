# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68

# Include any dependencies generated for this target.
include CMakeFiles/sdkdemo64.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sdkdemo64.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sdkdemo64.dir/flags.make

CMakeFiles/sdkdemo64.dir/face_landmark68.c.o: CMakeFiles/sdkdemo64.dir/flags.make
CMakeFiles/sdkdemo64.dir/face_landmark68.c.o: face_landmark68.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/sdkdemo64.dir/face_landmark68.c.o"
	/mnt/fileroot/junyi.shen/buildroot/toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/bin//aarch64-linux-gnu-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sdkdemo64.dir/face_landmark68.c.o   -c /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/face_landmark68.c

CMakeFiles/sdkdemo64.dir/face_landmark68.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sdkdemo64.dir/face_landmark68.c.i"
	/mnt/fileroot/junyi.shen/buildroot/toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/bin//aarch64-linux-gnu-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/face_landmark68.c > CMakeFiles/sdkdemo64.dir/face_landmark68.c.i

CMakeFiles/sdkdemo64.dir/face_landmark68.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sdkdemo64.dir/face_landmark68.c.s"
	/mnt/fileroot/junyi.shen/buildroot/toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/bin//aarch64-linux-gnu-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/face_landmark68.c -o CMakeFiles/sdkdemo64.dir/face_landmark68.c.s

CMakeFiles/sdkdemo64.dir/main.cpp.o: CMakeFiles/sdkdemo64.dir/flags.make
CMakeFiles/sdkdemo64.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sdkdemo64.dir/main.cpp.o"
	/mnt/fileroot/junyi.shen/buildroot/toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/bin//aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sdkdemo64.dir/main.cpp.o -c /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/main.cpp

CMakeFiles/sdkdemo64.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sdkdemo64.dir/main.cpp.i"
	/mnt/fileroot/junyi.shen/buildroot/toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/bin//aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/main.cpp > CMakeFiles/sdkdemo64.dir/main.cpp.i

CMakeFiles/sdkdemo64.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sdkdemo64.dir/main.cpp.s"
	/mnt/fileroot/junyi.shen/buildroot/toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/bin//aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/main.cpp -o CMakeFiles/sdkdemo64.dir/main.cpp.s

# Object files for target sdkdemo64
sdkdemo64_OBJECTS = \
"CMakeFiles/sdkdemo64.dir/face_landmark68.c.o" \
"CMakeFiles/sdkdemo64.dir/main.cpp.o"

# External object files for target sdkdemo64
sdkdemo64_EXTERNAL_OBJECTS =

sdkdemo64: CMakeFiles/sdkdemo64.dir/face_landmark68.c.o
sdkdemo64: CMakeFiles/sdkdemo64.dir/main.cpp.o
sdkdemo64: CMakeFiles/sdkdemo64.dir/build.make
sdkdemo64: CMakeFiles/sdkdemo64.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable sdkdemo64"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sdkdemo64.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sdkdemo64.dir/build: sdkdemo64

.PHONY : CMakeFiles/sdkdemo64.dir/build

CMakeFiles/sdkdemo64.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sdkdemo64.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sdkdemo64.dir/clean

CMakeFiles/sdkdemo64.dir/depend:
	cd /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68 /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68 /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68 /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68 /mnt/fileroot/junyi.shen/nanoQ/nn_sdk/Linux/SDK_V1.8.2/demo_face_landmark68/face_landmark68/CMakeFiles/sdkdemo64.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sdkdemo64.dir/depend

