# Cài đặt các dependencies cần thiết cho face_recognition trên Windows
1. Cài CMake & Visual C++ Build Tools
Cài CMake for Windows

Cài Build Tools for Visual Studio 2019/2022 từ: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Khi cài, bạn chỉ cần chọn:

+ C++ build tools

+ Bao gồm Windows 10 SDK

2. Cài dlib (cần cho face_recognition)
```bash
pip install cmake
pip install dlib
```
3. Cài face_recognition
```bash
pip install face_recognition
```