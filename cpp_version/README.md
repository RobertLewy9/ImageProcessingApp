# 遥感图像处理系统 - C++版本

这是一个使用C++、OpenCV和Qt实现的遥感图像处理系统。该系统提供了多种图像处理功能，包括灰度化、边缘检测、模糊和锐化等，并具有图形用户界面。

## 功能特性

- 灰度化处理
- Canny边缘检测
- Sobel边缘检测
- Laplacian边缘检测
- 高斯模糊
- 图像锐化
- 图形用户界面（基于Qt）
- 单元测试框架
- 开发者文档

## 项目结构

```
cpp_version/
├── CMakeLists.txt          # CMake构建配置文件
├── build/                  # 构建目录
├── include/                # 头文件目录
│   ├── image_processing/   # 图像处理模块头文件
│   │   ├── filters/        # 各种滤镜头文件
│   │   └── ImageProcessor.h
│   └── ui/                 # 用户界面头文件
├── src/                    # 源代码目录
│   ├── image_processing/   # 图像处理模块源文件
│   │   ├── filters/        # 各种滤镜源文件
│   │   └── ImageProcessor.cpp
│   ├── ui/                 # 用户界面源文件
│   └── main.cpp            # 主程序入口点
└── resources/              # 资源文件目录
    └── resources.qrc       # Qt资源文件
```

## 依赖项

- C++17 或更高版本
- OpenCV 4.x
- Qt 5.x (Core, Widgets模块)
- CMake 3.16 或更高版本

## 构建说明

### 使用CMake和命令行

1. 创建构建目录：
   ```bash
   mkdir build
   cd build
   ```

2. 运行CMake配置：
   ```bash
   cmake ..
   ```

3. 编译项目：
   ```bash
   make  # Linux/macOS
   mingw32-make  # Windows (MinGW)
   ```

### 使用Qt Creator

该项目可以使用Qt Creator进行开发：

1. 打开Qt Creator
2. 选择"打开项目"
3. 选择cpp_version目录下的CMakeLists.txt文件
4. 配置构建套件
5. 构建并运行项目

### 使用Visual Studio (Windows)

1. 安装Visual Studio和CMake工具
2. 打开Visual Studio
3. 选择"打开本地文件夹"
4. 选择cpp_version目录
5. Visual Studio会自动配置CMake项目
6. 构建并运行项目

### Windows (简单编译脚本)

如果上述方法不适用，可以使用简单的编译脚本：

```bash
compile_simple.bat
```

## 使用方法

构建完成后，运行生成的可执行文件：

```bash
./build/ImageProcessingApp  # Linux/macOS
build\ImageProcessingApp.exe  # Windows
```

在GUI界面中：

1. 点击"加载图像"按钮选择要处理的图像
2. 从下拉菜单中选择处理操作
3. 点击"处理图像"按钮应用所选操作
4. 点击"保存图像"按钮保存处理后的图像

## 测试

项目包含了单元测试，可以验证各个图像处理功能是否正常工作。可以通过以下方式运行测试：

```bash
# 构建测试
mkdir build
cd build
cmake ..
make

# 运行测试
./src/tests/TestImageProcessor
```

## 文档

开发者文档位于`docs/`目录中，包含以下内容：
- AddingNewFilters.md: 如何添加新的图像处理滤镜

## 开发说明

### 添加新的图像处理操作

1. 在`include/image_processing/filters/`目录下创建新的滤镜头文件
2. 在`src/image_processing/filters/`目录下创建对应的实现文件
3. 更新`ImageProcessor.h`和`ImageProcessor.cpp`以包含新的操作
4. 更新`MainWindow.cpp`中的操作选择下拉框以包含新操作

## 许可证

本项目采用MIT许可证。详情请见[LICENSE](LICENSE)文件。