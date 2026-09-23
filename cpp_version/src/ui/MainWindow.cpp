#include "ui/MainWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QMessageBox>
#include <QPixmap>
#include <QScrollArea>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // Set window properties
    setWindowTitle("遥感图像处理系统");
    resize(800, 600);
    
    // Create central widget and layout
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    
    // Create and setup UI elements
    imageLabel = new QLabel("请加载一张图像", this);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setMinimumSize(400, 300);
    
    // Create scroll area for image
    QScrollArea* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setWidget(imageLabel);
    
    // Create buttons
    loadImageButton = new QPushButton("加载图像", this);
    processButton = new QPushButton("处理图像", this);
    saveButton = new QPushButton("保存图像", this);
    saveButton->setEnabled(false);
    
    // Create operation selection combo box
    operationComboBox = new QComboBox(this);
    operationComboBox->addItem("选择处理操作", -1);
    operationComboBox->addItem("灰度化", static_cast<int>(ImageProcessor::Operation::GRAYSCALE));
    operationComboBox->addItem("Canny边缘检测", static_cast<int>(ImageProcessor::Operation::CANNY));
    operationComboBox->addItem("Sobel边缘检测", static_cast<int>(ImageProcessor::Operation::SOBEL));
    operationComboBox->addItem("Laplacian边缘检测", static_cast<int>(ImageProcessor::Operation::LAPLACIAN));
    operationComboBox->addItem("高斯模糊", static_cast<int>(ImageProcessor::Operation::GAUSSIAN_BLUR));
    operationComboBox->addItem("图像锐化", static_cast<int>(ImageProcessor::Operation::SHARPEN));
    
    // Create layouts for buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(loadImageButton);
    buttonLayout->addWidget(operationComboBox);
    buttonLayout->addWidget(processButton);
    buttonLayout->addWidget(saveButton);
    
    // Add widgets to main layout
    mainLayout->addWidget(scrollArea);
    mainLayout->addLayout(buttonLayout);
    
    // Connect signals and slots
    connect(loadImageButton, &QPushButton::clicked, this, &MainWindow::on_loadImageButton_clicked);
    connect(processButton, &QPushButton::clicked, this, &MainWindow::on_processButton_clicked);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::on_saveButton_clicked);
    
    // Create menu bar
    QMenuBar* menuBar = new QMenuBar(this);
    setMenuBar(menuBar);
    
    QMenu* fileMenu = menuBar->addMenu("文件");
    QAction* exitAction = fileMenu->addAction("退出");
    connect(exitAction, &QAction::triggered, this, &QApplication::quit);
    
    QMenu* helpMenu = menuBar->addMenu("帮助");
    QAction* aboutAction = helpMenu->addAction("关于");
    connect(aboutAction, &QAction::triggered, [this]() {
        QMessageBox::about(this, "关于", "遥感图像处理系统 v1.0\n使用C++和OpenCV实现");
    });
}

MainWindow::~MainWindow()
{
}

void MainWindow::on_loadImageButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("打开图像"), "", tr("图像文件 (*.png *.jpg *.bmp)"));
    
    if (!fileName.isEmpty()) {
        currentImagePath = fileName;
        originalImage = cv::imread(fileName.toStdString());
        
        if (originalImage.empty()) {
            QMessageBox::warning(this, "错误", "无法加载图像文件");
            return;
        }
        
        // Display original image
        processedImage = originalImage;
        updateImageView();
        operationComboBox->setCurrentIndex(0); // Reset operation selection
        saveButton->setEnabled(false);
    }
}

void MainWindow::on_processButton_clicked()
{
    if (originalImage.empty()) {
        QMessageBox::warning(this, "警告", "请先加载一张图像");
        return;
    }
    
    int operationIndex = operationComboBox->currentIndex();
    if (operationIndex == 0) {
        QMessageBox::warning(this, "警告", "请选择一个处理操作");
        return;
    }
    
    // Get selected operation
    ImageProcessor::Operation operation = static_cast<ImageProcessor::Operation>(
        operationComboBox->currentData().toInt());
    
    // Process image
    processedImage = ImageProcessor::processImage(originalImage, operation);
    
    // Update display
    updateImageView();
    saveButton->setEnabled(true);
}

void MainWindow::on_saveButton_clicked()
{
    if (processedImage.empty()) {
        QMessageBox::warning(this, "警告", "没有可保存的图像");
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("保存图像"), "processed_image.jpg", tr("JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)"));
    
    if (!fileName.isEmpty()) {
        if (cv::imwrite(fileName.toStdString(), processedImage)) {
            QMessageBox::information(this, "成功", "图像保存成功");
        } else {
            QMessageBox::warning(this, "错误", "无法保存图像文件");
        }
    }
}

void MainWindow::updateImageView()
{
    if (processedImage.empty()) return;
    
    // Convert cv::Mat to QImage
    QImage img;
    if (processedImage.channels() == 3) {
        cv::cvtColor(processedImage, processedImage, cv::COLOR_BGR2RGB);
        img = QImage((const unsigned char*)(processedImage.data),
                     processedImage.cols, processedImage.rows,
                     processedImage.step, QImage::Format_RGB888);
    } else if (processedImage.channels() == 1) {
        img = QImage((const unsigned char*)(processedImage.data),
                     processedImage.cols, processedImage.rows,
                     processedImage.step, QImage::Format_Grayscale8);
    } else {
        return;
    }
    
    // Scale image to fit label while maintaining aspect ratio
    QPixmap pixmap = QPixmap::fromImage(img);
    imageLabel->setPixmap(pixmap.scaled(imageLabel->size(),
        Qt::KeepAspectRatio, Qt::SmoothTransformation));
}