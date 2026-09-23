#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include "image_processing/ImageProcessor.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_loadImageButton_clicked();
    void on_processButton_clicked();
    void on_saveButton_clicked();

private:
    void updateImageView();
    
    QLabel* imageLabel;
    QPushButton* loadImageButton;
    QPushButton* processButton;
    QPushButton* saveButton;
    QComboBox* operationComboBox;
    
    cv::Mat originalImage;
    cv::Mat processedImage;
    QString currentImagePath;
};

#endif // MAINWINDOW_H