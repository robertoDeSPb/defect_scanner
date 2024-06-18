#include <iostream>
#include <opencv2/opencv.hpp>

void resizing(cv::Mat *img)
{
    cv::Mat resized;
    int new_width = 30;
    int new_height = 20;
    cv::resize(*img, resized, cv::Size(new_width, new_height));
}


void cropping(cv::Mat  img)
{
    int x = 100, y = 50, width = 200, height = 150;
    cv::Mat cropped = img(cv::Rect(x, y, width, height));
}

void rotating(cv::Mat &img)
{
    double angle = 45.0;
    cv::Point2f center(img.cols / 2.0, img.rows / 2.0);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(img, rotated, rotation_matrix, img.size());
}

int main() {
    // Read the image
    cv::Mat img = cv::imread("D:/Cpp Projects/cv/cvStudy/cvStudy/empty_image.jpg", cv::IMREAD_COLOR);

    // Check if the image is loaded successfully
    if (img.empty()) {
        std::cerr << "Error: Could not load the image.\n";
        return 1;
    }


    
    cv::Mat resized;
    int new_width = 30;
    int new_height = 20;
    cv::resize(img, resized, cv::Size(new_width, new_height));
    
    // Display the image
    cv::imshow("My Image", img);

    //cropping(img);
    cv::imshow("My Image", img);

    //rotating(img);
    cv::imshow("My Image", img);

    // Wait for a key press and close the window
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}