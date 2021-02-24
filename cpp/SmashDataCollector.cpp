#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "match.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat img1 = imread("E:\\Archivio\\Programmazione\\VSCode\\SmashDataCollector\\readme_images\\2021010119113300_c.jpg");
    Mat img2 = imread("E:\\Archivio\\Programmazione\\VSCode\\SmashDataCollector\\readme_images\\2021010119115300_c.jpg");
    Mat result;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for(int i = 0; i < 1000; i++)
    {
        absdiff(img1, img2, result);
        sum(result);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    // Mat image1,image2,dst;
    // image1 = imread("E:\\Archivio\\Programmazione\\VSCode\\SmashDataCollector\\readme_images\\2021010119113300_c.jpg");
    // if( !image1.data ) { printf("Error loading image1 \n"); return -1;}
    // image2 = imread("E:\\Archivio\\Programmazione\\VSCode\\SmashDataCollector\\readme_images\\2021010119115300_c.jpg");
    // if( !image2.data ) { printf("Error loading image2 \n"); return -1;}
           
    // absdiff( image1,  image2,  dst);
  
    // namedWindow( "Display window");  
    // imshow( "Display window", image2 );                 

    // namedWindow( "Display windo");  
    // imshow( "Display windo", image1 );         

    // namedWindow( "Result window");   
    // imshow( "Result window", dst );

    cout << "aa" << endl;
    waitKey(0);
    return 0;
}