#include <iostream>
#include "opencv2/opencv.hpp"
#include <cvaux.h>
#include <highgui.h>
 
using namespace cv;

void colorExtraction(cv::Mat* src, cv::Mat* dst,
    int code,
    int ch1Lower, int ch1Upper,
    int ch2Lower, int ch2Upper,
    int ch3Lower, int ch3Upper
    )
{
    cv::Mat colorImage;
    int lower[3];
    int upper[3];

    cv::Mat lut = cv::Mat(256, 1, CV_8UC3);   

    cv::cvtColor(*src, colorImage, code);

    lower[0] = ch1Lower;
    lower[1] = ch2Lower;
    lower[2] = ch3Lower;

    upper[0] = ch1Upper;
    upper[1] = ch2Upper;
    upper[2] = ch3Upper;

    for (int i = 0; i < 256; i++){
        for (int k = 0; k < 3; k++){
            if (lower[k] <= upper[k]){
                if ((lower[k] <= i) && (i <= upper[k])){
                    lut.data[i*lut.step+k] = 255;
                }else{
                    lut.data[i*lut.step+k] = 0;
                }
            }else{
                if ((i <= upper[k]) || (lower[k] <= i)){
                    lut.data[i*lut.step+k] = 255;
                }else{
                    lut.data[i*lut.step+k] = 0;
                }
            }
        }
    }

    //LUTを使用して二値化
    cv::LUT(colorImage, lut, colorImage);

    //Channel毎に分解
    std::vector<cv::Mat> planes;
    cv::split(colorImage, planes);

    //マスクを作成
    cv::Mat maskImage;
    cv::bitwise_and(planes[0], planes[1], maskImage);
    cv::bitwise_and(maskImage, planes[2], maskImage);

    //出力
    cv::Mat maskedImage;
    src->copyTo(maskedImage, ~maskImage);
    *dst = maskedImage;
}

int main (int argc, char **argv)
{
					// (1)load a specified file as a 3-channel color image
					const char *imagename = argc > 1 ? argv[1] : "../image/library.png";
					Mat im = imread(imagename);		/* 画像の取得 */
					Mat hsv, lab, ycr;
					if(!im.data) return -1;			/* エラー処理 */

					Mat roi(im, Rect(2500, 600, 2400, 1800));	/*トリミング */
					Mat extractedImage;
				 colorExtraction(&roi, &extractedImage, CV_BGR2HSV, 90, 150, 100, 255, 70, 255);

					int left, right, top, bottom;
					// まずはtopを見てみる
					cv::Vec3b *src = extractedImage.ptr<cv::Vec3b>(900 - 1);
					for ( int i = 0; i < 2400; i++) {
										cv::Vec3b hsv = src[i];
										if (hsv[2] != 0) {
															left = i;
															std::cout << "left pixel is " << left << std::endl;
															break;
										}
					}
					for ( int i = 2400 - 1; 0 < i; i--) {
										cv::Vec3b hsv = src[i];
										if (hsv[2] != 0) {
															right = i;
															std::cout << "right pixel is " << right << std::endl;
															break;
										}
					}
					src = extractedImage.ptr<cv::Vec3b>(0);
					for (int i = 0; i < 1800; i++) {
										cv::Vec3b hsv = src[i * 2400 + 1200];
										if (hsv[2] != 0) {
															top = i;
															std::cout << "top pixel is " << top << std::endl;
															break;
										}
					}
					
					src = extractedImage.ptr<cv::Vec3b>(top + 500);
					for (int i = 0; i < 1800 - top - 500; i++) {
										cv::Vec3b hsv0 = src[i * 2400 + 1200];
										// cv::Vec3b hsv1 = src[(i + 10) * 2400 + 1200];
										// cv::Vec3b hsv2 = src[(i + 20) * 2400 + 1200];
										if (hsv0[0] != 0 || hsv0[1] != 0 || hsv0[2] != 0) { continue;	}
										// if (hsv1[0] != 0 || hsv1[1] != 0 || hsv1[2] != 0) { continue; }
										// if (hsv2[0] != 0 || hsv2[1] != 0 || hsv2[2] != 0) { continue; }
										
										// if (hsv0[2] == hsv1[2] && hsv1[2] == hsv2[2] && hsv0[1] == hsv1[1] && hsv1[1] == hsv2[1] && hsv0[0] == hsv1[0] && hsv1[0] == nhsv2[0]) {
										// if (hsv[2] == 0) {
										bottom = i + top + 500;
										std::cout << "bottom pixel is " << bottom << std::endl;
										break;
									
					}
					Mat tile(extractedImage, Rect(left, top, right - left, bottom - top));
					/* 結果表示 */
					// imshow("Original", im);
					// imshow("ROI", roi);
					imshow("extract", extractedImage);
					imshow("tile", tile);
					waitKey(0);						/* 入力待機 */
 
					return 0;
}



