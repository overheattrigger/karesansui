#include <iostream>
#include "opencv2/opencv.hpp"
#include <cvaux.h>
#include <highgui.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
// #include <direct.h>

using namespace cv;

#define ROWS 2400
#define COLS 2400
const int kCutRows = 60;
const int kCutCols = 70;
const string kRawDataDirectory = "../../RawData/";
const string kDataSetDirectory = "../../TrainData/";
const string kTestDataDirectory = "../../TestData/";
					
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

int extractTile(const string inputFileName, const string outputFileName)
{
					Mat im = imread(inputFileName.c_str());		/* 画像の取得 */
					Mat hsv, lab, ycr;
					if(!im.data) return -1;			/* エラー処理 */

					if (im.cols != 5312 && im.rows != 2988) { return -1; }
					
					Mat roi(im, Rect(2500, 500, COLS, ROWS));	/*トリミング */
					Mat extractedImage;
				 colorExtraction(&roi, &extractedImage, CV_BGR2HSV, 90, 150, 100, 255, 70, 255);

					int left, right, top, bottom;
					// まずはleftを見てみる
					cv::Vec3b *src = extractedImage.ptr<cv::Vec3b>(ROWS / 2 - 1);
					for ( int i = 0; i < COLS; i++) {
										cv::Vec3b hsv = src[i];
										if (hsv[2] != 0) {
															left = i;
															std::cout << "left pixel is " << left << std::endl;
															break;
										}
					}
					for ( int i = COLS - 1; 0 < i; i--) {
										cv::Vec3b hsv = src[i];
										if (hsv[2] != 0) {
															right = i;
															std::cout << "right pixel is " << right << std::endl;
															break;
										}
					}
					src = extractedImage.ptr<cv::Vec3b>(0);
					for (int i = 0; i < ROWS; i++) {
										cv::Vec3b hsv = src[i * COLS + COLS / 2];
										if (hsv[2] != 0) {
															top = i;
															std::cout << "top pixel is " << top << std::endl;
															break;
										}
					}
					
					src = extractedImage.ptr<cv::Vec3b>(top + 300);
					for (int i = 0; i < ROWS - top - 300 - 1; i++) {
										cv::Vec3b hsv0 = src[i * COLS + COLS / 2];
										if (hsv0[0] != 0 || hsv0[1] != 0 || hsv0[2] != 0) { continue;	}
										bottom = i + top + 300;
										std::cout << "bottom pixel is " << bottom << std::endl;
										break;
					}
					Mat tile(extractedImage, Rect(left, top, right - left, bottom - top));

					double reshapeRateRow = 1.0 * kCutRows / tile.rows;
					double reshapeRateCol = 1.0 * kCutCols / tile.cols;
					double rate = (reshapeRateCol < reshapeRateRow) ? reshapeRateCol : reshapeRateRow;
					resize(tile, tile, Size(), rate, rate);
					std::cout << "cols = " << tile.cols << " rows = " << tile.rows << std::endl;
					if (tile.cols < kCutCols && tile.rows < kCutRows) {
										std::cerr << "cols < kCutCols, rows < kCutRows" << std::endl;
										cv::Mat restore_aspect_img(cv::Size(kCutCols, kCutRows), CV_8UC3, CV_RGB(0,0,0));
										cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect((kCutCols - tile.cols) / 2, (kCutRows - tile.rows) / 2, tile.cols, tile.rows));
										tile.copyTo(dar16_9_roi);
										imwrite(outputFileName, restore_aspect_img);
										return 0;
					}

					if (tile.cols >= kCutCols && tile.rows < kCutRows) {
										std::cerr << "cols >= kCutCols, rows < kCutRows" << std::endl;
										Mat out(tile, Rect((kCutCols - tile.cols) / 2, 0, kCutCols, tile.rows));	/*トリミング */
										cv::Mat restore_aspect_img(cv::Size(kCutCols, kCutRows), CV_8UC3, CV_RGB(0,0,0));
										cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect(0, (kCutRows - out.rows) / 2, kCutCols, out.rows));
										out.copyTo(dar16_9_roi);
										imwrite(outputFileName, restore_aspect_img);
										return 0;
					}

					if (tile.cols < kCutCols && tile.rows >= kCutRows) {
										std::cerr << "cols < kCutCols, rows >= kCutRows" << std::endl;
										Mat out(tile, Rect(0, (tile.rows - kCutRows) / 2, tile.cols, kCutRows));	/*トリミング */
										cv::Mat restore_aspect_img(cv::Size(kCutCols, kCutRows), CV_8UC3, CV_RGB(0,0,0));
										cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect((kCutCols - out.cols) / 2, 0, out.cols, out.rows));
										out.copyTo(dar16_9_roi);
										
										imwrite(outputFileName, restore_aspect_img);
										return 0;
					}
					if (tile.cols >= kCutCols && tile.rows >= kCutRows) {
										std::cerr << "cols >= kCutCols, rows >= kCutRows" << std::endl;
										Mat out(tile, Rect((tile.cols - kCutCols) / 2, (tile.rows - kCutRows) / 2, kCutCols, kCutRows));	/*トリミング */
										imwrite(outputFileName, out);
										return 0;
					}
					std::cout << "output = " << outputFileName << std::endl;
					return 0;
}

void make_data_set(string inputID)
{
					// 元データのディレクトリ
					string input =inputID;
					string imageID;
					if (input.find("/") != -1) {
										imageID = inputID;
					} else {
										imageID = inputID + "/";
					}
									
					int labelCount = 0;
					// 元データを一つずつ列挙して画像を切って保存
					DIR* dp=opendir((kRawDataDirectory + imageID).c_str());
					if (dp!=NULL)
					{
										struct dirent* dent;
										do{
															dent = readdir(dp);
															if (dent == NULL) { continue; }
															string fileName = dent->d_name;
															if (fileName == "." || fileName == "..") { continue;	}
															string targetFile = kRawDataDirectory + imageID + fileName;
															char buf[8];
															sprintf(buf, "%04d", labelCount);
															string count = buf;
															string outputFile = (labelCount < 90) ? kDataSetDirectory : kTestDataDirectory;
															outputFile += imageID + count + ".jpg";

															std::cout << "targetFile is = " << targetFile << " output = " << outputFile <<std::endl;
															if (extractTile(targetFile, outputFile) == -1) { continue; }
															labelCount++;

										}while(dent!=NULL);
										closedir(dp);
					}
}

int main (int argc, char **argv)
{
					DIR* dp=opendir((kRawDataDirectory).c_str());
					if (dp!=NULL)
					{
										struct dirent* dent;
										do{
															dent = readdir(dp);
															if (dent == NULL) { continue; }
															string fileName = dent->d_name;
															if (fileName == "." || fileName == "..") { continue;	}
															make_data_set(fileName);
										}while(dent!=NULL);
										closedir(dp);
					}
					return 0;
}



