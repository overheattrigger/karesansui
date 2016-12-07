#include <iostream>
#include "opencv2/opencv.hpp"
#include <cvaux.h>
#include <highgui.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
// #include <direct.h>

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

int extractTile(const string inputFileName, const string outputFileName)
{
					Mat im = imread(inputFileName.c_str());		/* 画像の取得 */
					Mat hsv, lab, ycr;
					if(!im.data) return -1;			/* エラー処理 */

					if (im.cols != 5312 && im.rows != 2988) { return -1; }
					
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
					
					src = extractedImage.ptr<cv::Vec3b>(top + 300);
					for (int i = 0; i < 1800 - top - 300 - 1; i++) {
										cv::Vec3b hsv0 = src[i * 2400 + 1200];
										if (hsv0[0] != 0 || hsv0[1] != 0 || hsv0[2] != 0) { continue;	}
										bottom = i + top + 300;
										std::cout << "bottom pixel is " << bottom << std::endl;
										break;
					}
					
					Mat tile(extractedImage, Rect(left, top, right - left, bottom - top));
					resize(tile, tile, Size(), 0.44, 0.44);
					std::cout << "cols = " << tile.cols << " rows = " << tile.rows << std::endl;
					if (tile.cols < 800 && tile.rows < 600) {
										cv::Mat restore_aspect_img(cv::Size(800, 600), CV_8UC3, CV_RGB(0,0,0));
										cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect((800 - tile.cols) / 2, (600 - tile.rows) / 2, tile.cols, tile.rows));
										tile.copyTo(dar16_9_roi);
										// imshow("tile", restore_aspect_img);
										imwrite(outputFileName, restore_aspect_img);

										return 0;
					}

					if (tile.cols >= 800 && tile.rows < 600) {
										Mat out(tile, Rect((800 - tile.cols) / 2, 0, 800, tile.rows));	/*トリミング */
										cv::Mat restore_aspect_img(cv::Size(800, 600), CV_8UC3, CV_RGB(0,0,0));
										cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect(800, (600 - out.rows) / 2, 800, out.rows));
										out.copyTo(dar16_9_roi);
										imwrite(outputFileName, restore_aspect_img);
//										imshow("tile", restore_aspect_img); // 

										return 0;
					}

					if (tile.cols < 800 && tile.rows >= 600) {
										Mat out(tile, Rect(0, (tile.rows - 600) / 2, tile.cols, 600));	/*トリミング */
										cv::Mat restore_aspect_img(cv::Size(800, 600), CV_8UC3, CV_RGB(0,0,0));
										cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect((800 - out.cols) / 2, 0, out.cols, 600));
										out.copyTo(dar16_9_roi);
										imwrite(outputFileName, restore_aspect_img);
										//								imshow("tile", restore_aspect_img);

										return 0;
					}

					if (tile.cols >= 800 && tile.rows >= 600) {
										Mat out(tile, Rect((tile.cols - 800) / 2, (tile.rows - 600) / 2, 800, 600));	/*トリミング */
										imwrite(outputFileName, out);
										//	imshow(outputFileName, out);

										return 0;
					}
  // 画像リサイズ

					// if (tile.cols < 800 && tile.rows < 600) {
					// 					cv::Mat restore_aspect_img(cv::Size(800, 600), CV_8UC3, CV_RGB(0,0,0));
					// 					cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect((tile.cols- 800) / 2,(tile.rows - 600) / 2, tile.cols, tile.rows));
					// 					tile.copyTo(dar16_9_roi);
					// 					imshow("tile", restore_aspect_img);
					// }
					/* 結果表示 */
					// imshow("extract", extractedImage);
					
					std::cout << "output = " << outputFileName << std::endl;
					// imwrite(outputFileName, tile);

					return 0;

}

int main (int argc, char **argv)
{
					const string rawDataDirectory = "../../RawData/";
					const string dataSetDirectory = "../../DataSet/";

					// 元データのディレクトリ
					string imageID = "ID-0000/";
					if (argc > 1) {
										string input = argv[1];
										if (input.find("/") != -1) {
															imageID = input;
										} else {
															imageID = input + "/";
										}
					} 
					// 出力先のディレクトリの作成
					// struct stat st;
					// if(stat((dataSetDirectory + imageID).c_str(), &st) != 0){
     //     mkdir((dataSetDirectory + imageID).c_str(), 0777);
					// }
					
					int labelCount = 0;
					// 元データを一つずつ列挙して画像を切って保存
					DIR* dp=opendir((rawDataDirectory + imageID).c_str());
					if (dp!=NULL)
					{
										struct dirent* dent;
										do{
															dent = readdir(dp);
															if (dent == NULL) { continue; }
															string fileName = dent->d_name;
															if (fileName == "." || fileName == "..") { continue;	}
															string targetFile = rawDataDirectory + imageID + fileName;
															char buf[8];
															sprintf(buf, "%04d", labelCount);
															string count = buf;
															string outputFile = dataSetDirectory + imageID + count + ".jpg";
															std::cout << "targetFile is = " << targetFile << " output = " << outputFile <<std::endl;
															if (extractTile(targetFile, outputFile) == -1) { continue; }
															labelCount++;

										}while(dent!=NULL);
										closedir(dp);
					}


					return 0;
}



