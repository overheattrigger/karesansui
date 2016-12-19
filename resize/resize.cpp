#include <iostream>
#include "opencv2/opencv.hpp"
#include <cvaux.h>
#include <highgui.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
// #include <direct.h>

using namespace cv;
const int kCutRows = 60;
const int kCutCols = 70;
					
int resizeTile(const string inputFileName, const string outputFileName)
{
					Mat im = imread(inputFileName.c_str());		/* 画像の取得 */
					Mat hsv, lab, ycr;
					if(!im.data) return -1;			/* エラー処理 */
					
					int r = im.rows;
					int c = im.cols;
					double reshapeRateRow = 1.0 * kCutRows / im.rows;
					double reshapeRateCol = 1.0 * kCutCols / im.cols;
					double rate = (reshapeRateCol < reshapeRateRow) ? reshapeRateCol : reshapeRateRow;
					Mat tile;
					resize(im, tile, Size(), rate, rate);
					std::cout << "cols = " << im.cols << " rows = " << im.rows << std::endl;
					
					if (tile.cols < kCutCols && tile.rows < kCutRows) {
										std::cerr << "cols < kCutCols, rows < kCutRows" << std::endl;
										cv::Mat restore_aspect_img(cv::Size(kCutCols, kCutRows), CV_8UC3, CV_RGB(0,0,0));
										cv::Mat dar16_9_roi(restore_aspect_img, cv::Rect((kCutCols - tile.cols) / 2, (kCutRows - tile.rows) / 2, tile.cols, tile.rows));
										tile.copyTo(dar16_9_roi);
										// imshow("tile", restore_aspect_img);
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

int main (int argc, char **argv)
{
					const string targetDataDirectory = "../../targetData/";
					const string rawDataDirectory = "../../targetRawData/";
					
					int labelCount = 0;
					// 元データを一つずつ列挙して画像を切って保存
					DIR* dp=opendir((rawDataDirectory).c_str());
					if (dp!=NULL)
					{
										struct dirent* dent;
										do{
															dent = readdir(dp);
															if (dent == NULL) { continue; }
															string fileName = dent->d_name;
															if (fileName == "." || fileName == "..") { continue;	}
															string targetFile = rawDataDirectory + fileName;
															char buf[8];
															sprintf(buf, "%04d", labelCount);
															string count = buf;
															string outputFile = targetDataDirectory;
															outputFile += count + ".jpg";

															std::cout << "targetFile is = " << targetFile << " output = " << outputFile <<std::endl;
															if (resizeTile(targetFile, outputFile) == -1) { continue; }
															labelCount++;

										}while(dent!=NULL);
										closedir(dp);
					}
					return 0;
}



