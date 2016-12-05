#include <iostream>
#include "opencv2/opencv.hpp"
#include <cvaux.h>
#include <highgui.h>
 
using namespace cv;
 
int main (int argc, char **argv)
{
					// (1)load a specified file as a 3-channel color image
					const char *imagename = argc > 1 ? argv[1] : "../image/library.png";
					Mat im = imread(imagename);		/* 画像の取得 */
					Mat hsv, lab, ycr;
					if(!im.data) return -1;			/* エラー処理 */

					Mat roi(im, Rect(2500, 600, 2400, 1800));	/*トリミング */
					/* 結果表示 */
					// imshow("Original", im);
					imshow("ROI", roi);
					waitKey(0);						/* 入力待機 */
 
					return 0;
}



