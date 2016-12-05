#include <cv.h>
#include <highgui.h>
#include <math.h>

int
main (int argc, char **argv)
{
					int i;
					float *line, rho, theta;
					double a, b, x0, y0;
					IplImage *src_img_std = 0, *src_img_prob = 0, *src_img_gray = 0, *dst_img1 = 0;
					CvMemStorage *storage;
					CvSeq *lines = 0;
					CvPoint *point, pt1, pt2;

					// (1)画像の読み込み
					if (argc >= 2)
										src_img_gray = cvLoadImage (argv[1], CV_LOAD_IMAGE_GRAYSCALE);
					if (src_img_gray == 0)
										return -1;
					src_img_std = cvLoadImage (argv[1], CV_LOAD_IMAGE_COLOR);
					src_img_prob = cvCloneImage (src_img_std);

					// (2)ハフ変換のための前処理
					cvCanny (src_img_gray, src_img_gray, 50, 200, 3);
					storage = cvCreateMemStorage (0);

					// (3)標準的ハフ変換による線の検出と検出した線の描画
					// lines = cvHoughLines2 (src_img_gray, storage, CV_HOUGH_STANDARD, 1, CV_PI / 180, 50, 0, 0);
					// for (i = 0; i < MIN (lines->total, 100); i++) {
					//   line = (float *) cvGetSeqElem (lines, i);
					//   rho = line[0];
					//   theta = line[1];
					//   a = cos (theta);
					//   b = sin (theta);
					//   x0 = a * rho;
					//   y0 = b * rho;
					//   pt1.x = cvRound (x0 + 1000 * (-b));
					//   pt1.y = cvRound (y0 + 1000 * (a));
					//   pt2.x = cvRound (x0 - 1000 * (-b));
					//   pt2.y = cvRound (y0 - 1000 * (a));
					//   cvLine (src_img_std, pt1, pt2, CV_RGB (255, 0, 0), 3, 8, 0);
					// }

					// (4)確率的ハフ変換による線分の検出と検出した線分の描画
					lines = 0;
					lines = cvHoughLines2 (src_img_gray, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 30, 100, 1000);
					for (i = 0; i < lines->total; i++) {
										double a = 1.0 * abs(point[0].y - point[1].y) / abs(point[0].x - point[1].x);
										if (abs(point[0].x - point[1].x) < 10) {
										} else if (a > cos(3.14 * 5 / 360)) {
															continue;
										}
										point = (CvPoint *) cvGetSeqElem (lines, i);
										cvLine (src_img_prob, point[0], point[1], CV_RGB (255, 0, 0), 3, 8, 0);
					}

					dst_img1 = cvCreateImage( cvSize(src_img_std->width / 2,src_img_std->height / 2),IPL_DEPTH_8U, 3);
					// (5)検出結果表示用のウィンドウを確保し表示する
					// cvResize(src_img_std, dst_img1, CV_INTER_CUBIC);
					// cvNamedWindow ("Hough_line_standard", CV_WINDOW_AUTOSIZE);
					// cvShowImage ("Hough_line_standard", dst_img1);
					cvResize(src_img_prob,  dst_img1, CV_INTER_CUBIC);
					cvNamedWindow ("Hough_line_probalistic", CV_WINDOW_AUTOSIZE);
					cvShowImage ("Hough_line_probalistic", dst_img1);
					cvResizeWindow("Hough_line_standard", 800, 600);
					cvResizeWindow("Hough_line_probalistic", 800, 600);
					cvWaitKey (0);

					cvDestroyWindow ("Hough_line_standard");
					cvDestroyWindow ("Hough_line_standard");
					cvReleaseImage (&src_img_std);
					cvReleaseImage (&src_img_prob);
					cvReleaseImage (&src_img_gray);
					cvReleaseMemStorage (&storage);

					return 0;
}
