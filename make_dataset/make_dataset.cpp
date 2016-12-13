#include <iostream>
#include "opencv2/opencv.hpp"
#include <cvaux.h>
#include <highgui.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cassert>
#include <vector>
#include <fstream>
int main (int argc, char **argv)
{

					const std::string dataSetDirectory = "../../DataSet/";
					const std::string testDataDirectory = "../../TestData/";
					const std::string judgeDirectory = "../judge/";

					std::vector<std::string> tileID;
					
					int labelCount = 0;
					// dataSetのディレクトリにある画像にIDをつけてtxtファイルに出力
					DIR* dp=opendir((dataSetDirectory).c_str());
					struct dirent* dent;
// データセットの取得				
					if (dp!=NULL)
					{
										do{
															dent = readdir(dp);
															if (dent == NULL) { continue; }
															std::string dirName = dent->d_name;
															if (dirName == "." || dirName == "..") { continue;	}
															int i;
															for (i = 0; i < tileID.size(); i++) {
																				if (std::atoi(dirName.c_str()) < std::atoi(tileID.at(i).c_str())) {
																									break;
																				}
															}
															if (i == tileID.size()) {
																				tileID.push_back(dirName);
															} else {
																				tileID.insert(tileID.begin() + i, dirName);
															}
										}while(dent!=NULL);
										closedir(dp);
					}

					for (int i = 0; i < tileID.size(); i++) {
										std::cout << "num = " << i << " name = " << tileID.at(i) << std::endl;
					}
					// ここからtrain.txtをつくる
					const std::string trainFileName = judgeDirectory + "train.txt";
					std::ofstream trainFile(trainFileName.c_str());
					for (int i = 0; i < tileID.size(); i++) {
										dp=opendir((dataSetDirectory+tileID.at(i)).c_str());
// データセットの取得				
										if (dp!=NULL)
										{
															do{
																				dent = readdir(dp);
																				if (dent == NULL) { continue; }
																				std::string fileName = dent->d_name;
																				if (fileName == "." || fileName == "..") { continue;	}
																				trainFile << dataSetDirectory << tileID.at(i) << "/" <<fileName << " " << i << std::endl;
															}while(dent!=NULL);
															closedir(dp);
										}
					}
					
					// ここからtest.txtをつくる
					const std::string testFileName = judgeDirectory + "test.txt";
					std::ofstream testFile(testFileName.c_str());
					for (int i = 0; i < tileID.size(); i++) {
										dp=opendir((testDataDirectory+tileID.at(i)).c_str());
// データセットの取得				
										if (dp!=NULL)
										{
															do{
																				dent = readdir(dp);
																				if (dent == NULL) { continue; }
																				std::string fileName = dent->d_name;
																				if (fileName == "." || fileName == "..") { continue;	}
																				testFile << testDataDirectory << tileID.at(i) << "/" <<fileName << " " << i << std::endl;
															}while(dent!=NULL);
															closedir(dp);
										}
					}

					return 0;
}



