#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::dnn;

string yolo_cfg = "yolov3-tiny-coco/yolov3-tiny.cfg";
string yolo_model = "yolov3-tiny-coco/yolov3-tiny.weights";
string classes_yolov3 = "yolov3-tiny-coco/object_detection_classes_yolov3.txt";

class YoloNet
{
 public:
	YoloNet(string &yolo_cfg, string &yolo_model) {
		net_ = readNetFromDarknet(yolo_cfg, yolo_model);
		net_.setPreferableBackend(DNN_BACKEND_OPENCV);
		net_.setPreferableTarget(DNN_TARGET_CPU);
		outNames_ = net_.getUnconnectedOutLayersNames();
	}
	void setInputBlob(Mat &inputBlob) {
		net_.setInput(inputBlob);
	}
	void getOuts(vector<Mat> &outs) {
		net_.forward(outs, outNames_);
	}
 
 private:
	Net net_;
	vector<String> outNames_;

};

int main(int argc, char** argv)
{
	// Get class name vector
	vector<string> classNamesVec;
	ifstream classNamesFile(classes_yolov3);
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	// Input image and resize it
	string imageFile = "";
    if( argc > 1 ) {
        imageFile = argv[1];
    } else {
        imageFile = "image.jpg";
    }
	Mat frame = imread(imageFile);
	Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);

	// Init a YoloNet instance and get outs
	YoloNet net(yolo_cfg, yolo_model);
	net.setInputBlob(inputBlob);
	vector<Mat> outs;
	net.getOuts(outs);
	
	// Parse the outs
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Network produces output blob with a shape NxC where N is a number of
		// detected objects and C is a number of classes + 4 where the first 4
		// numbers are [center_x, center_y, width, height]
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.5)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Draw all the outs
	vector<int> indices;
	NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		string className = classNamesVec[classIds[idx]];
		if (className.compare("hot dog"))
			continue; 
		putText(frame, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
		rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
	}

	// Show all the hot dogs
	imshow("Hot Dog Detector Result", frame);
	waitKey(0);
}
