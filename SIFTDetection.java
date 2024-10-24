package com.example.javasp;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;

public class SIFTDetection {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Load images
        Mat img1 = Imgcodecs.imread("app/src/main/assets/Template/beaker.png"); // template
        Mat img2 = Imgcodecs.imread("app/src/main/assets/testimage/beaker1.png");  // scene
        Imgproc.resize(img2, img2, new Size(img2.cols() * 0.3, img2.rows() * 0.3));

        // Initialize SIFT detector
        SIFT sift = SIFT.create();

        // Detect keypoints and compute descriptors
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        sift.detectAndCompute(img1, new Mat(), keypoints1, descriptors1);
        sift.detectAndCompute(img2, new Mat(), keypoints2, descriptors2);

        // Matching descriptor vectors with a FLANN based matcher
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        // Filter matches using the Lowe's ratio test
        float ratioThresh = 0.75f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (MatOfDMatch knnMatch : knnMatches) {
            if (knnMatch.rows() > 1) {
                DMatch[] matches = knnMatch.toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }

        // Draw good matches
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);
        Mat imgMatches = new Mat();
        Features2d.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar.all(-1),
                Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

//        // Calculate the rectangle to be drawn
//        List<Point> objPts = new ArrayList<>();
//        List<Point> scenePts = new ArrayList<>();
//        for (DMatch match : listOfGoodMatches) {
//            objPts.add(keypoints1.toArray()[match.queryIdx].pt);
//            scenePts.add(keypoints2.toArray()[match.trainIdx].pt);
//        }
//        MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f(objPts.toArray(new Point[0]));
//        MatOfPoint2f sceneMatOfPoint2f = new MatOfPoint2f(scenePts.toArray(new Point[0]));
//
//        Mat H = Calib3d.findHomography(objMatOfPoint2f, sceneMatOfPoint2f, Calib3d.RANSAC, 5);

//        // Draw rectangle around detected object
//        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2);
//        Mat sceneCorners = new Mat(4, 1, CvType.CV_32FC2);
//
//        objCorners.put(0, 0, new double[]{0, 0});
//        objCorners.put(1, 0, new double[]{img1.cols(), 0});
//        objCorners.put(2, 0, new double[]{img1.cols(), img1.rows()});
//        objCorners.put(3, 0, new double[]{0, img1.rows()});
//
//        Core.perspectiveTransform(objCorners, sceneCorners, H);

//        Imgproc.line(imgMatches, new Point(sceneCorners.get(0, 0)), new Point(sceneCorners.get(1, 0)), new Scalar(0, 255, 0), 4);
//        Imgproc.line(imgMatches, new Point(sceneCorners.get(1, 0)), new Point(sceneCorners.get(2, 0)), new Scalar(0, 255, 0), 4);
//        Imgproc.line(imgMatches, new Point(sceneCorners.get(2, 0)), new Point(sceneCorners.get(3, 0)), new Scalar(0, 255, 0), 4);
//        Imgproc.line(imgMatches, new Point(sceneCorners.get(3, 0)), new Point(sceneCorners.get(0, 0)), new Scalar(0, 255, 0), 4);

        // Display the result
        HighGui.imshow("Good Matches & Object detection", imgMatches);
        HighGui.waitKey(0);
    }
}
