package com.example.javasp;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.features2d.SIFT;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;
import org.opencv.ml.SVM;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SIFTFruitT {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Mendefinisikan direktori untuk setiap kategori
        File dirBanana = new File("app/src/main/assets/banana");
        File dirBlueberry = new File("app/src/main/assets/blueberry");

        List<Mat> descriptors = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>(); // Simpan label sebagai integer

        // Proses gambar untuk setiap kategori
        processCategory(dirBanana, descriptors, labelList, 1); // Label 1 untuk Banana
        processCategory(dirBlueberry, descriptors, labelList, 2); // Label 2 untuk blueberry

        // Gabungkan semua deskriptor ke dalam satu Mat
        Mat trainingData = new Mat();
        for (int i = 0; i < descriptors.size(); i++) {
            Mat desc = descriptors.get(i);
            trainingData.push_back(desc);
        }

        // Konversi labelList ke Mat of Integer
        Mat labels = new Mat(labelList.size(), 1, CvType.CV_32S);
        for (int i = 0; i < labelList.size(); i++) {
            labels.put(i, 0, labelList.get(i));
        }

        // Pelatihan SVM
        SVM classifier = SVM.create();
        classifier.setType(SVM.C_SVC);
        classifier.setKernel(SVM.LINEAR);
        classifier.setC(1);
        TrainData trainData = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        classifier.train(trainData.getSamples(), Ml.ROW_SAMPLE, trainData.getResponses());

        // Simpan classifier
        classifier.save("app/src/main/assets/SVM_Model/svm_fruit_classifier.xml");

        System.out.println("Training complete, classifier saved.");
    }

    private static void processCategory(File directory, List<Mat> descriptors, List<Integer> labelList, int label) {
        for (File imgFile : directory.listFiles()) {
            Mat img = Imgcodecs.imread(imgFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            MatOfKeyPoint keypoints = new MatOfKeyPoint();
            Mat descriptor = new Mat();
            SIFT sift = SIFT.create();

            sift.detectAndCompute(img, new Mat(), keypoints, descriptor);
            descriptors.add(descriptor);
            for (int i = 0; i < descriptor.rows(); i++) {
                labelList.add(label);
            }
        }
    }
}
