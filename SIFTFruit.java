package com.example.javasp;

import org.opencv.core.Core;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.features2d.SIFT;

public class SIFTFruit {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Memuat classifier yang telah disimpan
        SVM classifier = SVM.load("app/src/main/assets/SVM_Model/svm_fruit_classifier.xml");

        // Baca gambar uji
        Mat testImage = Imgcodecs.imread("app/src/main/assets/test/blueberry.jpeg", Imgcodecs.IMREAD_GRAYSCALE);

        // Inisialisasi SIFT dan mendapatkan deskriptor
        SIFT sift = SIFT.create();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();
        sift.detectAndCompute(testImage, new Mat(), keypoints, descriptors);

        // Menghitung rata-rata dari semua deskriptor
        Mat meanDescriptor = new Mat();
        Core.reduce(descriptors, meanDescriptor, 0, Core.REDUCE_AVG, -1);

        // Pastikan meanDescriptor berbentuk 1xN untuk prediksi
        if (meanDescriptor.rows() > 1)
            Core.transpose(meanDescriptor, meanDescriptor);

        // Klasifikasi menggunakan classifier SVM dengan deskriptor rata-rata
        float response = classifier.predict(meanDescriptor);

        // Menginterpretasikan respons dan mencetak hasilnya
        System.out.println("Response: " + response);
        switch ((int) response) {
            case 1:
                System.out.println("Gambar adalah banana.");
                break;
            case 2:
                System.out.println("Gambar adalah blueberry.");
                break;
            default:
                System.out.println("Kategori tidak dikenali.");
        }

        // Draw bounding box based on keypoints if the image is classified as "beaker"
        if (response == 1 || response == 2) {
            // Convert keypoints to an array to extract min and max points
            KeyPoint[] keypointArray = keypoints.toArray();

            if (keypointArray.length > 0) {
                // Find the min and max coordinates for the bounding box
                double minX = keypointArray[0].pt.x;
                double minY = keypointArray[0].pt.y;
                double maxX = keypointArray[0].pt.x;
                double maxY = keypointArray[0].pt.y;

                for (KeyPoint kp : keypointArray) {
                    if (kp.pt.x < minX) minX = kp.pt.x;
                    if (kp.pt.x > maxX) maxX = kp.pt.x;
                    if (kp.pt.y < minY) minY = kp.pt.y;
                    if (kp.pt.y > maxY) maxY = kp.pt.y;
                }

                // Create the bounding box rectangle
                Rect boundingBox = new Rect(new Point(minX, minY), new Point(maxX, maxY));

                // Convert the image to BGR to draw colored rectangle
                Mat colorImage = new Mat();
                Imgproc.cvtColor(testImage, colorImage, Imgproc.COLOR_GRAY2BGR);

                // Draw the bounding box in red color
                Imgproc.rectangle(colorImage, boundingBox.tl(), boundingBox.br(), new Scalar(0, 0, 255), 2);

                // Display the image with the bounding box
                HighGui.imshow("Deteksi Beaker dengan Bounding Box", colorImage);
                HighGui.waitKey(0);
                HighGui.destroyAllWindows();
            } else {
                System.out.println("Tidak ada keypoints yang terdeteksi untuk menggambar bounding box.");
            }
        }
    }
}
