package com.example.javasp;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;

public class TemplateMatching {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Membaca gambar sumber
        Mat sourceImage = Imgcodecs.imread("app/src/main/assets/Gambar4.png");
        Imgproc.resize(sourceImage, sourceImage, new Size(sourceImage.cols() * 2, sourceImage.rows() * 2));

        // Mendefinisikan template dan label yang sesuai
        String[][] templates = {
                {"app/src/main/assets/TemplateGambar.jpg", "Apel"}
        };

        for (String[] templateData : templates) {
            Mat templateImage = Imgcodecs.imread(templateData[0]);
            String label = templateData[1];
            Mat outputImage = new Mat();

            // Melakukan template matching
            Imgproc.matchTemplate(sourceImage, templateImage, outputImage, Imgproc.TM_CCOEFF_NORMED);

            // Menentukan threshold
            double threshold = 0.65;

            // List untuk menyimpan semua bounding box yang ditemukan
            List<Rect> boundingBoxes = new ArrayList<>();

            // Melakukan deteksi semua titik yang sesuai dengan threshold
            for (int y = 0; y < outputImage.rows(); y++) {
                for (int x = 0; x < outputImage.cols(); x++) {
                    // Cek apakah nilai kecocokan pada titik ini lebih besar dari threshold
                    if (outputImage.get(y, x)[0] >= threshold) {
                        Point matchLoc = new Point(x, y);
                        Rect boundingBox = new Rect(new Point(matchLoc.x, matchLoc.y),
                                new Point(matchLoc.x + templateImage.width(), matchLoc.y + templateImage.height()));
                        boundingBoxes.add(boundingBox);
                    }
                }
            }

            // Lakukan Non-Maximum Suppression (NMS) untuk menggabungkan bounding box yang tumpang tindih
            List<Rect> finalBoxes = nonMaximumSuppression(boundingBoxes, 0.5);  // Threshold IoU = 0.5

            // Menandai hasil akhir pada gambar
            for (Rect box : finalBoxes) {
                Imgproc.rectangle(sourceImage, box.tl(), box.br(), new Scalar(0, 255, 0), 2);
                Imgproc.putText(sourceImage, label, new Point(box.x, box.y - 5),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 1);
            }

            // Menampilkan hasil
            HighGui.imshow("Matched Result - " + label, sourceImage);
            HighGui.waitKey();
        }
    }

    // Fungsi untuk melakukan Non-Maximum Suppression (NMS)
    private static List<Rect> nonMaximumSuppression(List<Rect> boxes, double iouThreshold) {
        List<Rect> finalBoxes = new ArrayList<>();

        // Selama masih ada box dalam list, terus lakukan NMS
        while (!boxes.isEmpty()) {
            // Ambil box pertama
            Rect bestBox = boxes.remove(0);
            finalBoxes.add(bestBox);

            // Cek overlap untuk setiap box lainnya
            Iterator<Rect> iterator = boxes.iterator();
            while (iterator.hasNext()) {
                Rect box = iterator.next();
                if (computeIoU(bestBox, box) > iouThreshold) {
                    // Jika IoU lebih besar dari threshold, hapus box yang tumpang tindih
                    iterator.remove();
                }
            }
        }

        return finalBoxes;
    }

    // Fungsi untuk menghitung Intersection over Union (IoU)
    private static double computeIoU(Rect box1, Rect box2) {
        // Hitung koordinat titik overlap
        double x1 = Math.max(box1.x, box2.x);
        double y1 = Math.max(box1.y, box2.y);
        double x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
        double y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

        // Area overlap
        double overlapArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

        // Total area dari kedua bounding box
        double box1Area = box1.width * box1.height;
        double box2Area = box2.width * box2.height;

        // Hitung IoU
        return overlapArea / (box1Area + box2Area - overlapArea);
    }
}
