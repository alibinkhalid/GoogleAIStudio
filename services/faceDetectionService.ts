/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

// Define a simple BoundingBox type
export interface BoundingBox {
  originX: number;
  originY: number;
  width: number;
  height: number;
}

let faceDetector: FaceDetector | null = null;

const initializeFaceDetector = async (): Promise<FaceDetector> => {
    if (faceDetector) {
        return faceDetector;
    }

    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
                delegate: "GPU"
            },
            runningMode: "IMAGE"
        });
        console.log("Face detector initialized successfully.");
        return faceDetector;
    } catch (error) {
        console.error("Error initializing face detector:", error);
        throw error;
    }
};

/**
 * Detects faces in an HTMLImageElement.
 * @param image The HTMLImageElement to process.
 * @returns A promise that resolves to an array of BoundingBox objects.
 */
export const detectFaces = async (image: HTMLImageElement): Promise<BoundingBox[]> => {
    const detector = await initializeFaceDetector();
    
    const detections = detector.detect(image);
    
    if (!detections || detections.detections.length === 0) {
        console.log("No faces detected.");
        return [];
    }
    
    console.log(`Detected ${detections.detections.length} faces.`);

    return detections.detections.map(detection => {
        const { originX, originY, width, height } = detection.boundingBox;
        return { originX, originY, width, height };
    });
};
