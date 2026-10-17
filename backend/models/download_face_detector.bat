@echo off
echo Downloading FaceDetector model (blaze_face_short_range.tflite)...
echo.
echo Download URL: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
echo.
curl -o blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
echo.
if exist blaze_face_short_range.tflite (
    echo Download successful!
) else (
    echo Download failed. Please download manually from the URL above.
)
pause
