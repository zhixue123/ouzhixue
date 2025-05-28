import cv2
import face_recognition
import numpy as np

# 加载已知人脸图像并编码
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# 初始化摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 获取一帧视频
    ret, frame = video_capture.read()

    # 将图像从BGR颜色(OpenCV使用)转换为RGB颜色(face_recognition使用)
    rgb_frame = frame[:, :, ::-1]

    # 查找视频帧中的所有面部和面部编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 遍历每个检测到的人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 查看是否与已知人脸匹配
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Unknown"

        # 计算匹配度
        face_distances = face_recognition.face_distance([known_encoding], face_encoding)
        match_percentage = (1 - face_distances[0]) * 100

        if matches[0] and match_percentage > 70:  # 设置匹配阈值
            name = "Known Person"

        # 在脸部周围绘制矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 在矩形框下方绘制标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({match_percentage:.1f}%)", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # 显示结果图像
    cv2.imshow('Video', frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
video_capture.release()
cv2.destroyAllWindows()