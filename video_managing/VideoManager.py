import cv2


class VideoManager:
    def read_video_file(self, path):
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print("Unable to read camera feed")
            return None

        return cap

    def get_frame_at_index(self, video, frame_index):
        video.set(1, frame_index - 1 if frame_index > 0 else 0)
        _, frame = video.read()
        video.set(1, 0)
        return frame

    def write_video_file(self, output_path, video, impact_frames_data):
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        fps = float(video.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (frame_width, frame_height))

        frame_idx = 0

        while video.isOpened():
            ret, frame = video.read()

            if ret:
                out.write(frame)

                if frame_idx in [iframe[1] for iframe in impact_frames_data]:
                    impact_frames = [iframe[0] for iframe in impact_frames_data if iframe[1] == frame_idx]

                    i = 1
                    for iframe in impact_frames[0]:
                        i += 1
                        out.write(iframe)
            else:
                break

            frame_idx += 1

        video.release()
        out.release()
