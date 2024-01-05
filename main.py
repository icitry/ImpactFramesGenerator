from defs import Constants
from frame_detection import DynamicFrameDetector
from image_processing import ImageProcessingController
from video_managing import VideoManager


def main():
    image_processing_controller = ImageProcessingController()
    video_manager = VideoManager()
    dynamic_frame_detector = DynamicFrameDetector()

    video = video_manager.read_video_file(Constants.INPUT_VIDEO_PATH)
    impact_frames_data = dynamic_frame_detector.get_most_dynamic_frames_indexes(video,
                                                                                Constants.MAX_MOST_DYNAMIC_FRAMES)

    impact_frames_data = [(image_processing_controller.get_impact_frames(
        video_manager.get_frame_at_index(video, iframe_index)), iframe_index) for iframe_index in impact_frames_data]

    video_manager.write_video_file(Constants.OUTPUT_VIDEO_PATH, video, impact_frames_data)


if __name__ == '__main__':
    main()
