import cv2
import numpy as np


class DynamicFrameDetector:
    def _merge_overlapping_segments(self, segments):
        def find_element(elem_id, partition):
            if partition[elem_id] < 0:
                return elem_id
            partition[elem_id] = find_element(partition[elem_id], partition)
            return partition[elem_id]

        def union_elements(elem_id_1, elem_id_2, partition):
            elem_id_1 = find_element(elem_id_1, partition)
            elem_id_2 = find_element(elem_id_2, partition)
            if elem_id_1 != elem_id_2:
                partition[elem_id_2] = elem_id_1

        partitions = {}

        for segment in segments:
            for item in segment:
                partitions[item] = -1

        for segment in segments:
            for i in range(1, len(segment)):
                union_elements(segment[i - 1], segment[i], partitions)

        ans = {}
        for segment in segments:
            for item in segment:
                if find_element(item, partitions) not in ans:
                    ans[find_element(item, partitions)] = []
                ans[find_element(item, partitions)].append(item)

        ans = [list(set(x)) for x in ans.values()]

        return ans

    def get_most_dynamic_frames_indexes(self, video, max_num_frames, threshold=96, avg_frames=10):
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        _, prev_frame = video.read()

        dynamic_segments = []
        current_segment = []
        frame_number = 1

        while frame_number < frame_count:
            _, current_frame = video.read()
            frame_number += 1

            frame_diff = cv2.absdiff(prev_frame, current_frame)
            diff_percentage = np.count_nonzero(frame_diff) / (prev_frame.size / 3) * 100

            current_segment.append(diff_percentage)

            if len(current_segment) >= avg_frames:
                avg_diff_percentage = np.mean(current_segment)
                if avg_diff_percentage > threshold:
                    dynamic_segments.append(list(range(frame_number - avg_frames + 1, frame_number + 1)))
                current_segment.pop(0)

            prev_frame = current_frame

        merged_dynamic_segments = self._merge_overlapping_segments(dynamic_segments)

        merged_dynamic_segments.sort(key=lambda segment: np.mean(np.array(segment)), reverse=True)

        top_n_dynamic_segments = [segm[0] for segm in merged_dynamic_segments[:max_num_frames]]

        video.set(1, 0)

        return top_n_dynamic_segments
