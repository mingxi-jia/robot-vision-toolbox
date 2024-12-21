import numpy as np

def save_array_to_video(frames, save_path):
    pass

class VideoSegmentor():
    def __init__(self):
        self.sam = SAM()
        self.positive_prompt = None
        self.negative_prompt = None

    def segment_frame(self, rgb: np.array):
        # rgb is uin8
        rgb = self.sam.preprocess(rgb) # adjust image height width etc
        segmented_rgb, mask = self.sam.segment(rgb, self.positive_prompt, self.negative_prompt)

    def segment_video(self, video):
        segmented_rgbs = []
        for frame in video:
            segmented_rgb, mask = self.segment_frame(frame)
            segmented_rgbs.append(segmented_rgb)
        return np.stack(segmented_rgbs)

  if __name__ == "__main__":
      video_segmentor = VideoSegmentor()
      exmaple_video = None
      output = video_segmentor.segment_video(exmaple_video)
      save_array_to_video(output, save_path)
      
      
