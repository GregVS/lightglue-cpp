import pybind11
import numpy as np
import torch
from lightglue import SuperPoint, LightGlue
import cv2

class FeatureExtractor:
    def __init__(self, max_keypoints=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def extract_features(self, img):
        """
        Extract features from an image
        Args:
            img: numpy array (grayscale)
        Returns:
            keypoints: numpy array (N,2)
            descriptors: numpy array (N,256)
        """
        torch_img = torch.from_numpy(img).float() / 255.0
        torch_img = torch_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(torch_img)
        
        keypoints = feats['keypoints'][0].cpu().numpy()
        descriptors = feats['descriptors'][0].cpu().numpy()
        
        return keypoints, descriptors

    def match_features(self, kpts0, kpts1, desc0, desc1):
        """
        Match features between two images
        Args:
            kpts0, kpts1: numpy arrays (N,2) of keypoints
            desc0, desc1: numpy arrays (N,256) of descriptors
        Returns:
            matches: numpy array (M,2) of indices
            scores: numpy array (M,) of matching scores
        """
        feats0 = {
            'keypoints': torch.from_numpy(kpts0).to(self.device)[None],
            'descriptors': torch.from_numpy(desc0).to(self.device)[None],
        }
        feats1 = {
            'keypoints': torch.from_numpy(kpts1).to(self.device)[None],
            'descriptors': torch.from_numpy(desc1).to(self.device)[None],
        }

        with torch.no_grad():
            matches = self.matcher({"image0": feats0, "image1": feats1})

        matches = matches['matches0'][0].cpu().numpy()
        scores = matches['scores'][0].cpu().numpy()
        
        valid = matches > -1
        matches = np.stack([np.where(valid)[0], matches[valid]], axis=1)
        scores = scores[valid]
        
        return matches, scores
