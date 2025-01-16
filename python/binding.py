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
        Args:
            img: numpy array in BGR (H,W,3)
        Returns:
            keypoints: numpy array (N,2)
            descriptors: numpy array (N,256)
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        torch_img = torch.from_numpy(gray_img).float() / 255.0
        torch_img = torch_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(torch_img)
        
        keypoints = feats['keypoints'][0].cpu().numpy()
        descriptors = feats['descriptors'][0].cpu().numpy()

        return keypoints, descriptors

    def match_features(self, kpts0, kpts1, desc0, desc1, w0, h0, w1, h1):
        """
        Args:
            kpts0, kpts1: numpy arrays (N,2) of keypoints
            desc0, desc1: numpy arrays (N,256) of descriptors
            w0, h0, w1, h1: image sizes
        Returns:
            matches: numpy array (M,2) of indices
            scores: numpy array (M,) of matching scores
        """
        feats0 = {
            'keypoints': torch.from_numpy(kpts0).to(self.device).unsqueeze(0),
            'descriptors': torch.from_numpy(desc0).to(self.device).unsqueeze(0),
            'image_size': torch.tensor([w0, h0]).to(self.device).unsqueeze(0),
        }
        feats1 = {
            'keypoints': torch.from_numpy(kpts1).to(self.device).unsqueeze(0),
            'descriptors': torch.from_numpy(desc1).to(self.device).unsqueeze(0),
            'image_size': torch.tensor([w1, h1]).to(self.device).unsqueeze(0),
        }

        with torch.no_grad():
            result = self.matcher({"image0": feats0, "image1": feats1})

        matches = result['matches'][0].cpu().numpy()
        scores = result['scores'][0].cpu().numpy()
        
        return matches, scores
