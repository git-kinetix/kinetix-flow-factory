"""
GVHMR-based MPJPE pose scorer for RL reward computation.

Runs GVHMR on both reference (original) and generated videos to estimate
camera-space 3D human pose, then computes pelvis-aligned MPJPE between them.

Data flow:
    Reference video [C,T,H,W] -> GVHMR -> SMPL -> ref_joints [T,17,3]  (cached by video_id)
    Generated video [C,T,H,W] -> GVHMR -> SMPL -> gen_joints [T,17,3]
                                                       |
                               pelvis-aligned MPJPE(ref_joints, gen_joints) -> reward

Vendored from DiffusionNFT/flow_grpo/gvhmr_pose_scorer.py to avoid
external dependency on the DiffusionNFT package.
"""

import os
import sys
import tempfile
import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# COCO17 hip indices for pelvis midpoint
COCO17_LEFT_HIP, COCO17_RIGHT_HIP = 11, 12


class GVHMRPoseScorer(nn.Module):
    """Negative pelvis-aligned MPJPE as reward, comparing GVHMR predictions on ref vs generated videos."""

    def __init__(self, device, gvhmr_root="/data/GVHMR"):
        super().__init__()
        self.device = device
        self.gvhmr_root = gvhmr_root

        if gvhmr_root not in sys.path:
            sys.path.insert(0, gvhmr_root)

        # Cache: video_id -> joints [T, 17, 3] on CPU
        self._ref_cache = {}

        self._init_gvhmr()
        self.eval()

    def _init_gvhmr(self):
        """Load GVHMR model, SMPL body model, and COCO17 joint regressor."""
        import hydra
        from hydra import initialize_config_module, compose
        from hydra.core.global_hydra import GlobalHydra

        from hmr4d.configs import register_store_gvhmr
        import hmr4d.model.gvhmr.gvhmr_pl_demo  # registers gvhmr_pl_demo in hydra ConfigStore
        from hmr4d.utils.smplx_utils import make_smplx
        from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
        from hmr4d.utils.geo_transform import compute_cam_angvel

        # Store utility references
        self._get_bbx_xys_from_xyxy = get_bbx_xys_from_xyxy
        self._estimate_K = estimate_K
        self._compute_cam_angvel = compute_cam_angvel
        self._Tracker = Tracker
        self._VitPoseExtractor = VitPoseExtractor
        self._Extractor = Extractor

        # Lazy-loaded preprocessors (heavy GPU models)
        self._tracker = None
        self._vitpose_extractor = None
        self._feature_extractor = None

        # GVHMR uses hydra configs with relative paths -- chdir temporarily
        orig_cwd = os.getcwd()
        os.chdir(self.gvhmr_root)
        try:
            GlobalHydra.instance().clear()
            with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
                register_store_gvhmr()
                cfg = compose(config_name="demo", overrides=["static_cam=True"])
                self.gvhmr_model = hydra.utils.instantiate(cfg.model, _recursive_=False)
                ckpt_path = cfg.ckpt_path
            self.gvhmr_model.load_pretrained_model(ckpt_path)
        finally:
            os.chdir(orig_cwd)

        self.gvhmr_model = self.gvhmr_model.eval().to(self.device)

        # SmplxLiteCoco17: directly outputs COCO17 joints (faster than full SMPL-X -> vertices -> regressor)
        self.smplx_coco17 = make_smplx("supermotion_coco17").to(self.device)

        logger.info("GVHMRPoseScorer initialized")

    # -- Lazy-loaded preprocessors (allocated on first use) --

    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = self._Tracker()
        return self._tracker

    @property
    def vitpose_extractor(self):
        if self._vitpose_extractor is None:
            self._vitpose_extractor = self._VitPoseExtractor()
        return self._vitpose_extractor

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            self._feature_extractor = self._Extractor()
        return self._feature_extractor

    # -- Internal helpers --

    def _video_tensor_to_mp4(self, video_tensor, tmp_dir, filename="video.mp4"):
        """Convert [C, T, H, W] float [-1,1] tensor -> temp mp4 on disk.

        Returns (path, length, width, height).
        """
        import imageio

        frames = ((video_tensor + 1) / 2 * 255).clamp(0, 255).byte()
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
        path = os.path.join(tmp_dir, filename)
        imageio.mimwrite(path, frames, fps=30)
        T, H, W, _ = frames.shape
        return path, T, W, H

    def _preprocess_video(self, video_path, length, width, height):
        """Run GVHMR preprocessing (tracking -> ViTPose -> ViT features)."""
        bbx_xyxy = self.tracker.get_one_track(video_path).float()
        bbx_xys = self._get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

        vitpose = self.vitpose_extractor.extract(video_path, bbx_xys)
        vit_features = self.feature_extractor.extract_video_features(video_path, bbx_xys)

        R_w2c = torch.eye(3).repeat(length, 1, 1)
        K_fullimg = self._estimate_K(width, height).repeat(length, 1, 1)

        return {
            "length": torch.tensor(length),
            "bbx_xys": bbx_xys,
            "kp2d": vitpose,
            "K_fullimg": K_fullimg,
            "cam_angvel": self._compute_cam_angvel(R_w2c),
            "f_imgseq": vit_features,
        }

    def _predict_joints(self, data):
        """GVHMR predict -> SmplxLiteCoco17 forward -> COCO17 joints [T, 17, 3]."""
        from hmr4d.utils.net_utils import to_cuda

        pred = self.gvhmr_model.predict(data, static_cam=True)
        params = to_cuda(pred["smpl_params_incam"])
        joints_17 = self.smplx_coco17(**params)  # [T, 17, 3]
        return joints_17

    def _video_to_joints(self, video_tensor):
        """Full pipeline: video tensor -> GVHMR -> COCO17 joints [T, 17, 3]."""
        # GVHMR preprocessors use relative checkpoint paths -- chdir to gvhmr_root
        orig_cwd = os.getcwd()
        os.chdir(self.gvhmr_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                video_path, length, width, height = self._video_tensor_to_mp4(video_tensor, tmp_dir)
                data = self._preprocess_video(video_path, length, width, height)
                return self._predict_joints(data)
        finally:
            os.chdir(orig_cwd)

    def _get_ref_joints(self, ref_video, video_id):
        """Get reference joints, using cache if available."""
        if video_id in self._ref_cache:
            return self._ref_cache[video_id].to(self.device)

        joints = self._video_to_joints(ref_video)
        self._ref_cache[video_id] = joints.cpu()
        logger.info(f"Cached GVHMR ref joints for {video_id}: {list(joints.shape)}")
        return joints

    @staticmethod
    def _pelvis_aligned_mpjpe(pred_joints, gt_joints):
        """Pelvis-aligned MPJPE over all 17 COCO joints.

        Pelvis = midpoint of left_hip (11) and right_hip (12).
        """
        pred_pelvis = (
            pred_joints[:, COCO17_LEFT_HIP] + pred_joints[:, COCO17_RIGHT_HIP]
        ).unsqueeze(1) / 2
        gt_pelvis = (
            gt_joints[:, COCO17_LEFT_HIP] + gt_joints[:, COCO17_RIGHT_HIP]
        ).unsqueeze(1) / 2

        return torch.norm(
            (pred_joints - pred_pelvis) - (gt_joints - gt_pelvis), dim=-1
        ).mean().item()

    # -- Public interface --

    @torch.no_grad()
    def __call__(self, generated_videos, ref_videos, video_ids):
        """
        Args:
            generated_videos: list of [C, T, H, W] tensors in [-1, 1]
            ref_videos: list of [C, T, H, W] tensors in [-1, 1] (original videos)
            video_ids: list of str identifiers for caching ref predictions
        Returns:
            numpy array of rewards in [0, 1] (higher = better pose match).
            Uses exp(-mpjpe / sigma) mapping: perfect=1.0, 20mm~0.37, 57mm~0.06.
        """
        sigma = 0.02  # meters; controls reward sensitivity
        scores = []
        for i, (gen_video, ref_video, vid_id) in enumerate(
            zip(generated_videos, ref_videos, video_ids)
        ):
            try:
                ref_joints = self._get_ref_joints(ref_video, vid_id)
                gen_joints = self._video_to_joints(gen_video)

                T = min(gen_joints.shape[0], ref_joints.shape[0])
                mpjpe = self._pelvis_aligned_mpjpe(gen_joints[:T], ref_joints[:T])
                reward = float(np.exp(-mpjpe / sigma))
                scores.append(reward)

            except Exception as e:
                logger.warning(f"GVHMR failed for video {i} ({vid_id}): {e}")
                scores.append(0.0)

        return np.array(scores, dtype=np.float32)
