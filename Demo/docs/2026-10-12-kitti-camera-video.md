### Goal

Add real matching left/right colour KITTI images to the existing 174-frame video.

### Assumptions

- The user supplied `Downloads/kitti_training_174frames_images.zip` from Colab,
  containing both cameras for all 174 Object IDs. Use that archive locally.
- Reuse the saved predictions and classification maps; RGB is display-only.
- Export a separate five-panel MP4 at 1920x1080, 10 FPS, with stereo cameras above
  the original three panels; optionally select only the left camera.

### Plan

1. Validate and extract the user-supplied Colab ZIP.
   - Files: `Code/tools/visualization/export_kitti_camera_video.py`,
     `Code/tests/test_kitti_camera_video.py`.
   - Change: Extract only exact training camera paths; check CRC, source names,
     order, RGB dimensions and hashes. Retain the optional remote downloader.
   - Verify: Focused pytest; all 348 actual PNGs match the saved frame list.
2. Add camera layout and replay cached inference.
   - Files: `Code/tools/visualization/export_kitti_sequence_video.py`,
     `Code/tools/visualization/export_kitti_camera_video.py`.
   - Change: Optional stereo camera layout; pair image by exact Object ID and
     verify cached scan/config/map hashes before replaying.
   - Verify: Tests reject missing camera frames and check unchanged image RGB.
3. Export and inspect the real video.
   - Files: `Code/tools/visualization/README_kitti_overview.md`,
     `Result/artifacts/kitti_sequence_video_camera/`.
   - Change: Encode matching images, LiDAR, BEV and heatmap into a separate MP4.
   - Verify: All 174 source pairs; ffprobe 174 frames/17.4s/10FPS/1920x1080;
     decode MP4 and inspect preview/contact sheet. Run related visualization tests.

### Risks & mitigations

- Incorrect image pairing: exact training/image_2/<ObjectID>.png, no nearest-frame fallback.
- Full archive transfer: reject HTTP responses that do not honour Range; bound reads.
- Semantic confusion: record RGB display-only, no new model inference or RGB detector.

### Rollback plan

Keep the original renderer layout/API as defaults and original video untouched.
New images and video live in ignored artifact directories.
