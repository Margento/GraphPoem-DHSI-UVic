import os
import cv2
import requests
from scenedetect import ContentDetector, SceneManager, open_video

# Configurations
# VIDEO_PATH = "/Users/raluca/310/MARGENTO_GraphPoem_DHSI_2022.mp4"
# VIDEO_PATH = "/Users/raluca/310/MARGENTO_GraphPoem_DHSI_2021.mp4"
# VIDEO_PATH = "/Users/raluca/310/MARGENTO_GraphPoem_DHSI_2020.mp4"
# VIDEO_PATH = "Margento_GraphPoem_DHSI_2023.mp4"
# VIDEO_PATH = "costin_graphpoem_dhsi.mp4"
# VIDEO_PATH = "comm_singularity_graphpoem_dhsi_uvic.mp4"
VIDEO_PATH = "graphpoem_dhsi_uvic_1.mp4"
API_URL = "http://127.0.0.1:8000/predict"
# OUTPUT_DIR = "./graphpoem_dhsi22_extracted_shots"
# OUTPUT_DIR = "./graphpoem_dhsi21_extracted_shots"
# OUTPUT_DIR = "./graphpoem_dhsi20_extracted_shots"
# OUTPUT_DIR = "./graphpoem_dhsi23_extracted_shots"
# OUTPUT_DIR = "./costin_graphpoem_dhsi_shots"
# OUTPUT_DIR = "./comm_singularity_graphpoem_dhsi_shots"
OUTPUT_DIR = "./graphpoem_dhsi_uvic_1_shots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_and_upload_shots(video_path: str):
    print(f"🎬 Opening video: {video_path}")
    video = open_video(video_path)
    scene_manager = SceneManager()
    
    # ContentDetector finds cuts by comparing differences in consecutive frames
    # threshold=27.0 is standard. Lower means more sensitive (more clips)
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    print("🔍 Analyzing video for shot changes... (This runs fast on M2 via OpenCV)")
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    
    print(f"✨ Detected {len(scene_list)} unique scenes/cuts.")
    
    # Re-open with standard OpenCV to fast-forward and extract frames
    cap = cv2.VideoCapture(video_path)
    
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        start_time_sec = scene[0].get_seconds()
        
        # Jump directly to the first frame of the new scene (Zero overhead seeking)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Save shot locally for verification
        shot_filename = f"shot_{i:04d}_time_{int(start_time_sec)}s.jpg"
        shot_path = os.path.join(OUTPUT_DIR, shot_filename)
        cv2.imwrite(shot_path, frame)
        
        # Stream the extracted shot straight into your MPS Starlette server
        print(f"🚀 Uploading shot {i+1}/{len(scene_list)} to MPS Server...")
        try:
            with open(shot_path, 'rb') as f:
                response = requests.post(API_URL, files={'file': (shot_filename, f, 'image/jpeg')})
                
            if response.status_code == 200:
                res_data = response.json()
                print(f"  └ ✅ Match found: {res_data['results']['prediction']} ({res_data['results']['confidence']:.2f})")
            else:
                print(f"  └ ❌ Server error: {response.text}")
        except Exception as e:
            print(f"  └ ❌ Failed to hit API: {e}")
            
    cap.release()
    print("\n🏁 Video processing and smart sampling complete!")

if __name__ == "__main__":
    if os.path.exists(VIDEO_PATH):
        extract_and_upload_shots(VIDEO_PATH)
    else:
        print(f"❌ Error: Video file not found at {VIDEO_PATH}. Please fix the path in the script!")
