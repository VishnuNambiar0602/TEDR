import requests
import json
import os

url = 'http://127.0.0.1:8000/analyze_video'

video_name = 'test_video.mp4'
if not os.path.exists(video_name):
    # Try parent directory
    video_name = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_video.mp4')

if not os.path.exists(video_name):
    raise FileNotFoundError(f"Could not find test video: test_video.mp4")

files = {'file': open(video_name, 'rb')}
data = {'frame_skip': '2'}
try:
    response = requests.post(url, files=files, data=data)
    res_json = response.json()
    print(json.dumps(res_json))
    
    if res_json.get('success') and res_json.get('download_url'):
        download_url = 'http://127.0.0.1:8000' + res_json['download_url']
        out_file = 'downloaded_test_output.mp4'
        r = requests.get(download_url)
        with open(out_file, 'wb') as f:
            f.write(r.content)
        print(f"File Exists: {os.path.exists(out_file)}")
        print(f"File Size: {os.path.getsize(out_file)}")
except Exception as e:
    print(f"Error: {e}")
