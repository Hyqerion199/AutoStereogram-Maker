import cv2
import os
import shutil
from moviepy.editor import *

def extract_frames(video_path):
    cap= cv2.VideoCapture(video_path)
    i=0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not os.path.exists('images'):
        os.mkdir('images')
        print('images folder created')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('images/frame'+str(i)+'.jpg',frame)
        i+=1
        print(i)
    cap.release()
    cv2.destroyAllWindows()
    return fps

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("audio.wav")

def main():
    video_path = 'input.mp4'
    fps = extract_frames(video_path)
    print(fps)
    extract_audio(video_path)
    print('extract audio done')

    #shutil.rmtree('images')
    os.remove('project_no_audio.avi')
    os.remove('audio.wav')
    print('done')

if __name__ == "__main__":
    main()