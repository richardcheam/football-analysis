from utils import read_vdo, save_vdo
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
import numpy as np
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read Video
    video_frames = read_vdo('input/08fd33_4.mp4') #a single frame (1080, 1920, 3) #full vod (750, 1080, 1920, 3)
    
    # Init Tracker
    tracker = Tracker('models/best.pt')
    #20 frames at a time as batch_size = 20
    #shape (1, 3, 384, 640) → The shape of the input tensor used for inference:
    #1 → Batch size (1 frame at a time)
    #3 → Number of color channels RGB
    #384x640 → The resized frame resolution

    ##### Track process #####
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True, 
                                       stub_path='stubs/track_stubs.pkl')
    #print(len(tracks["players"][0])) 
    ##is a dict of bbox for each frame 0
    ##{ 1: {'bbox': [1348.13818359375, 604.8981323242188, 1378.079345703125, 671.2489624023438]}, 2: {....}, n: {...} }

    # Get object positions 
    tracker.add_position_to_tracks(tracks)    

    ########## Crop image of a player ##########
    # for track_id, player in tracks['players'][0].items(): #first frame, only need one image
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     #crop bbox from frame to get player
    #     cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] #y1:y2, x1:x2

    #     #save image 
    #     cv2.imwrite(f'output/cropped_img.jpg', cropped_img)
    #     # break after getting cropped img
    #     break 
    ##############################################
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0]) #bbox for each player in the first frame
    #print(tracks['players'][0])
    #iterate for each frame and each player
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            # add new value 'team' to dictionary
            tracks['players'][frame_num][player_id]['team'] = team
            # add 'team_color' to the dict
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    ##### Visualize tracking ###### 
    # Draw Output/Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save Video
    save_vdo(output_video_frames, 'output/output_video.mp4')

if __name__ == '__main__':
    main()