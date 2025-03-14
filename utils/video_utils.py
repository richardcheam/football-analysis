import cv2

def read_vdo(video_path):
    """
    Reads a video file frame by frame and stores all frames in a list.
    Args:
        video_path (str): Path to the video file.
    Returns:
        list: A list of frames (numpy arrays) extracted from the video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    # Check if the video file is successfully opened
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()
        # If no more frames are available, break the loop
        if not ret:
            break
        # Append the frame to the list
        frames.append(frame)
    # Release the video capture object to free resources
    cap.release()
    return frames

def save_vdo(output_video_frames, output_video_path):
    """
    Saves a list of frames as a video file.
    Args:
        output_video_frames (list): List of frames (numpy arrays) to be saved as a video.
        output_video_path (str): Path where the output video will be saved.
    Returns:
        None
    """
    # Define the video codec (XVID format)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # Get the frame dimensions (width and height) from the first frame
    frame_width = output_video_frames[0].shape[1]  # Width (x-axis)
    frame_height = output_video_frames[0].shape[0]  # Height (y-axis)
    # Initialize the VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (frame_width, frame_height))
    # Write each frame to the video file
    for frame in output_video_frames:
        out.write(frame)
    # Release the VideoWriter to free resources
    out.release()
