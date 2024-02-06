# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 01:29:48 2024

@author: anlun
"""

from moviepy.editor import VideoFileClip

def convert_gif_to_mp4(gif_path, output_path):
    """
    Convert a GIF file to an MP4 file.

    :param gif_path: Path to the GIF file.
    :param output_path: Path where the MP4 file will be saved.
    """
    # Load the GIF
    clip = VideoFileClip(gif_path)

    # Write the clip as an MP4 file
    clip.write_videofile(output_path, codec="libx264", fps=24)

# Example usage
gif_path = 'outputs/output_7.gif'
mp4_output_path = 'outputs/output_7.mp4'

convert_gif_to_mp4(gif_path, mp4_output_path)
