# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 19:23:45 2024

@author: anlun
"""

import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

sns.set(style="darkgrid", context='talk')

def get_video_properties(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        return None, None, None, None, "Unknown"
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((codec_code >> 8 * i) & 0xFF) for i in range(4)])
    duration_seconds = frames / fps
    cap.release()
    return round(duration_seconds / 60, 2), f"{width}x{height}", round(width / height, 2), round(fps, 2), codec

def seconds_to_dhms(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(days)} days, {int(hours):02}:{int(minutes):02}:{int(seconds):02}"

video_dir = '4-Data download/Videos'
output_dir = '5-Data exploration/Outputs'

video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]

video_data = []
for file in tqdm(video_files, desc="Analyzing Videos"):
    duration, resolution, aspect_ratio, fps, codec = get_video_properties(file)
    if duration is not None:
        video_data.append({
            "Video Name": os.path.basename(file),
            "Duration (minutes)": duration,
            "Resolution": resolution,
            "Aspect Ratio": aspect_ratio,
            "Frame Rate (fps)": fps,
            "Codec": codec
        })

df = pd.DataFrame(video_data)

def save_plot(filename, data, x, kind, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(12, 8))
    if kind == 'hist':
        sns.histplot(data[x], bins=int(max(data[x]) // 30), ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(title, fontsize=18, fontweight='bold')
        bin_edges = range(0, int(max(data[x])) + 30, 30)
        bin_labels = [f"{edge}-{edge + 30}" for edge in bin_edges[:-1]]
        ax.set_xticks([(edge + edge + 30) / 2 for edge in bin_edges[:-1]])
        ax.set_xticklabels(bin_labels, rotation=45)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')
    elif kind == 'count':
        sns.countplot(y=x, data=data, ax=ax, palette='viridis', hue=x, legend=False)
        ax.set_title(title, fontsize=18, fontweight='bold')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=12, color='black', xytext=(5, 0),
                        textcoords='offset points')
    elif kind == 'box':
        sns.boxplot(x=data[x], ax=ax, color='lightgreen')
        ax.set_title(title, fontsize=18, fontweight='bold')
        stats = data[x].describe()
        stats_text = f"Count: {stats['count']:.0f}\nMean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}\nMin: {stats['min']:.2f}\n25%: {stats['25%']:.2f}\n50% (Median): {stats['50%']:.2f}\n75%: {stats['75%']:.2f}\nMax: {stats['max']:.2f}"
        ax.text(0.95, 0.85, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

histogram_path = save_plot('duration_histogram.png', df, 'Duration (minutes)', 'hist', 'Histogram of Video Durations', 'Duration Range (minutes)', 'Number of Videos')
frame_rate_plot_path = save_plot('frame_rate_histogram.png', df, 'Frame Rate (fps)', 'count', 'Histogram of Frame Rates', 'Frame Rate (fps)', 'Number of Videos')
aspect_ratio_plot_path = save_plot('aspect_ratio_histogram.png', df, 'Aspect Ratio', 'count', 'Histogram of Aspect Ratios', 'Aspect Ratio', 'Number of Videos')
resolution_plot_path = save_plot('resolution_count.png', df, 'Resolution', 'count', 'Count of Video Resolutions', 'Resolution', 'Number of Videos')
boxplot_path = save_plot('duration_boxplot.png', df, 'Duration (minutes)', 'box', 'Boxplot of Video Durations', 'Duration (minutes)', '')

def add_image_to_pdf(canvas, image_path, y_offset):
    img = ImageReader(image_path)
    canvas.drawImage(img, 30, y_offset, width=550, height=350, preserveAspectRatio=True)

def create_pdf_report():
    c = canvas.Canvas(os.path.join(output_dir, 'video_analysis_report.pdf'), pagesize=letter)
    width, height = letter

    add_image_to_pdf(c, histogram_path, height - 400)
    c.showPage()
    add_image_to_pdf(c, frame_rate_plot_path, height - 400)
    c.showPage()
    add_image_to_pdf(c, aspect_ratio_plot_path, height - 400)
    c.showPage()
    add_image_to_pdf(c, resolution_plot_path, height - 400)
    c.showPage()
    add_image_to_pdf(c, boxplot_path, height - 400)
    c.showPage()

    c.drawString(30, height - 40, "Summary Statistics:")
    total_duration_seconds = df['Duration (minutes)'].sum() * 60
    c.drawString(30, height - 60, f"Total Videos: {len(df)}")
    c.drawString(30, height - 80, f"Average Duration: {round(df['Duration (minutes)'].mean(), 2)} minutes")
    c.drawString(30, height - 100, f"Median Duration: {round(df['Duration (minutes)'].median(), 2)} minutes")
    c.drawString(30, height - 120, f"Standard Deviation: {round(df['Duration (minutes)'].std(), 2)} minutes")
    c.drawString(30, height - 140, f"Total Time: {seconds_to_dhms(total_duration_seconds)}")
    
    c.save()

create_pdf_report()
print(f"PDF report generated: {os.path.join(output_dir, 'video_analysis_report.pdf')}")
