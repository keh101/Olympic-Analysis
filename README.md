# Olympic-Analysis
A data analysis project of Olympic Games in history.
File 'medal_gif.py' is a Python file that can generate gifs for winter Olympic Games and summer Olympic Games using methods summer_gif() and winter_gif() in the class OlympicMedalGif. Users can customize the start year, end year and gif delay time to create customized gifs. Such gifs show the distribution of total number of medals for nearly each country for each Olympics. Additional Python package includes Cartopy, Seaborn and Imageio. It can also make two horizontal bar charts from method summer_bar_chart() and winter_bar_chart() in the class TopNCountry. These charts visualize the number of medals of gold, silver and bronze w.r.t customized number of countries for the last 20 years. 

# How to run Background info.ipynb
This file is used for generating different kinds of plots in MP4 version to show the detailed data distributions with respect to variables in `120-years-of-olympic-history-athletes-and-results` Dataset. It mainly shows how Olympic Games change throughout the years in the aspects of history, gender inequality problem, conditions of athletes and types of sports.

Before running, please make sure that the following packages are installed:

  ffmpeg

     Download ffmpeg from https://www.ffmpeg.org/download.html into a folder named ffmpeg. 
     
     Install it. And add it to the environment path of your computer.
     
     Make sure to run "animation.FFMpegWriter.isAvailable()" in the first section of Background info.ipynb.
     
     If it is True, then ffmpeg is installed successfully.
  
  os, pandas, matplotlib, numpy, seaborn

     For windows users, use pip install XXX command.
 
 

