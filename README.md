I present the first author's project of **Old Photo Color** neural network, designed for colorization of single and group images, as well as colorization of video. I would like to note that it can be run both locally on your computer and via Google Colab. Let's take a look at everything in order.
The pre-trained modelcope/iic/cv_ddcolor_image-colorization model is used as the colorization model

![image](https://github.com/user-attachments/assets/fae3c145-f80e-4e12-b3a8-a2c60b0894a6)
![image](https://github.com/user-attachments/assets/d0d078d0-ef3e-4723-bbe2-e0a2a9b098db)
![image](https://github.com/user-attachments/assets/95314c42-b7c3-4f20-98e9-8594bf3c5cfb)
![image](https://github.com/user-attachments/assets/a0a7273c-7f6a-470d-abef-02c0bc540815)


**Launch**. If you will run it on a local machine, you can safely skip this point.

![image](https://github.com/user-attachments/assets/e35fbc48-1b5f-4b56-a137-5045614ed60a)

Here you can choose where the processed images will be saved - Google Disk or Google Colab space. In the first case you will save to the old_photo_color_output folder in the root of your Google Disk. Otherwise, it will be saved to the Google Colab space output folder, but you will be able to download the finished files from the interface to your desktop.

**If you are working locally, all saves will be made to the project output folder**

Once launched, you will see an interface that has 4 tabs.

![image](https://github.com/user-attachments/assets/b0eb6257-0903-4193-b41a-4d2356bb81cd)


**Single** - Designed to work with single files.

**Batch** - Designed to work with image archives. Beforehand you need to pack all necessary files into a ZIP-archive. Source file names must not contain spaces, special characters and alphabetic characters other than Latin. Working with folders inside the archive is not supported.

**Video** - Designed for working with video files. You will get a file WITHOUT an audio track.

**Clear output folder** - Clears the output folder, regardless of where it is located. Be very careful.

**Workflow** - The panel for enabling Enhance and Coloring modes and setting them up.

<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/old_photo_color/blob/main/old_photo_color.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Old Photo Color Colab version</td>
  </tr>
</table>

Update log

v2.0
1. Add Echance Mode
2.Added selection of downloads (zip or individual files) in Batch Mode

v1.0
1. First version

All suggestions and questions can be voiced in the [Telegram-group](https://t.me/+xlhhGmrz9SlmYzg6)

![image](https://github.com/user-attachments/assets/5cf86b6d-e378-4d85-aed1-c48920b6c107)
