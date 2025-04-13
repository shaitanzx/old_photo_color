import gradio as gr
import numpy as np
import argparse
import time
import os
import zipfile
import cv2
import version
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.hub.snapshot_download import snapshot_download
video_name=""
##os.makedirs("input", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
##os.makedirs("frames", exist_ok=True)
model_dir = snapshot_download('iic/cv_ddcolor_image-colorization',cache_dir='checkpoints')
img_colorization = pipeline(task=Tasks.image_colorization,model=model_dir)
HEADER_MD = f"""# Old Photo Color {version.version}

Pre-trained colorization model modelscope/iic/cv_ddcolor_image-colorization is used  

Code and Portable by Shahmatist^RMDA. Other projects can be viewed [here](https://github.com/shaitanzx)
"""
js_func="""
        () => {
            document.body.classList.toggle('dark');
        }
        """
def color(image):
    rgb_image = image[..., ::-1] if image.shape[-1] == 3 else image
    output = img_colorization(rgb_image)
    result = output[OutputKeys.OUTPUT_IMG].astype(np.uint8)
    result = result[...,::-1]
    return result
def delete_input(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                            delete_input(file_path)
                            os.rmdir(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            return
def batch_color(file_in):
##    delete_input('input')                         
    os.makedirs("input", exist_ok=True)
    extract_folder = "input"
    zip_ref=zipfile.ZipFile(file_in.name, 'r')
    zip_ref.extractall(extract_folder)
    zip_ref.close()
    index = 0
    files = sorted(os.listdir(extract_folder))
    while index < len(files):
        if index==0:
            file_name_view = os.path.join(extract_folder, files[index])             
            yield file_name_view

        else:
            file_name_view = os.path.join(extract_folder, files[index])
            file_name_recolor = os.path.join(extract_folder, files[index-1])
            print(f"{index} of {len(files)} processing ({files[index-1]})")
            gr.Info(f"{index} of {len(files)} processing ({files[index-1]})")
            pil_img = Image.fromarray(color(np.array(Image.open(os.path.join(extract_folder, files[index-1])))))
            pil_img.save(f"{output_path}/{os.path.splitext(files[index-1])[0]}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png", optimize=True, compress_level=0)           
            yield file_name_view

        index += 1
    file_name_view=None
    file_name_recolor = os.path.join(extract_folder, files[index-1])
    print(f"{index} of {len(files)} processing ({files[index-1]})")
    gr.Info(f"{index} of {len(files)} processing ({files[index-1]})")
    pil_img = Image.fromarray(color(np.array(Image.open(os.path.join(extract_folder, files[index-1])))))
    pil_img.save(f"{output_path}/{os.path.splitext(files[index-1])[0]}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png", optimize=True, compress_level=0)                          
    yield None
    delete_input('input')
def zip_batch():
    if get_gui_args.google!=True:
        zip_file='outputs.zip'
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, output_path))
        zipf.close()
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, "outputs.zip")
        return file_path, gr.update(visible=False), gr.update(visible=True)
    return None, gr.update(visible=True), gr.update(visible=False)
def batch_video(video):
    global video_name
##    delete_input('frames')                         
    os.makedirs("frames", exist_ok=True)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for i in range(total_frames):
        ret, frame = cap.read()
        cv2.imwrite(f'frames/frame_{i:06d}.png', cv2.cvtColor(color(frame), cv2.COLOR_BGR2RGB))
        yield f'Saved color frame number {i+1} of {total_frames} from videofile'
    cap.release()


    images = [img for img in os.listdir('frames') 
              if img.endswith(f".png")]
    images.sort()
    first_image = cv2.imread(os.path.join('frames', images[0]))    
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name=f"{output_path}/video_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"

    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    
    for i, image_name in enumerate(images):
        image_path = os.path.join('frames', image_name)
        frame = cv2.imread(image_path)
        out.write(frame)
        yield f'Saved color frame number {i} of {total_frames} to videofile'
    out.release()
    yield 'Video colorization complete'
    delete_input('frames')
def video_google():
    global video_name
    if get_gui_args.google==True:
        return None,gr.update(visible=True),gr.update(visible=False)
    return video_name,gr.update(visible=False),gr.update(visible=True)
def clear():
    delete_input(output_path)

with gr.Blocks(title=f"Old Photo Color {version.version}",js=js_func) as demo:
    gr.Markdown(HEADER_MD)
    
    with gr.Tab(label="Single"):
        with gr.Row():
            image1=gr.Image(label='Source image',type='numpy')
            image2=gr.Image(label='Output image',type='numpy',format="png")
        with gr.Row():
            start_single=gr.Button(value='Start single inference')
        start_single.click(fn=color,inputs=image1,outputs=image2)
    with gr.Tab(label="Batch"):
        with gr.Row():
            with gr.Column():
                file_in=gr.File(label='Upload source ZIP-file',show_label=True,file_count='single',file_types=['.zip'],interactive=True) 
                preview= gr.Image(elem_id="input_image", type='filepath', label="Source Image",visible=False)
            with gr.Column():
                with gr.Row(visible=False) as google_batch:
                    gr.Markdown("## Color images are saved to your Google Drive")
                with gr.Row():    
                    download_link_batch = gr.File(label="Download ZIP-file",visible=False)
        with gr.Row():
            start_batch=gr.Button(value='Start batch inference')
        start_batch.click((lambda: (gr.update(visible=False),gr.update(visible=False),gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False))),
                    outputs=[download_link_batch,google_batch,file_in,preview,start_batch]) \
            .then(fn=batch_color,inputs=file_in,outputs=preview) \
            .then(fn=zip_batch,outputs=[download_link_batch,google_batch,download_link_batch]) \
            .then((lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True))),
                    outputs=[file_in,preview,start_batch])
    with gr.Tab(label="Video"):
        with gr.Row():
            with gr.Column():
                file_video=gr.Video(label='Upload source videofile',show_label=True,sources=['upload'], autoplay=True,interactive=True)                 
            with gr.Column():
                with gr.Row():
                    progress_video = gr.Textbox(label="Process",visible=False)
                with gr.Row(visible=False) as google_video:
                    gr.Markdown("## Color video are saved to your Google Drive")
                with gr.Row():
                    download_link_video = gr.File(label="Download videofile",visible=False)
        with gr.Row():
            start_video=gr.Button(value='Start video inference')   
        start_video.click((lambda: (gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(interactive=False))),
                    outputs=[google_video,download_link_video,progress_video,start_video]) \
            .then(fn=batch_video,inputs=file_video,outputs=progress_video) \
            .then(fn=video_google,outputs=[download_link_video,google_video,download_link_video]) \
            .then(lambda: (gr.update(interactive=True)),outputs=start_video)  
    with gr.Tab(label="Clear output folder"):
        with gr.Row():
            clear_output=gr.Button(value='CLEAR OUTPUT FOLDER')   
        clear_output.click(lambda: (gr.update(interactive=False)),outputs=clear_output) \
            .then(fn=clear) \
            .then(lambda: (gr.update(interactive=True)),outputs=clear_output) 
    gr.HTML("<div><p style='text-align:center;'>We are in <a href='https://t.me/+xlhhGmrz9SlmYzg6' target='_blank'>Telegram</a></p> </div>") 
def gui_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--google', action='store_true', help="save caption to google disk")
    parser.add_argument('--share', action='store_true', help="share gradio link")
    return parser.parse_args()
get_gui_args = gui_setup_args()
if get_gui_args.google==True:
        output_path='/content/drive/MyDrive/old_photo_color_output'
else:
        output_path='output'
os.makedirs(output_path, exist_ok=True)
demo.launch(share=get_gui_args.share,inbrowser=True)
