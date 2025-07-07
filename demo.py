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

import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from typing import List, Union, Dict, Set, Tuple
sys.path.append(os.path.abspath('CodeFormer'))
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY



video_name=""
os.makedirs("checkpoints", exist_ok=True)
model_dir = snapshot_download('iic/cv_ddcolor_image-colorization',cache_dir='checkpoints')
pretrain_model_url = {
        'codeformer': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/codeformer.pth',
        'detection': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/detection_Resnet50_Final.pth',
        'parsing': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/parsing_parsenet.pth',
        'realesrgan': 'https://huggingface.co/shaitanzx/FooocusExtend/resolve/main/RealESRGAN_x2plus.pth'
    }
    # download weights
if not os.path.exists('extentions/CodeFormer/weights/CodeFormer/codeformer.pth'):
        load_file_from_url(url=pretrain_model_url['codeformer'], model_dir='CodeFormer/weights/CodeFormer', progress=True, file_name=None)
if not os.path.exists('extentions/CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
        load_file_from_url(url=pretrain_model_url['detection'], model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
if not os.path.exists('extentions/CodeFormer/weights/facelib/parsing_parsenet.pth'):
        load_file_from_url(url=pretrain_model_url['parsing'], model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
if not os.path.exists('extentions/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
        load_file_from_url(url=pretrain_model_url['realesrgan'], model_dir='CodeFormer/weights/realesrgan', progress=True, file_name=None)

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
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler
def get_image(input_data: Union[list, np.ndarray]) -> np.ndarray:
    if isinstance(input_data, (list, tuple)) and len(input_data) > 0:        
        return input_data[0],True
    elif isinstance(input_data, np.ndarray):       
        return input_data,False

def color(image,faceenchance_enabled,face_align,background_enhance,
                face_upsample,codeformer_fidelity,coloring_enabled,
                upscale):
    if faceenchance_enabled:
        codeform_array=[]
        upsampler = set_realesrgan()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
            ).to(device)
        ckpt_path = "CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()
        try: # global try
                # take the default setting for the demo
                only_center_face = False
                draw_box = False
                detection_model = "retinaface_resnet50"

#        print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)
                face_align = face_align if face_align is not None else True
                background_enhance = background_enhance if background_enhance is not None else True
                face_upsample = face_upsample if face_upsample is not None else True
                upscale = upscale if (upscale is not None and upscale > 0) else 2

                has_aligned = not face_align
                upscale = 1 if has_aligned else upscale
                img_array,generator=get_image(image)
                img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                print('\timage size:', img.shape)

                upscale = int(upscale) # convert type to int
                #if upscale > 4: # avoid memory exceeded due to too large upscale
                #    upscale = 4 
                #if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
                #    upscale = 2 
                #if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
                #    upscale = 1
                #    background_enhance = False
                #    face_upsample = False

                face_helper = FaceRestoreHelper(
                    upscale,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model=detection_model,
                    save_ext="png",
                    use_parse=True,
                    device=device,
                )
                bg_upsampler = upsampler if background_enhance else None
                face_upsampler = upsampler if face_upsample else None

                if has_aligned:
                    # the input faces are already cropped and aligned

                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    face_helper.is_gray = is_gray(img, threshold=5)
                    if face_helper.is_gray:
                        print('\tgrayscale input: True')
                    face_helper.cropped_faces = [img]
                else:
                    face_helper.read_image(img)
                    # get face landmarks for each face
                    num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, resize=640, eye_dist_threshold=5
                    )
                    print(f'\tdetect {num_det_faces} faces')
                    # align and warp each face
                    face_helper.align_warp_face()

                # face restoration for each cropped face
                for idx, cropped_face in enumerate(face_helper.cropped_faces):
                    # prepare data
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:
                        with torch.no_grad():
                            output = codeformer_net(
                                cropped_face_t, w=codeformer_fidelity, adain=True
                            )[0]
                            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    except RuntimeError as error:
                        print(f"Failed inference for CodeFormer: {error}")
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    restored_face = restored_face.astype("uint8")
                    face_helper.add_restored_face(restored_face, cropped_face)

                # paste_back
                if not has_aligned:
                    # upsample the background
                    if bg_upsampler is not None:
                        # Now only support RealESRGAN for upsampling background
                        bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
                    else:
                        bg_img = None
                    face_helper.get_inverse_affine(None)
                    # paste each restored face to the input image
                    if face_upsample and face_upsampler is not None:
                        restored_img = face_helper.paste_faces_to_input_image(
                            upsample_img=bg_img,
                            draw_box=draw_box,
                            face_upsampler=face_upsampler,
                        )
                    else:
                        restored_img = face_helper.paste_faces_to_input_image(
                            upsample_img=bg_img, draw_box=draw_box
                        )
                else:
                    restored_img = restored_face

                restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                codeform_array.append(restored_img)
                if generator:
                    image = codeform_array
                else:
                    image = np.array(restored_img)

        except Exception as error:
                print('Global exception', error)
                return None, None
    if coloring_enabled:
        rgb_image = image[..., ::-1] if image.shape[-1] == 3 else image
        output = img_colorization(rgb_image)
        result = output[OutputKeys.OUTPUT_IMG].astype(np.uint8)
        image = result[...,::-1]
    return image
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
def batch_color(file_in,files,enable_zip,faceenchance_enabled,face_align,background_enhance,
                face_upsample,codeformer_fidelity,coloring_enabled,
                upscale):
##    delete_input('input')                         
    os.makedirs("input", exist_ok=True)
    extract_folder = "input"
    if enable_zip:
        zip_ref=zipfile.ZipFile(file_in.name, 'r')
        zip_ref.extractall(extract_folder)
        zip_ref.close()
    else:
        for file in files:
           original_name = getattr(file, 'orig_name', os.path.basename(file.name))
           save_path = os.path.join("input", original_name)
           try:
               with open(file.name, 'rb') as src:
                   with open(save_path, 'wb') as dst:
                       while True:
                           chunk = src.read(8192)  # Читаем по 8KB за раз
                           if not chunk:
                               break
                           dst.write(chunk)
           except Exception as e:
               print(f"Ошибка при копировании {original_name}: {str(e)}")
        
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
            pil_img = Image.fromarray(color(np.array(Image.open(os.path.join(extract_folder, files[index-1]))),
                faceenchance_enabled,face_align,background_enhance,
                face_upsample,codeformer_fidelity,coloring_enabled,
                upscale))
            pil_img.save(f"{output_path}/{os.path.splitext(files[index-1])[0]}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png", optimize=True, compress_level=0)           
            yield file_name_view

        index += 1
    file_name_view=None
    file_name_recolor = os.path.join(extract_folder, files[index-1])
    print(f"{index} of {len(files)} processing ({files[index-1]})")
    gr.Info(f"{index} of {len(files)} processing ({files[index-1]})")
    pil_img = Image.fromarray(color(np.array(Image.open(os.path.join(extract_folder, files[index-1]))),
                faceenchance_enabled,face_align,background_enhance,
                face_upsample,codeformer_fidelity,coloring_enabled,
                upscale))
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
def batch_video(video,faceenchance_enabled,face_align,background_enhance,
                face_upsample,codeformer_fidelity,coloring_enabled,
                upscale):
    global video_name
                        
    os.makedirs("frames", exist_ok=True)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for i in range(total_frames):
        ret, frame = cap.read()
        cv2.imwrite(f'frames/frame_{i:06d}.png', cv2.cvtColor(color(frame,faceenchance_enabled,face_align,
                background_enhance,face_upsample,codeformer_fidelity,coloring_enabled,
                upscale), cv2.COLOR_BGR2RGB))
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

def workflow():
    with gr.Accordion('Workflow', open=False)  as gen_acc:
        with gr.TabItem(label='Enhance'):
            with gr.Row():
                enchance_enabled = gr.Checkbox(label="Enabled", value=False,interactive=True)
            with gr.Row():
                with gr.Column():
                    faceenchance_preface=gr.Checkbox(value=True, label="Pre_Face_Align",interactive=True)
                    faceenchance_background_enhance=gr.Checkbox(label="Background Enchanced", value=True,interactive=True)
                    faceenchance_face_upsample=gr.Checkbox(label="Face Upsample", value=True,interactive=True)
                with gr.Column(): 
                    upscale = gr.Slider(label='Upscale', minimum=1.0, maximum=4.0, step=1.0, value=1,interactive=True)
                    faceenchance_fidelity =gr.Slider(label='Fidelity', minimum=0, maximum=1, value=0.5, step=0.01, info='0 for better quality, 1 for better identity (default=0.5)',interactive=True)
        with gr.TabItem(label='Coloring'):
            with gr.Row():
                coloring_enabled = gr.Checkbox(label="Enabled", value=False,interactive=True)
    def gen_acc_name(enchance, coloring):
                    main_name = "Workflow" + (f" — {', '.join(filter(None, ['Enhance enabled' if enchance else None, 'Coloring enabled' if coloring else None]))}" if any([enchance, coloring]) else "")
                    return gr.update(label=main_name)
    enchance_enabled.change(gen_acc_name,inputs=[enchance_enabled,coloring_enabled],
                        outputs=[gen_acc],queue=False)
    coloring_enabled.change(gen_acc_name,inputs=[enchance_enabled,coloring_enabled],
                        outputs=[gen_acc],queue=False)

                
    return (enchance_enabled,faceenchance_preface,faceenchance_background_enhance,
            faceenchance_face_upsample,faceenchance_fidelity,coloring_enabled,upscale)



with gr.Blocks(title=f"Old Photo Color {version.version}",js=js_func) as demo:
    gr.Markdown(HEADER_MD)
    
    with gr.Tab(label="Single"):
        with gr.Row():
            image1=gr.Image(label='Source image',type='numpy')
            image2=gr.Image(label='Output image',type='numpy',format="png")
        with gr.Row():
            (enchance_enabled,faceenchance_preface,faceenchance_background_enhance,
                faceenchance_face_upsample,faceenchance_fidelity,coloring_enabled,
                upscale) = workflow()
        with gr.Row():
            start_single=gr.Button(value='Start single inference')
        start_single.click(lambda: (gr.update(interactive=False)),outputs=start_single) \
            .then(fn=color,inputs=[image1,enchance_enabled,faceenchance_preface,
                            faceenchance_background_enhance,faceenchance_face_upsample,
                            faceenchance_fidelity,coloring_enabled,upscale],
                            outputs=image2) \
            .then(lambda: (gr.update(interactive=True)),outputs=start_single)
    with gr.Tab(label="Batch"):
        with gr.Row():
            with gr.Column():
                file_in=gr.File(label='Upload source ZIP-file',show_label=True,file_count='single',file_types=['.zip'],interactive=True,visible=False,height=268) 
                files = gr.Files(label="Drag (Select) 1 or more photos of your face",
                        file_types=["image"],visible=True,interactive=True,height=268)
                preview= gr.Image(elem_id="input_image", type='filepath', label="Source Image",visible=False)            
            with gr.Column():
                with gr.Row(visible=False) as google_batch:
                    gr.Markdown("## Color images are saved to your Google Drive")
                with gr.Row():    
                    download_link_batch = gr.File(label="Download ZIP-file",visible=False)
        with gr.Row():
                enable_zip = gr.Checkbox(label="Upload ZIP-file", value=False)
        with gr.Row():
            (enchance_enabled_b,faceenchance_preface_b,faceenchance_background_enhance_b,
                faceenchance_face_upsample_b,faceenchance_fidelity_b,coloring_enabled_b,
                upscale_b) = workflow()
        with gr.Row():
            start_batch=gr.Button(value='Start batch inference')
        enable_zip.change(lambda x: (gr.update(visible=x),gr.update(visible=not x)), inputs=enable_zip,
                                        outputs=[file_in,files], queue=False)
        start_batch.click((lambda: (gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False))),
                    outputs=[download_link_batch,google_batch,file_in,files,preview,start_batch]) \
            .then(fn=batch_color,inputs=[file_in,files,enable_zip,enchance_enabled_b,faceenchance_preface_b,
                faceenchance_background_enhance_b,
                faceenchance_face_upsample_b,faceenchance_fidelity_b,coloring_enabled_b,
                upscale_b],outputs=preview) \
            .then(fn=zip_batch,outputs=[download_link_batch,google_batch,download_link_batch]) \
            .then((lambda x: (gr.update(visible=x), gr.update(visible=not x), gr.update(visible=False), gr.update(interactive=True))),
                    inputs=enable_zip,
                    outputs=[file_in,files,preview,start_batch])
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
            (enchance_enabled_v,faceenchance_preface_v,faceenchance_background_enhance_v,
                faceenchance_face_upsample_v,faceenchance_fidelity_v,coloring_enabled_v,
                upscale_v) = workflow()
        with gr.Row():
            start_video=gr.Button(value='Start video inference')   
        start_video.click((lambda: (gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(interactive=False))),
                    outputs=[google_video,download_link_video,progress_video,start_video]) \
            .then(fn=batch_video,inputs=[file_video,enchance_enabled_v,faceenchance_preface_v,
                faceenchance_background_enhance_v,
                faceenchance_face_upsample_v,faceenchance_fidelity_v,coloring_enabled_v,
                upscale_v],outputs=progress_video) \
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
