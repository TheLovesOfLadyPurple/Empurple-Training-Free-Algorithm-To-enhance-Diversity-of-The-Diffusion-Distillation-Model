from pytorch_fid import fid_score
from PIL import Image
import numpy as np
import os

def resize_image(image_path, target_size):
    image = Image.open(image_path)
    return image.resize(target_size, Image.Resampling.BICUBIC)

def pad_image(image, size, fill_color=(0, 0, 0)):
    new_image = Image.new("RGB", size, fill_color)
    new_image.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return new_image

def preprocess_and_save_images(image_dir, target_size, output_dir):
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else: 
        return 
    
    # 处理每张图像并保存到输出目录
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        processed_image = resize_image(image_path, target_size)
        
        # 生成保存路径
        output_path = os.path.join(output_dir, image_name)
        
        # 保存图像
        processed_image.save(output_path)


if __name__ == '__main__':
    gen_img_path = "./gen_img_val_fetch_latent_val/samples-intermediate-ts00857-step-12-5.5-Flash"
    org_img_path = "./crops_org_coco" 
    output_path = "./crops_org_coco_bicubic" 
    preprocess_and_save_images(image_dir=org_img_path,target_size=(512,512),output_dir=output_path)
    
    score = fid_score.calculate_fid_given_paths(paths=[gen_img_path, output_path],device='cuda:0',dims=2048,batch_size=128)
    print(score)