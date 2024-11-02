import argparse
import time
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer


def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=5, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()
    print("start", int(time.mktime(time.localtime(time.time()))))

    # determine models according to model names
    # RealESRGAN_x2plus_netD.pth
    # RealESRGAN_x4plus_netD.pth
    # RealESRGANv2-animevideo-xsx2.pth
    # RealESRGANv2-animevideo-xsx4.pth
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    if args.model_name in ['RealESRGAN_x4plus_netD']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in ['realesr-animevideov3']:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name in ['realesr-general-x4v3', 'realesr-general-wdn-x4v3']:  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name in ['RealESRGANv2-animevideo-xsx4']:  # x4 VGG-style model (S size)
        # model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name in ['RealESRGANv2-animevideo-xsx2']:  # x4 VGG-style model (S size)
        # model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    print("gfpgan", int(time.mktime(time.localtime(time.time()))))

    face_enhancer = GFPGANer(
        model_path='experiments/pretrained_models/GFPGANv1.3.pth',
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)

    print("init done", int(time.mktime(time.localtime(time.time()))))

    face_enhancer.face_helper.clean_all()
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    face1 = cv2.imread('compose/44.jpg', cv2.IMREAD_UNCHANGED)
    bg1 = cv2.imread('compose/2.jpeg', cv2.IMREAD_UNCHANGED)
    face_enhancer.face_helper.read_image(bg1)

    print("landmarks", int(time.mktime(time.localtime(time.time()))))

    face_enhancer.face_helper.get_face_landmarks_5(
        only_center_face=False, eye_dist_threshold=5)
    # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
    # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
    # align and warp each face
    face_enhancer.face_helper.align_warp_face()

    face_enhancer.face_helper.add_restored_face(face1)
    print("landmarks done", int(time.mktime(time.localtime(time.time()))))

    face_enhancer.face_helper.get_inverse_affine(None)
    composed = face_enhancer.face_helper.paste_faces_to_input_image(upsample_img=bg1)

    cv2.imwrite('compose/composed33.jpeg', composed)
    print("work done", int(time.mktime(time.localtime(time.time()))))

    # =====
    face_enhancer = GFPGANer(
        model_path='experiments/pretrained_models/GFPGANv1.3.pth',
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)
    face_enhancer.face_helper.clean_all()

    face1 = cv2.imread('compose/22.jpeg', cv2.IMREAD_UNCHANGED)
    face2 = cv2.imread('compose/44.jpg', cv2.IMREAD_UNCHANGED)
    bg1 = cv2.imread('compose/640.jpg', cv2.IMREAD_UNCHANGED)
    bg2 = cv2.imread('compose/33.jpeg', cv2.IMREAD_UNCHANGED)
    face_enhancer.face_helper.read_image(bg1)

    print("landmarks", int(time.mktime(time.localtime(time.time()))))

    face_enhancer.face_helper.get_face_landmarks_5(
        only_center_face=False, eye_dist_threshold=5)
    # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
    # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
    # align and warp each face
    face_enhancer.face_helper.align_warp_face()

    face_enhancer.face_helper.add_restored_face(face1)
    face_enhancer.face_helper.add_restored_face(face2)
    print("landmarks done", int(time.mktime(time.localtime(time.time()))))

    face_enhancer.face_helper.get_inverse_affine(None)
    composed = face_enhancer.face_helper.paste_faces_to_input_image(upsample_img=bg2)

    cv2.imwrite('compose/composed33.jpeg', composed)
    print("work done", int(time.mktime(time.localtime(time.time()))))

    print("copy1", int(time.mktime(time.localtime(time.time()))))
    bbb = face_enhancer
    bbb.face_helper.clean_all()
    print("copy2", int(time.mktime(time.localtime(time.time()))))


if __name__ == '__main__':
    main()
