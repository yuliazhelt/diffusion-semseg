import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

import sys
from omegaconf import OmegaConf
from einops import rearrange, repeat

from ldm.util import instantiate_from_config
from models.util import UNetWrapper, TextAdapter
import json

@SEGMENTORS.register_module()
class CustomVPD(BaseSegmentor):
    """
    EncoderDecoder segmentor from https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/segmentors/encoder_decoder.py

    VPD's feature extraction and inference involve encoding images with an encoder and then
    passing them through a UNet, along with class embeddings and text adaptation for the segmentation task

    The VPD class has customized method for feature extraction (extract_feat) 

    Added features from MetaPrompts (meta prompts instead of class embeddings, refinement steps),
    and from TADP (BLIP-2 captions instead of class embeddings)
    """

    def __init__(
        self,
        decode_head,
        sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
        unet_config=dict(),
        class_embedding_path='checkpoints/class_embeddings.pth',
        refine_step=0,
        caption_type='unaligned', # ['unaligned', 'blip', 'meta_prompts', 'clip_probs']
        num_prompt=None,
        blip_caption_path=None,
        clip_probs_path=None,
        clip_captions_path=None,
        gamma_init_value=1e-4,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        ldm_dim=[320, 790, 1430, 1280],
        **args
    ):
        super().__init__(init_cfg)

        config = OmegaConf.load('stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
        config.model.params.ckpt_path = sd_path
        # config.model.params.cond_stage_config.target = 'ldm.modules.encoders.modules.AbstractEncoder'
        
        # prepare the unet        
        sd_model = instantiate_from_config(config.model)

        self.refine_step = refine_step
        self.caption_type = caption_type
        self.blip_caption_path = blip_caption_path
        self.clip_probs_path = clip_probs_path
        self.clip_captions_path = clip_captions_path

        assert self.caption_type in ['unaligned', 'blip', 'meta_prompts', 'clip_probs', 'clip_captions']

        print(f'Unet refine step: {self.refine_step}  Caption text usage: {self.caption_type}')
        
        for i in range(self.refine_step):
            if i > 0:
                cross_unet_conv = sd_model.model.diffusion_model.out
                setattr(self, f"cross_unet_conv{i + 1}", cross_unet_conv)
            t = nn.Parameter(torch.randn(1, 1280), requires_grad=True)
            setattr(self, f"t{i + 1}", t)

 
        self.encoder_vq = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, temb_refine=(refine_step > 0), **unet_config)
        sd_model.model = None
        sd_model.first_stage_model = None


        # class embeddings & text adapter
        if self.caption_type == 'unaligned':
            class_embeddings = torch.load(class_embedding_path)
            self.register_buffer('class_embeddings', class_embeddings)
            text_dim = self.class_embeddings.shape[-1]
            self.gamma = nn.Parameter(torch.full(size=(text_dim,), fill_value=gamma_init_value))
            self.text_adapter = TextAdapter(text_dim=text_dim)

        elif self.caption_type == 'blip':
            with open(self.blip_caption_path, 'r') as f:
                print('Loaded blip captions!')
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])
                for param in sd_model.cond_stage_model.parameters():
                    param.requires_grad = False
                text_dim = 768

            self.num_classes = decode_head['num_classes']
            enc16_size, enc32_size = self.compute_decoder_head_shapes()
            neck['in_channels'][1] = enc16_size
            neck['in_channels'][2] = enc32_size

        elif self.caption_type == 'meta_prompts':
            self.num_prompt = num_prompt 
            text_dim = 768
            self.meta_prompts = nn.Parameter(torch.randn(num_prompt, text_dim), requires_grad=True)

        elif self.caption_type == 'clip_probs':
            with open(self.clip_probs_path, 'r') as f:
                print('Loaded clip probs!')
                self.clip_probs = json.load(f)

            class_embeddings = torch.load(class_embedding_path)
            self.register_buffer('class_embeddings', class_embeddings)
            text_dim = self.class_embeddings.shape[-1]

        elif self.caption_type == 'clip_captions':
            with open(self.clip_captions_path, 'r') as f:
                print('Loaded clip captions!')
                self.clip_captions = json.load(f)
                # get max length
                self.clip_max_length = max([len(caption) for caption in self.clip_captions])
                for param in sd_model.cond_stage_model.parameters():
                    param.requires_grad = False
                text_dim = 768

            self.num_classes = decode_head['num_classes']
            enc16_size, enc32_size = self.compute_decoder_head_shapes()
            neck['in_channels'][1] = enc16_size
            neck['in_channels'][2] = enc32_size

        # del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        self.sd_model = sd_model

        if neck is not None:
            self.neck = builder.build_neck(neck)

        for i in range(len(ldm_dim)):
            upconv = nn.ModuleList([
            nn.Conv2d(ldm_dim[i], 768, 3, 1, 1),
            nn.GroupNorm(16, 768),
            nn.ReLU()])
            setattr(self, f"upconv{i + 1}", upconv)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def get_crossattn(self, latents, img, img_metas):
        
        if self.caption_type == 'unaligned':
            c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma) # NOTE: here the c_crossattn should be expand_dim as latents
        
        elif self.caption_type == 'blip':
            # texts = []
            _cs = []
            conds = []
            for img_meta in img_metas:
                img_id = img_meta['ori_filename']
                text = self.blip_captions[img_id]['captions']
                c = self.sd_model.get_learned_conditioning(text)
                # texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
            c_crossattn = torch.cat(conds, dim=1)

        elif self.caption_type == 'meta_prompts':
            c_crossattn = self.meta_prompts[None, :, :].expand(img.shape[0], -1, -1)

        elif self.caption_type == 'clip_probs':
            batch_embs = []
            for img_meta in img_metas:
                img_id = img_meta['ori_filename']
                probs = torch.FloatTensor(self.clip_probs[img_id]['probs']).to(img.device)
                weighted_class_embs = probs.repeat(self.class_embeddings.shape[-1], 1).T * self.class_embeddings
                batch_embs.append(weighted_class_embs)

            c_crossattn = torch.stack(batch_embs)

        elif self.caption_type == 'clip_captions':
            _cs = []
            conds = []
            for img_meta in img_metas:
                img_id = img_meta['ori_filename']
                text = self.clip_captions[img_id]['captions']
                c = self.sd_model.get_learned_conditioning(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
            c_crossattn = torch.cat(conds, dim=1)

        return c_crossattn


    def extract_feat(self, img, img_metas):
        """Extract features from images."""

        with torch.no_grad():
            latents = self.encoder_vq.encode(img).mode().detach()

        if self.refine_step == 0:
            # c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma) # NOTE: here the c_crossattn should be expand_dim as latents
            c_crossattn = self.get_crossattn(latents, img, img_metas)
            t = torch.ones((img.shape[0],), device=img.device).long()
            outs = self.unet(latents, t, c_crossattn=[c_crossattn])
            print(c_crossattn.shape)
            print(outs[1].shape)
            print(latents.shape)
            return outs

        for i in range(self.refine_step):
            if isinstance(latents, list):
                latents = latents[0]
            if i > 0:
                cross_unet_conv = getattr(self, f"cross_unet_conv{i + 1}")
                latents = cross_unet_conv(latents)
            # c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma) # NOTE: here the c_crossattn should be expand_dim as latents
            c_crossattn = self.get_crossattn(latents, img, img_metas)

            t = getattr(self, f"t{i + 1}")
            t = t.repeat(img.shape[0], 1)
            latents = self.unet(latents, t, c_crossattn=[c_crossattn])
        outs = []
        for i in range(4):
            upconv = getattr(self, f"upconv{i + 1}")
            feat = latents[i]
            feat = self.up(feat)
            for j in upconv:
                feat = j(feat)
            feat = torch.einsum('bchw,bnc->bnhw', feat, c_crossattn)
            outs.append(feat)
        
        return outs


    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def compute_decoder_head_shapes(self):
        enc_16_channels = 640
        enc_32_channels = 1280

        if self.caption_type == 'blip':
            enc_16_channels += 77
            enc_32_channels += 77

        return enc_16_channels, enc_32_channels

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img, img_metas)

        if self.with_neck:
            x = self.neck(x)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, img_metas)
        if self.with_neck:
            x = list(self.neck(x))
        out = self._decode_head_forward_test(x, img_metas)  
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

