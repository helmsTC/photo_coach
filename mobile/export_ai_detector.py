"""Export the AI-image detector used by the iOS app (ios/PhotoCoach/Resources/AIDetector.mlpackage).

Model: Community Forensics ViT-S/16-384 (Park & Owens, CVPR 2025), weights OwensLab/commfor-model-384, MIT license.
Input 'image' is RGB 0-255 [1,3,384,384] (shorter side resized to 440, centre crop 384); ImageNet
normalisation and the sigmoid are baked in, output 'ai' is P(AI-generated).  FP16: 8-bit palettisation
and int8 weight quantisation both changed verdicts on a test set, so they are not used.

    pip install timm coremltools safetensors huggingface_hub
    python mobile/export_ai_detector.py ios/PhotoCoach/Resources
"""
import sys, torch, timm, numpy as np, coremltools as ct
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
vit = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=False); vit.head = torch.nn.Linear(384, 1)
vit.load_state_dict({k.removeprefix('vit.'): v for k, v in load_file(hf_hub_download('OwensLab/commfor-model-384', 'model.safetensors')).items()}); vit.eval()
class W(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
        s.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        s.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
    def forward(s, x):          # x: RGB 0-255, [1,3,384,384]
        return torch.sigmoid(s.m((x - s.mean) / s.std))
w = W(vit).eval()
ex = torch.rand(1, 3, 384, 384) * 255
tr = torch.jit.trace(w, ex)
m = ct.convert(tr, inputs=[ct.TensorType(name='image', shape=(1, 3, 384, 384), dtype=np.float32)],
               outputs=[ct.TensorType(name='ai')], minimum_deployment_target=ct.target.iOS17,
               compute_precision=ct.precision.FLOAT16, convert_to='mlprogram')
m.short_description = 'AI-generated image detector (Community Forensics ViT-S/16-384, Park & Owens, CVPR 2025). MIT license.'
m.author = 'Jeongsoo Park, Andrew Owens; converted for ApertureIQ'
m.license = 'MIT'
m.input_description['image'] = 'RGB 0-255, shorter side resized to 440 then centre-cropped to 384'
m.output_description['ai'] = 'Probability the image is AI-generated'
m.save(sys.argv[1] + '/AIDetector.mlpackage')
with torch.no_grad(): print('torch', w(ex).item(), 'fp16', m.predict({'image': ex.numpy()})['ai'].item())
