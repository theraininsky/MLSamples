from .src.ImageUtil import NormalizeImageFormat
from .src.impl import *

def run_gradcam(image_paths, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval().to(device)

    # Last convolution layer of ResNet-18
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    for img_path in image_paths:

        if (img := NormalizeImageFormat(img_path)) is None:
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)

        cam, class_idx = gradcam.generate(input_tensor)

        overlay = overlay_heatmap(img, cam)

        # save as *.jpg file
        filename: str = Path(img_path).stem + '.jpg'
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"Saved Grad-CAM for {filename} (class {class_idx})")
