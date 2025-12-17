from PIL import Image, UnidentifiedImageError
from colorama import init, Fore, Style

init(autoreset=True)

def NormalizeImageFormat(img_path):
    try:
        # Primary: PIL
        return Image.open(img_path).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as e:
        # Fallback: ffmpeg
        print(f"{Fore.YELLOW}Warning: Input image format not recognized by PIL, converting input image to bitmap via ffmpeg instead.")
        # Todo: add ffmpeg support
        return
