"""Image preprocessing utilities for TEDR."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union


def resize_image(
    image: Union[Image.Image, np.ndarray],
    target_size: int = 800,
    keep_aspect_ratio: bool = True
) -> Union[Image.Image, np.ndarray]:
    """Resize image to target size.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size for the larger dimension
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image (same type as input)
    """
    is_pil = isinstance(image, Image.Image)
    
    if is_pil:
        width, height = image.size
    else:
        height, width = image.shape[:2]
    
    if keep_aspect_ratio:
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = target_size
    
    if is_pil:
        return image.resize((new_width, new_height), Image.LANCZOS)
    else:
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [0, 1] to [0, 255] range.
    
    Args:
        image: Normalized image
        
    Returns:
        Denormalized image as uint8
    """
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def convert_color(image: np.ndarray, conversion: str = 'BGR2RGB') -> np.ndarray:
    """Convert image color space.
    
    Args:
        image: Input image
        conversion: Color conversion type (e.g., 'BGR2RGB', 'RGB2BGR')
        
    Returns:
        Converted image
    """
    conversion_code = getattr(cv2, f'COLOR_{conversion}')
    return cv2.cvtColor(image, conversion_code)


def apply_augmentation(image: Image.Image, augmentation_type: str = 'none') -> Image.Image:
    """Apply data augmentation to image.
    
    Args:
        image: PIL Image
        augmentation_type: Type of augmentation ('none', 'flip', 'brightness', 'contrast')
        
    Returns:
        Augmented image
    """
    from PIL import ImageEnhance
    
    if augmentation_type == 'flip':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif augmentation_type == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)
    elif augmentation_type == 'contrast':
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)
    else:
        return image


def crop_image(
    image: Union[Image.Image, np.ndarray],
    bbox: Tuple[int, int, int, int]
) -> Union[Image.Image, np.ndarray]:
    """Crop image using bounding box.
    
    Args:
        image: PIL Image or numpy array
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped image (same type as input)
    """
    x1, y1, x2, y2 = bbox
    
    if isinstance(image, Image.Image):
        return image.crop((x1, y1, x2, y2))
    else:
        return image[y1:y2, x1:x2]


def pad_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int],
    pad_value: int = 0
) -> Union[Image.Image, np.ndarray]:
    """Pad image to target size.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size as (width, height)
        pad_value: Padding value
        
    Returns:
        Padded image (same type as input)
    """
    is_pil = isinstance(image, Image.Image)
    
    if is_pil:
        width, height = image.size
        target_width, target_height = target_size
        
        # Calculate padding
        pad_left = (target_width - width) // 2
        pad_top = (target_height - height) // 2
        pad_right = target_width - width - pad_left
        pad_bottom = target_height - height - pad_top
        
        # Apply padding
        from PIL import ImageOps
        return ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=pad_value)
    else:
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate padding
        pad_left = (target_width - width) // 2
        pad_top = (target_height - height) // 2
        pad_right = target_width - width - pad_left
        pad_bottom = target_height - height - pad_top
        
        # Apply padding
        if len(image.shape) == 3:
            return cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[pad_value] * image.shape[2]
            )
        else:
            return cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=pad_value
            )
