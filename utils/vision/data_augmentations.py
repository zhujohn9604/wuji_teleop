import torch
import torch.nn.functional as F

def random_crop(img, padding):
    """
    Randomly crop an image with padding.
    Args:
        img: Input image tensor of shape (H, W, C).
        padding: Padding size.
    Returns:
        Cropped image tensor of shape (H, W, C).
    """
    H, W, C = img.shape
    # Generate random crop start points
    crop_from_x = torch.randint(0, 2 * padding + 1, (1,), device=img.device)
    crop_from_y = torch.randint(0, 2 * padding + 1, (1,), device=img.device)

    # Pad the image
    padded_img = F.pad(img.permute(2, 0, 1), (padding, padding, padding, padding), mode='replicate')  # Pad (C, H, W)
    padded_img = padded_img.permute(1, 2, 0)  # Back to (H, W, C)

    # Perform the crop
    cropped_img = padded_img[crop_from_x:crop_from_x + H, crop_from_y:crop_from_y + W, :]

    return cropped_img

def batched_random_crop(img, padding, num_batch_dims: int = 1):
    """
    Apply random crop to a batch of images.
    Args:
        img: Input image tensor of shape (B, N, H, W, C).
        padding: Padding size.
        num_batch_dims: Number of batch dimensions.
    Returns:
        Cropped image tensor of shape (B, N, H, W, C).
    """
    # Flatten batch dims
    original_shape = img.shape
    img = img.view(-1, *img.shape[num_batch_dims:])  # Flatten to (B*N, H, W, C)

    # Apply random crop to each image in the batch
    cropped_imgs = []
    for i in range(img.shape[0]):
        cropped_img = random_crop(img[i], padding)
        cropped_imgs.append(cropped_img)

    # Stack the cropped images and restore batch dims
    cropped_imgs = torch.stack(cropped_imgs, dim=0)
    cropped_imgs = cropped_imgs.view(original_shape)

    return cropped_imgs