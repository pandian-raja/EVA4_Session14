def show(tensors, figsize=(10, 10), *args, **kwargs):
    try:
        tensors = tensors.detach().cpu()
    except:
        pass
    grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
    grid_image = grid_tensor.permute(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])

    plt.show()

def show_pred(tensors, *args, **kwargs):
    mean, std = torch.tensor([0.485, 0.456, 0.406])*255, torch.tensor([0.229, 0.224, 0.225])*255
    tensors = (tensors * std[None, :, None, None]) + mean[None, :, None, None]
    show(tensors, *args, **kwargs)
