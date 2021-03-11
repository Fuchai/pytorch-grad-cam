import argparse

import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms


class ModelWrapper:
    def __init__(self, model, feature_module):
        self.model = model
        self.feature_module = feature_module
        self.feature_gradients = None
        self.feature_output = None
        self.register_hooks()

    def register_hooks(self):
        target_layer = next(reversed(self.feature_module._modules))
        target_layer = self.feature_module._modules[target_layer]
        target_layer.register_backward_hook(self.save_gradient)
        target_layer.register_forward_hook(self.save_output)

    def save_gradient(self, module, grad_input, grad_output):
        self.feature_gradients = grad_input[0]

    def save_output(self, module, input, output):
        self.feature_output = output

    def __call__(self, x):
        self.feature_gradients = None
        self.feature_output = None
        return self.model(x)


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()

        self.model_wrapper = ModelWrapper(self.model, self.feature_module)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        output = self.model_wrapper(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = one_hot.to(input_img.device)

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.model_wrapper.feature_gradients.cpu().data.numpy()
        features = self.model_wrapper.feature_output

        features = features[-1].cpu().data.numpy()

        global_average_pooled_gradients = np.mean(grads_val, axis=(2, 3))[0, :]

        cam = np.expand_dims(global_average_pooled_gradients, axis=(1, 2)) * features
        cam = cam.sum(axis=0)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = input_img * positive_mask
        self.save_for_backward(positive_mask)
        return output

    @staticmethod
    def backward(self, grad_output):
        positive_mask_1 = self.saved_tensors[0]
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = grad_output * positive_mask_1 * positive_mask_2
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.recursive_relu_apply(self.model)

    def recursive_relu_apply(self, module_top):
        # replace ReLU with GuidedBackpropReLU
        for idx, module in module_top._modules.items():
            self.recursive_relu_apply(module)
            if module.__class__.__name__ == 'ReLU':
                module_top._modules[idx] = GuidedBackpropReLU.apply

    def __call__(self, input_img, target_category=None):
        input_img.requires_grad = True
        input_img.retain_grad()

        output = self.model(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = one_hot.to(input_img.device)

        one_hot = torch.sum(one_hot * output)
        one_hot.backward()

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    print(f"Device {args.device}")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for ResNet50 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    model = models.resnet50(pretrained=True).to(args.device)
    grad_cam = GradCam(model=model, feature_module=model.layer4)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img).to(args.device)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model)
    gb = gb_model(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite("grad_cam.jpg", cam)
    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('grad_cam_gb.jpg', cam_gb)
