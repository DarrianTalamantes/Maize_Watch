import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from datetime import datetime

def setup_logging(log_dir=None):
    """
    Set up a centralized logging configuration
    
    Args:
    - log_dir: Optional directory to save log files
    
    Returns:
    - Configured logger
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'gradcam_explanations.log'))
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class GradCAMExplainer:
    def __init__(self, model, target_layer, labels_for_viz, device, log_dir=None):
        """
        Initialize Grad-CAM Explainer with comprehensive saving capabilities
        
        Args:
        - model: Trained PyTorch model
        - target_layer: The layer to extract feature maps from
        - labels_for_viz: Dictionary mapping class indices to labels
        - device: Torch device
        - log_dir: Directory to save logs and visualizations
        """
        # Setup logging
        self.logger = logging.getLogger()
        # self.logger.setLevel(logging.INFO)
        
        # Create log directory if not exists
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), 'grad_cam_logs', 
                                   datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(log_dir, exist_ok=True)
        
        # Ensure directories for different visualization types
        self.original_dir = os.path.join(log_dir, 'original_images')
        self.gradcam_dir = os.path.join(log_dir, 'gradcam_images')
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.gradcam_dir, exist_ok=True)
        
        # File handler for logging
        file_handler = logging.FileHandler(os.path.join(log_dir, 'gradcam_explanations.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Explainer setup
        self.model = model
        self.target_layer = target_layer
        self.labels_for_viz = labels_for_viz
        self.device = device
        self.log_dir = log_dir
        
        # Grad-CAM tracking
        self.gradients = None
        self.activations = None

        self.orginal_filename = os.path.join(self.original_dir, "image.png")
        self.gradcam_filename = os.path.join(self.gradcam_dir, "grad.png")
        # Register hooks
        self.hook_layers()
        
        self.logger.info(f"GradCAM Explainer initialized")
        self.logger.info(f"Log directory: {log_dir}")
        self.logger.info(f"Labels: {labels_for_viz}")

    def hook_layers(self):
        """
        Registers forward and backward hooks to capture gradients and activations
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Attach hooks to the target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
        self.logger.info("Hooks registered for target layer")

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM visualization
        
        Args:
        - input_tensor: Input image tensor
        - target_class: Class to generate CAM for (defaults to predicted class)
        
        Returns:
        - CAM visualization
        - Predicted class
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax().item()
        
        # Log prediction details
        pred_prob = F.softmax(output, dim=1)
        top_k_probs, top_k_classes = torch.topk(pred_prob, k=3)
        
        self.logger.info(f"Top 3 Predictions:")
        for prob, cls in zip(top_k_probs[0], top_k_classes[0]):
            class_name = self.labels_for_viz[cls.item()]
            self.logger.info(f"  - {class_name}: {prob.item():.4f}")
        
        # Backward pass
        one_hot = F.one_hot(torch.tensor([target_class]), num_classes=output.size(-1)).float().to(input_tensor.device)
        loss = (output * one_hot).sum()
        loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # Generate heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap, target_class

    def plot_grad_cam(self, input_tensor, true_label, save_prefix='image'):
        """
        Plot and save Grad-CAM visualization
        
        Args:
        - input_tensor: Input image tensor
        - true_label: True label of the image
        - save_prefix: Prefix for saved image files
        
        Returns:
        - Predicted class label
        """
        # Generate CAM
        cam, pred_class = self.generate_cam(input_tensor)
        true_label_name = self.labels_for_viz[true_label]
        pred_label = self.labels_for_viz[pred_class]
        
        # Convert input tensor to image
        # Reverse normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = input_tensor.squeeze().cpu()
        img = img * std[:, None, None] + mean[:, None, None]
        img = img.permute(1, 2, 0).numpy()

            # Ensure cam is on CPU and convert to numpy
        cam = cam.cpu()
        
        # Resize CAM to match image size
        cam_resized = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), 
            size=(img.shape[0], img.shape[1]), 
            mode='bilinear', 
            align_corners=False
        ).squeeze().numpy()
        
        # Construct filenames
        original_filename = os.path.join(
            "output", 
            f'image.png'
        )
        gradcam_filename = os.path.join(
            "output", 
            f'gradcam.png'
        )
        
        # Save original image
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f'Original Image\nTrue: {true_label_name}, Predicted: {pred_label}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(original_filename)
        plt.close()
        
        # Save Grad-CAM visualization
        plt.figure(figsize=(12, 5))
        
        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Original Image\nTrue: {true_label_name}')
        plt.axis('off')
        
        # CAM Overlay
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM\nPredicted: {pred_label}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(gradcam_filename)
        plt.close()
        
        # Log file paths
        self.logger.info(f"Original image saved: {original_filename}")
        self.logger.info(f"Grad-CAM visualization saved: {gradcam_filename}")
        
        return pred_label

def visualize_explanations(
    model, 
    test_dataloader, 
    labels_for_viz, 
    device, 
    base_path, 
    num_images=5
):
    """
    Visualize Grad-CAM explanations for multiple images with comprehensive logging
    
    Args:
    - model: Trained PyTorch model
    - test_dataloader: DataLoader for test set
    - labels_for_viz: Dictionary mapping class indices to labels
    - device: Torch device
    - base_path: Base path for saving logs
    - num_images: Number of images to explain
    """
    # Setup logging
    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    
    # Create log directory 
    log_dir = os.path.join(base_path, 'explanation_logs', 
                           datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler for logging
    file_handler = logging.FileHandler(os.path.join(log_dir, 'explanations_summary.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("Starting Grad-CAM Explanation Visualization")
    logger.info(f"Number of images to explain: {num_images}")
    
    # Use the last convolutional layer for Grad-CAM
    # For ResNeXt, this would typically be the last layer of the base model
    target_layer = model.base[-1]
    
    # Create Grad-CAM object
    grad_cam = GradCAMExplainer(
        model=model, 
        target_layer=target_layer, 
        labels_for_viz=labels_for_viz, 
        device=device,
        log_dir=log_dir
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Tracking misclassifications
    misclassification_count = 0
    total_explained = 0
    
    # Visualize explanations
    for i, (inputs, labels) in enumerate(test_dataloader):
        if total_explained >= num_images:
            break
        
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
        
        # Process each image in the batch
        for j in range(inputs.size(0)):
            if total_explained >= num_images:
                break
            
            # Check if prediction matches ground truth
            input_tensor = inputs[j].unsqueeze(0)
            true_label = labels[j].item()
            pred_label = preds[j].item()
            
            # Log prediction details
            logger.info(f"\n--- Image {total_explained + 1} ---")
            logger.info(f"True Label: {labels_for_viz[true_label]}")
            logger.info(f"Predicted Label: {labels_for_viz[pred_label]}")
            
            # Generate Grad-CAM visualization
            explained_pred = grad_cam.plot_grad_cam(
                input_tensor, 
                true_label,
                save_prefix=f'image_{total_explained}'
            )
            
            # Track misclassifications
            if true_label != pred_label:
                misclassification_count += 1
                logger.warning("MISCLASSIFICATION DETECTED")
            
            total_explained += 1
    
    # Summary logging
    logger.info("\n--- Explanation Summary ---")
    logger.info(f"Total images explained: {total_explained}")
    logger.info(f"Misclassifications: {misclassification_count}")
    logger.info(f"Misclassification Rate: {misclassification_count/total_explained * 100:.2f}%")
    logger.info(f"Detailed logs saved in: {log_dir}")

# Usage remains the same as before
# visualize_explanations(
#     model=test_model_instance, 
#     test_dataloader=test_dataloader, 
#     labels_for_viz=labels_for_viz, 
#     device=device,
#     base_path=base_path,
#     num_images=5  # Number of images to explain
# )














# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import logging
# import os
# from datetime import datetime

# class GradCAMExplainer:
#     def __init__(self, model, target_layer, labels_for_viz, device, log_dir=None):
#         """
#         Initialize Grad-CAM Explainer with logging capabilities
        
#         Args:
#         - model: Trained PyTorch model
#         - target_layer: The layer to extract feature maps from
#         - labels_for_viz: Dictionary mapping class indices to labels
#         - device: Torch device
#         - log_dir: Directory to save logs and visualizations
#         """
#         # Setup logging
#         self.logger = logging.getLogger('GradCAM_Explainer')
#         self.logger.setLevel(logging.INFO)
        
#         # Create log directory if not exists
#         if log_dir is None:
#             log_dir = os.path.join(os.getcwd(), 'grad_cam_logs', 
#                                    datetime.now().strftime('%Y%m%d_%H%M%S'))
#         os.makedirs(log_dir, exist_ok=True)
        
#         # File handler for logging
#         file_handler = logging.FileHandler(os.path.join(log_dir, 'gradcam_explanations.log'))
#         file_handler.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
#         file_handler.setFormatter(formatter)
#         self.logger.addHandler(file_handler)
        
#         # Console handler
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
#         console_handler.setFormatter(formatter)
#         self.logger.addHandler(console_handler)
        
#         # Explainer setup
#         self.model = model
#         self.target_layer = target_layer
#         self.labels_for_viz = labels_for_viz
#         self.device = device
#         self.log_dir = log_dir
        
#         # Grad-CAM tracking
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         self.hook_layers()
        
#         self.logger.info(f"GradCAM Explainer initialized")
#         self.logger.info(f"Log directory: {log_dir}")
#         self.logger.info(f"Labels: {labels_for_viz}")

#     def hook_layers(self):
#         """
#         Registers forward and backward hooks to capture gradients and activations
#         """
#         def forward_hook(module, input, output):
#             self.activations = output.detach()

#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()

#         # Attach hooks to the target layer
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)
        
#         self.logger.info("Hooks registered for target layer")

#     def generate_cam(self, input_tensor, target_class=None):
#         """
#         Generate Grad-CAM visualization
        
#         Args:
#         - input_tensor: Input image tensor
#         - target_class: Class to generate CAM for (defaults to predicted class)
        
#         Returns:
#         - CAM visualization
#         - Predicted class
#         """
#         # Ensure model is in evaluation mode
#         self.model.eval()
        
#         # Forward pass
#         self.model.zero_grad()
#         output = self.model(input_tensor)
        
#         # Determine target class
#         if target_class is None:
#             target_class = output.argmax().item()
        
#         # Log prediction details
#         pred_prob = F.softmax(output, dim=1)
#         top_k_probs, top_k_classes = torch.topk(pred_prob, k=3)
        
#         self.logger.info(f"Top 3 Predictions:")
#         for prob, cls in zip(top_k_probs[0], top_k_classes[0]):
#             class_name = self.labels_for_viz[cls.item()]
#             self.logger.info(f"  - {class_name}: {prob.item():.4f}")
        
#         # Backward pass
#         one_hot = F.one_hot(torch.tensor([target_class]), num_classes=output.size(-1)).float().to(input_tensor.device)
#         loss = (output * one_hot).sum()
#         loss.backward()
        
#         # Get gradients and activations
#         gradients = self.gradients
#         activations = self.activations
        
#         # Global average pooling of gradients
#         pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
#         # Weight activations
#         for i in range(activations.size(1)):
#             activations[:, i, :, :] *= pooled_gradients[i]
        
#         # Generate heatmap
#         heatmap = torch.mean(activations, dim=1).squeeze()
#         heatmap = F.relu(heatmap)
#         heatmap /= torch.max(heatmap)
        
#         return heatmap, target_class

#     def plot_grad_cam(self, input_tensor, save_prefix='gradcam'):
#         """
#         Plot and save Grad-CAM visualization
        
#         Args:
#         - input_tensor: Input image tensor
#         - save_prefix: Prefix for saved image files
        
#         Returns:
#         - Predicted class label
#         """
#         # Generate CAM
#         cam, pred_class = self.generate_cam(input_tensor)
#         pred_label = self.labels_for_viz[pred_class]
        
#         # Convert input tensor to image
#         # Reverse normalization
#         mean = torch.tensor([0.485, 0.456, 0.406])
#         std = torch.tensor([0.229, 0.224, 0.225])
#         img = input_tensor.squeeze().cpu()
#         img = img * std[:, None, None] + mean[:, None, None]
#         img = img.permute(1, 2, 0).numpy()
        
#         # Resize CAM to match image size
#         cam_resized = F.interpolate(
#             cam.unsqueeze(0).unsqueeze(0), 
#             size=(img.shape[0], img.shape[1]), 
#             mode='bilinear', 
#             align_corners=False
#         ).squeeze().numpy()
        
#         # Plot
#         plt.figure(figsize=(12, 5))
        
#         # Original Image
#         plt.subplot(1, 2, 1)
#         plt.imshow(img)
#         plt.title(f'Original Image\nPredicted: {pred_label}')
#         plt.axis('off')
        
#         # CAM Overlay
#         plt.subplot(1, 2, 2)
#         plt.imshow(img)
#         plt.imshow(cam_resized, cmap='jet', alpha=0.5)
#         plt.title(f'Grad-CAM\nPredicted: {pred_label}')
#         plt.axis('off')
        
#         plt.tight_layout()
        
#         # Create save paths
#         os.makedirs(self.log_dir, exist_ok=True)
#         cam_save_path = os.path.join(self.log_dir, f'{save_prefix}_gradcam.png')
        
#         # Save figure
#         plt.savefig(cam_save_path)
#         plt.close()
        
#         # Log file paths
#         self.logger.info(f"Grad-CAM visualization saved: {cam_save_path}")
        
#         return pred_label

# def visualize_explanations(
#     model, 
#     test_dataloader, 
#     labels_for_viz, 
#     device, 
#     base_path, 
#     num_images=5
# ):
#     """
#     Visualize Grad-CAM explanations for multiple images with comprehensive logging
    
#     Args:
#     - model: Trained PyTorch model
#     - test_dataloader: DataLoader for test set
#     - labels_for_viz: Dictionary mapping class indices to labels
#     - device: Torch device
#     - base_path: Base path for saving logs
#     - num_images: Number of images to explain
#     """
#     # Setup logging
#     logger = logging.getLogger('Explanation_Visualizer')
#     logger.setLevel(logging.INFO)
    
#     # Create log directory 
#     log_dir = os.path.join(base_path, 'explanation_logs', 
#                            datetime.now().strftime('%Y%m%d_%H%M%S'))
#     os.makedirs(log_dir, exist_ok=True)
    
#     # File handler for logging
#     file_handler = logging.FileHandler(os.path.join(log_dir, 'explanations_summary.log'))
#     file_handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
    
#     # Console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(formatter)
#     logger.addHandler(console_handler)
    
#     logger.info("Starting Grad-CAM Explanation Visualization")
#     logger.info(f"Number of images to explain: {num_images}")
    
#     # Use the last convolutional layer for Grad-CAM
#     # For ResNeXt, this would typically be the last layer of the base model
#     target_layer = model.base[-1]
    
#     # Create Grad-CAM object
#     grad_cam = GradCAMExplainer(
#         model=model, 
#         target_layer=target_layer, 
#         labels_for_viz=labels_for_viz, 
#         device=device,
#         log_dir=log_dir
#     )
    
#     # Set model to evaluation mode
#     model.eval()
    
#     # Tracking misclassifications
#     misclassification_count = 0
#     total_explained = 0
    
#     # Visualize explanations
#     for i, (inputs, labels) in enumerate(test_dataloader):
#         if total_explained >= num_images:
#             break
        
#         # Move to device
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         # Run prediction
#         with torch.no_grad():
#             outputs = model(inputs)
#             preds = outputs.argmax(dim=1)
        
#         # Process each image in the batch
#         for j in range(inputs.size(0)):
#             if total_explained >= num_images:
#                 break
            
#             # Check if prediction matches ground truth
#             input_tensor = inputs[j].unsqueeze(0)
#             true_label = labels[j].item()
#             pred_label = preds[j].item()
            
#             # Log prediction details
#             logger.info(f"\n--- Image {total_explained + 1} ---")
#             logger.info(f"True Label: {labels_for_viz[true_label]}")
#             logger.info(f"Predicted Label: {labels_for_viz[pred_label]}")
            
#             # Generate Grad-CAM visualization
#             explained_pred = grad_cam.plot_grad_cam(
#                 input_tensor, 
#                 save_prefix=f'image_{total_explained}'
#             )
            
#             # Track misclassifications
#             if true_label != pred_label:
#                 misclassification_count += 1
#                 logger.warning("MISCLASSIFICATION DETECTED")
            
#             total_explained += 1
    
#     # Summary logging
#     logger.info("\n--- Explanation Summary ---")
#     logger.info(f"Total images explained: {total_explained}")
#     logger.info(f"Misclassifications: {misclassification_count}")
#     logger.info(f"Misclassification Rate: {misclassification_count/total_explained * 100:.2f}%")
#     logger.info(f"Detailed logs saved in: {log_dir}")

# # Modify your existing main script to include this line after testing the model
# # At the end of your __main__ block, add:
# # visualize_explanations(
# #     model=test_model_instance, 
# #     test_dataloader=test_dataloader, 
# #     labels_for_viz=labels_for_viz, 
# #     device=device,
# #     base_path=base_path,
# #     num_images=5  # Number of images to explain
# # )