import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ClassifierService:
    """
    Service for loading transfer-learned EfficientNet-B0 PyTorch models
    and running inference for hairline and jawline classification.
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default to backend/models directory relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir

        # Shared image preprocessing pipeline (BGR→RGB, 224×224, ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Hairline Model (2 output classes: m_shaped, straight)
        self.hairline_model, hairline_names = self._load_model(
            filename="hairline_efficientnet_b0.pth",
            num_classes=2
        )
        self.hairline_classes = hairline_names or ['m_shaped', 'straight']

        # Jawline Model (3 output classes: angular, sharp, soft)
        self.jawline_model, jawline_names = self._load_model(
            filename="jawline_efficientnet_b0.pth",
            num_classes=3
        )
        self.jawline_classes = jawline_names or ['angular', 'sharp', 'soft']

    def _load_model(self, filename: str, num_classes: int):
        """
        Initializes efficientnet_b0, modifies the final classifier layer for num_classes,
        loads state_dict weights from models_dir with map_location='cpu', and sets to eval mode.

        Returns:
            tuple: (nn.Module, list or None) — loaded model and optional class names from checkpoint
        """
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        class_names = None
        model_path = os.path.join(self.models_dir, filename)

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    class_names = checkpoint.get('class_names')
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Failed to load model weights from {model_path}: {e}")
        else:
            print(f"Warning: Model file not found at {model_path}")

        model.eval()
        return model, class_names

    def preprocess_image(self, cv2_image: np.ndarray) -> torch.Tensor:
        """
        Shared image preprocessing method:
        - Converts BGR OpenCV image to RGB
        - Resizes to 224x224
        - Converts to PyTorch Tensor
        - Applies ImageNet normalization (mean/std)
        - Adds batch dimension [1, 3, 224, 224]
        """
        if cv2_image is None or not isinstance(cv2_image, np.ndarray) or cv2_image.size == 0:
            raise ValueError("Invalid or empty image array provided for preprocessing.")

        # Convert BGR OpenCV image to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Apply transformation pipeline
        tensor_image = self.transform(rgb_image)

        # Add batch dimension [1, 3, 224, 224]
        return tensor_image.unsqueeze(0)

    def predict_hairline(self, cv2_image: np.ndarray) -> tuple:
        """
        Preprocesses OpenCV image and runs inference on the Hairline EfficientNet-B0 model.

        Returns:
            tuple: (predicted_class: str, confidence: float)
        """
        predicted_class, confidence, _ = self.predict_hairline_detailed(cv2_image)
        return predicted_class, confidence

    def predict_hairline_detailed(self, cv2_image: np.ndarray) -> tuple:
        """
        Preprocesses OpenCV image and runs inference on the Hairline EfficientNet-B0 model.
        
        Returns:
            tuple: (predicted_class: str, confidence: float, probabilities: dict)
        """
        try:
            tensor_input = self.preprocess_image(cv2_image)
            with torch.no_grad():
                outputs = self.hairline_model(tensor_input)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

                idx = predicted_idx.item()
                conf = float(confidence.item())
                
                # Build probabilities dict
                probs_dict = {}
                for i, class_name in enumerate(self.hairline_classes):
                    probs_dict[class_name] = float(probabilities[0][i].item())
                
                # Check uncertainty threshold
                if conf < 0.60:
                    predicted_class = "uncertain"
                else:
                    predicted_class = self.hairline_classes[idx] if idx < len(self.hairline_classes) else "unknown"

                return predicted_class, conf, probs_dict
        except Exception as e:
            raise RuntimeError(f"Hairline prediction error: {str(e)}")

    def predict_jawline(self, cv2_image: np.ndarray) -> tuple:
        """
        Preprocesses OpenCV image and runs inference on the Jawline EfficientNet-B0 model.

        Returns:
            tuple: (predicted_class: str, confidence: float)
        """
        predicted_class, confidence, _ = self.predict_jawline_detailed(cv2_image)
        return predicted_class, confidence

    def predict_jawline_detailed(self, cv2_image: np.ndarray) -> tuple:
        """
        Preprocesses OpenCV image and runs inference on the Jawline EfficientNet-B0 model.
        
        Returns:
            tuple: (predicted_class: str, confidence: float, probabilities: dict)
        """
        try:
            tensor_input = self.preprocess_image(cv2_image)
            with torch.no_grad():
                outputs = self.jawline_model(tensor_input)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

                idx = predicted_idx.item()
                conf = float(confidence.item())
                
                # Build probabilities dict
                probs_dict = {}
                for i, class_name in enumerate(self.jawline_classes):
                    probs_dict[class_name] = float(probabilities[0][i].item())
                
                # Check uncertainty threshold
                if conf < 0.60:
                    predicted_class = "uncertain"
                else:
                    predicted_class = self.jawline_classes[idx] if idx < len(self.jawline_classes) else "unknown"

                return predicted_class, conf, probs_dict
        except Exception as e:
            raise RuntimeError(f"Jawline prediction error: {str(e)}")
