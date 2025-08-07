"""Vision processor for analyzing images in repositories."""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class ImageMetadata(BaseModel):
    """Image metadata information."""
    
    path: str
    format: str
    size: Tuple[int, int]
    file_size: int
    has_transparency: bool = False
    color_mode: str = ""
    is_animated: bool = False


class ProcessedImage(BaseModel):
    """Processed image with analysis data."""
    
    metadata: ImageMetadata
    base64_data: str
    thumbnail_base64: Optional[str] = None
    technical_score: float = 0.0
    detected_elements: List[str] = []


class VisionProcessingError(Exception):
    """Vision processing error."""
    pass


class VisionProcessor:
    """Processes and analyzes images from repositories."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.max_image_size = self.settings.vision.max_image_size
        self.supported_formats = self.settings.vision.supported_image_formats
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if image format is supported."""
        return self.settings.is_image_file(filename)
    
    def preprocess_image(self, image_data: bytes, image_path: str) -> Optional[ProcessedImage]:
        """Preprocess image for analysis."""
        try:
            # Check file size
            if len(image_data) > self.max_image_size:
                logger.warning(f"Image {image_path} too large: {len(image_data)} bytes")
                return None
            
            # Open image with PIL
            image = Image.open(BytesIO(image_data))
            
            # Get metadata
            metadata = self._extract_metadata(image, image_path, len(image_data))
            
            # Convert to RGB if necessary (for JPEG compatibility)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Preserve transparency info
                has_transparency = True
                # Convert to RGB with white background for analysis
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
                metadata.has_transparency = has_transparency
            
            # Resize if too large for processing
            max_dimension = 1024
            if max(image.size) > max_dimension:
                image = self._resize_image(image, max_dimension)
            
            # Convert to base64
            base64_data = self._image_to_base64(image, 'JPEG')
            
            # Create thumbnail
            thumbnail_base64 = self._create_thumbnail(image)
            
            # Calculate technical relevance score
            technical_score = self._calculate_technical_score(image_path, metadata)
            
            # Detect basic elements
            detected_elements = self._detect_basic_elements(image, image_path)
            
            return ProcessedImage(
                metadata=metadata,
                base64_data=base64_data,
                thumbnail_base64=thumbnail_base64,
                technical_score=technical_score,
                detected_elements=detected_elements
            )
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            return None
    
    def _extract_metadata(self, image: Image.Image, image_path: str, file_size: int) -> ImageMetadata:
        """Extract metadata from image."""
        return ImageMetadata(
            path=image_path,
            format=image.format or "UNKNOWN",
            size=(image.width, image.height),
            file_size=file_size,
            has_transparency=image.mode in ('RGBA', 'LA', 'P'),
            color_mode=image.mode,
            is_animated=getattr(image, 'is_animated', False)
        )
    
    def _resize_image(self, image: Image.Image, max_dimension: int) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        width, height = image.size
        
        if width > height:
            new_width = max_dimension
            new_height = int((height * max_dimension) / width)
        else:
            new_height = max_dimension
            new_width = int((width * max_dimension) / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _image_to_base64(self, image: Image.Image, format: str = 'JPEG') -> str:
        """Convert PIL image to base64 string."""
        buffer = BytesIO()
        
        # Ensure RGB mode for JPEG
        if format == 'JPEG' and image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(buffer, format=format, quality=85, optimize=True)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (200, 200)) -> str:
        """Create thumbnail of image."""
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            return self._image_to_base64(thumbnail, 'JPEG')
        except Exception as e:
            logger.warning(f"Failed to create thumbnail: {e}")
            return ""
    
    def _calculate_technical_score(self, image_path: str, metadata: ImageMetadata) -> float:
        """Calculate technical relevance score for image."""
        score = 0.0
        path_lower = image_path.lower()
        
        # Path-based scoring
        technical_keywords = {
            'architecture': 0.9,
            'diagram': 0.8,
            'design': 0.7,
            'flow': 0.8,
            'schema': 0.8,
            'wireframe': 0.7,
            'mockup': 0.6,
            'ui': 0.6,
            'ux': 0.6,
            'system': 0.7,
            'component': 0.7,
            'api': 0.8,
            'database': 0.8,
            'er': 0.8,  # Entity-Relationship
            'uml': 0.9,
            'sequence': 0.8,
            'class': 0.8
        }
        
        # Check for technical keywords in path
        for keyword, weight in technical_keywords.items():
            if keyword in path_lower:
                score = max(score, weight)
        
        # Format-based adjustments
        if metadata.format in ['SVG', 'PNG']:
            score += 0.1  # These formats are common for diagrams
        
        # Size-based adjustments
        width, height = metadata.size
        if width > 800 or height > 600:
            score += 0.1  # Larger images might be more detailed diagrams
        
        # Aspect ratio considerations
        aspect_ratio = width / height if height > 0 else 1
        if 0.5 <= aspect_ratio <= 2.0:
            score += 0.05  # Reasonable aspect ratios for technical content
        
        # Directory-based scoring
        path_parts = Path(image_path).parts
        technical_dirs = ['docs', 'documentation', 'design', 'assets', 'images', 'diagrams']
        for part in path_parts:
            if part.lower() in technical_dirs:
                score += 0.1
                break
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _detect_basic_elements(self, image: Image.Image, image_path: str) -> List[str]:
        """Detect basic elements in image using simple heuristics."""
        elements = []
        
        # Analyze file name for clues
        filename = Path(image_path).stem.lower()
        
        # Common diagram types based on filename
        if any(keyword in filename for keyword in ['architecture', 'arch']):
            elements.append('architecture_diagram')
        if any(keyword in filename for keyword in ['flow', 'workflow']):
            elements.append('flowchart')
        if any(keyword in filename for keyword in ['ui', 'mockup', 'wireframe']):
            elements.append('ui_design')
        if any(keyword in filename for keyword in ['schema', 'er', 'database']):
            elements.append('database_schema')
        if any(keyword in filename for keyword in ['sequence', 'uml']):
            elements.append('uml_diagram')
        if any(keyword in filename for keyword in ['component', 'system']):
            elements.append('system_diagram')
        if any(keyword in filename for keyword in ['api', 'endpoint']):
            elements.append('api_documentation')
        
        # Basic image analysis
        try:
            width, height = image.size
            
            # Aspect ratio analysis
            aspect_ratio = width / height
            if aspect_ratio > 2:
                elements.append('wide_layout')
            elif aspect_ratio < 0.5:
                elements.append('tall_layout')
            else:
                elements.append('standard_layout')
            
            # Size categories
            if width * height > 500000:  # ~700x700
                elements.append('high_resolution')
            elif width * height < 50000:  # ~220x220
                elements.append('low_resolution')
            else:
                elements.append('medium_resolution')
            
            # Color analysis
            if image.mode == 'L':
                elements.append('grayscale')
            elif image.mode in ['RGB', 'RGBA']:
                elements.append('color')
                
                # Simple color distribution analysis
                if self._is_mostly_monochrome(image):
                    elements.append('monochrome_content')
                else:
                    elements.append('colorful_content')
        
        except Exception as e:
            logger.warning(f"Basic element detection failed for {image_path}: {e}")
        
        return elements
    
    def _is_mostly_monochrome(self, image: Image.Image) -> bool:
        """Check if image is mostly monochromatic using simple sampling."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Sample pixels from center area
            width, height = image.size
            sample_size = min(100, width // 4, height // 4)
            
            center_x, center_y = width // 2, height // 2
            sample_box = (
                center_x - sample_size // 2,
                center_y - sample_size // 2,
                center_x + sample_size // 2,
                center_y + sample_size // 2
            )
            
            sample_region = image.crop(sample_box)
            pixels = list(sample_region.getdata())
            
            # Calculate color variance
            if len(pixels) == 0:
                return True
            
            # Check if most pixels have similar R, G, B values (grayscale-like)
            grayscale_count = 0
            for r, g, b in pixels[:100]:  # Sample first 100 pixels
                # If R, G, B values are close, it's grayscale-like
                if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
                    grayscale_count += 1
            
            return grayscale_count > len(pixels[:100]) * 0.7  # 70% threshold
            
        except Exception:
            return False
    
    def batch_process_images(self, images_data: Dict[str, bytes]) -> Dict[str, ProcessedImage]:
        """Process multiple images in batch."""
        processed_images = {}
        
        logger.info(f"Processing {len(images_data)} images")
        
        for image_path, image_data in images_data.items():
            if not self.is_supported_format(image_path):
                logger.warning(f"Unsupported image format: {image_path}")
                continue
            
            processed = self.preprocess_image(image_data, image_path)
            if processed:
                processed_images[image_path] = processed
                logger.debug(f"Processed image: {image_path} (score: {processed.technical_score:.2f})")
            else:
                logger.warning(f"Failed to process image: {image_path}")
        
        # Sort by technical relevance score
        sorted_images = dict(sorted(
            processed_images.items(),
            key=lambda x: x[1].technical_score,
            reverse=True
        ))
        
        logger.info(f"Successfully processed {len(processed_images)} images")
        return sorted_images
    
    def filter_by_relevance(
        self,
        processed_images: Dict[str, ProcessedImage],
        min_score: float = 0.3,
        max_count: Optional[int] = None
    ) -> Dict[str, ProcessedImage]:
        """Filter images by technical relevance score."""
        filtered = {
            path: img for path, img in processed_images.items()
            if img.technical_score >= min_score
        }
        
        if max_count and len(filtered) > max_count:
            # Keep top scoring images
            sorted_items = sorted(
                filtered.items(),
                key=lambda x: x[1].technical_score,
                reverse=True
            )
            filtered = dict(sorted_items[:max_count])
        
        logger.info(f"Filtered to {len(filtered)} relevant images (min_score: {min_score})")
        return filtered
    
    def generate_image_summary(self, processed_images: Dict[str, ProcessedImage]) -> Dict[str, Any]:
        """Generate summary of processed images."""
        if not processed_images:
            return {"total": 0, "formats": {}, "avg_score": 0.0, "categories": {}}
        
        formats = {}
        scores = []
        elements = {}
        
        for img in processed_images.values():
            # Count formats
            fmt = img.metadata.format
            formats[fmt] = formats.get(fmt, 0) + 1
            
            # Collect scores
            scores.append(img.technical_score)
            
            # Count elements
            for element in img.detected_elements:
                elements[element] = elements.get(element, 0) + 1
        
        return {
            "total": len(processed_images),
            "formats": formats,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "categories": elements,
            "high_relevance_count": len([s for s in scores if s > 0.7]),
            "medium_relevance_count": len([s for s in scores if 0.3 <= s <= 0.7]),
            "low_relevance_count": len([s for s in scores if s < 0.3])
        }