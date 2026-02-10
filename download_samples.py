"""
Download sample test images for TEDR object detection.
"""
import requests
from pathlib import Path
from PIL import Image
import io


def download_sample_images():
    """Download sample images from public sources."""
    
    output_dir = Path("data/sample_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Sample Image Download Helper")
    print("=" * 60)
    print("\nThis script helps you download sample images for testing.")
    print("\nYou can:")
    print("1. Add your own images to: data/sample_images/")
    print("2. Use any JPEG or PNG images of:")
    print("   - Indian road scenes")
    print("   - Traffic scenarios")
    print("   - Vehicles (cars, bikes, trucks, buses)")
    print("   - Pedestrians")
    print("   - Animals (especially cows)")
    print("\nSupported formats: JPEG, PNG")
    print("Recommended size: 800x600 or higher")
    print("\n" + "=" * 60)
    
    # Sample image URLs (using placeholder/example images)
    # In a real scenario, you would use actual URLs or local images
    sample_urls = {
        "sample_readme.txt": "Place your test images here"
    }
    
    readme_path = output_dir / "IMAGES_README.txt"
    readme_path.write_text("""
TEDR Sample Images Directory
=============================

Place your test images here for object detection.

Supported formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)

Recommended test images:
- Indian road scenes with vehicles
- Traffic scenarios with multiple vehicle types
- Auto rickshaws (for testing fine-tuned models)
- Pedestrians and cyclists
- Animals on roads (cows, dogs)
- Mixed traffic scenarios

Image guidelines:
- Resolution: 800x600 or higher recommended
- Good lighting and clarity
- Multiple objects for better testing
- Various scenarios (day/night, urban/rural)

Example sources for test images:
1. Your own photos of Indian roads
2. Public domain image databases
3. Creative Commons licensed images
4. Stock photo websites (check licensing)

Note: Ensure you have proper rights to use any images.
    """)
    
    print(f"\n✓ Created README at: {readme_path}")
    print(f"\nDirectory ready: {output_dir}")
    print("\nAdd your test images to this directory and run:")
    print("  python example_detect.py data/sample_images/your_image.jpg")
    print("\nOr use the web UI at:")
    print("  http://localhost:8000/static/index.html")


if __name__ == "__main__":
    download_sample_images()
