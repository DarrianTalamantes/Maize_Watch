from PIL import Image
import sys

def process_image(input_path, output_path):
    # Open the image
    image = Image.open(input_path)

    # Convert to grayscale
    grayscale_image = image.convert("L")

    # Save the grayscale image
    grayscale_image.save(output_path)

    # Print "grayscale" (this will be captured by Streamlit)
    print("grayscale")

if __name__ == "__main__":
    input_image_path = sys.argv[1]  # Input image path from command line
    output_image_path = sys.argv[2]  # Output grayscale image path

    process_image(input_image_path, output_image_path)
