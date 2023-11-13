from image_processor import imageProcessor, viewSegmentation
from nd2reader import ND2Reader
from skimage import io



# Replace 'your_file.nd2' with the path to your ND2 file
nd2_file_path = 'your_file.nd2'

# Open the ND2 file
with ND2Reader(nd2_file_path) as images:
    # Iterate through the frames in the ND2 file
    for frame in images:
        # Convert each frame to a TIFF image
        tiff_image = frame.astype('uint16')  # Ensure the data type is suitable for TIFF
        # Replace 'output_directory' with the directory where you want to save the TIFF images
        io.imsave(f'output_directory/frame_{frame.frame_number}.tiff', tiff_image)


#put
