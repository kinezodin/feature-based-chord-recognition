from utils import io_utils
import glob

def download():
	annotations_path="./data/annotations/Robbie Williams Annotations/"
	io_utils.download_albums("Robbie Williams",annotations_path,annotations_type='chords')	


