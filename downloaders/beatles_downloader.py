from utils import io_utils
import glob

def download():
	annotations_path="./data/annotations/The Beatles Annotations/The Beatles/"
	io_utils.download_albums("The Beatles",annotations_path,annotations_type='lab')	


