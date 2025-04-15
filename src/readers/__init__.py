from .utils import parse_multiple_files
from .extractor import FileExtractor
# document = parse_multiple_files(
#         str(file_path),
#         extractor=file_extractor.get_extractor_for_file(file_path),
#     ) text, metadata
__all__=["parse_multiple_files","FileExtractor"]