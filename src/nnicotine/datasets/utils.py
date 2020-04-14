import os

# def download_and_extract_archive(url, download_root, extract_root=None, filename=None, md5=None, remove_finished=False):
#
#     download_root = os.path.expanduser(download_root)
#     if extract_root is None:
#         extract_root = download_root
#
#     if not filename:
#         filename = os.path.basename(url)
#
#     download_url(url, download_root, filename, md5)
#
#     archive = os.path.join(download_root, filename)
#
#     print("Extracting {} to {}".format(archive, extract_root))
#     extract_archive(archive, extract_root, remove_finished)
#
#     return
#
# def download_url(url, download_root, filename, md5):
#     return
#
# def extract_archive(archive, extract_root, remove_finished=False):
#     return
